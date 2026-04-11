/*
 * CUDA kernel for paged attention computation. Works for single sequence.
 *
 * Implements the core paged attention operation:
 *
 *   out = softmax(Q @ K^T / sqrt(head_dim)) @ V
 *
 * where K and V are stored non-contiguously in a paged physical pool.
 * K/V tokens are accessed on-the-fly via (phys_block_id, slot_offset) pairs,
 * eliminating the need for a separate gather step before attention.
 *
 * Kernel: paged_attn_kernel
 * ──────────────────────────────────────────────────────────────────────────
 *   Grid:  (T_q, N_HEADS)    — one CUDA block per (query_token, head)
 *   Block: (HEAD_DIM,)       — one thread per output dimension
 *
 *   Each block:
 *     1. Loads this head's query vector Q[q_tok, head, :] into registers
 *     2. Iterates over all T_total KV positions
 *        a. Loads K[t, kv_head, :] from paged pool via phys_blocks[t], slots[t]
 *        b. Computes dot(Q, K) / sqrt(d) via parallel tree reduction
 *        c. Applies causal mask (positions t > start_pos + q_tok are masked)
 *        d. Updates online softmax running stats (max, sum) — safe numerics
 *        e. Loads V[t, kv_head, :] from pool and accumulates weighted output
 *     3. Normalises output by running sum and writes fp16 result
 *
 * Online softmax (flash-attention style numerics):
 *   Maintains running_max and running_sum as we scan KV tokens.
 *   When a new max is found, previously accumulated output is rescaled.
 *   Final: out = acc / running_sum  (no separate normalisation pass)
 *
 * GQA (Grouped Query Attention):
 *   kv_head = q_head / (n_heads / n_kv_heads)
 *   Multiple Q heads share one KV head — no K/V duplication in pool.
 *
 * Causal masking:
 *   Token at KV position t is masked if t > q_global (= start_pos + q_tok).
 *   Masked scores are set to -1e20 before softmax → ~0 attention weight.
 *   Masking is applied as a branch-free float factor so all threads
 *   participate in every __syncthreads() — no deadlock risk.
 *
 * Shared memory layout (per block):
 *   smem[0 .. head_dim-1]             — s_dot: scratch for dot-product reduction
 *   smem[head_dim .. 2*head_dim-1]    — s_out: running weighted V accumulator
 *   Total: 2 * head_dim * sizeof(float) bytes
 *
 * Pool memory layout (identical to kv_cache_kernels.cu):
 *   pool[phys_id, slot, head, dim]
 *   flat index = phys_id * BLOCK_SIZE * N_KV_HEADS * HEAD_DIM
 *              + slot    * N_KV_HEADS * HEAD_DIM
 *              + head    * HEAD_DIM
 *              + dim
 *
 * All pool tensors are fp16.  Internal accumulation is fp32 for precision.
 *
 * Compiled and loaded at runtime by kernels/paged_attention_kernels.py.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#define NEG_INF (-1e20f)


/* ─────────────────────────────────────────────────────────────────────────────
 * paged_attn_kernel
 *
 * Args (all device pointers unless noted):
 *   q            — (T_q, N_HEADS, HEAD_DIM) fp16 — query tokens
 *   pool_k       — (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16
 *   pool_v       — same layout as pool_k
 *   phys_blocks  — (T_total,) int32 — physical block id for each KV position
 *   slots        — (T_total,) int32 — slot offset within block for each KV pos
 *   out          — (T_q, N_HEADS, HEAD_DIM) fp16 — output
 *   T_q          — number of query tokens (1 during decode, >1 during prefill)
 *   T_total      — total number of KV tokens (past + current)
 *   start_pos    — absolute sequence position of q[0]
 *   n_heads      — number of query attention heads
 *   n_kv_heads   — number of KV heads (<= n_heads for GQA)
 *   head_dim     — dimension per head
 *   block_size   — tokens per physical block
 *   scale        — pre-computed 1.0 / sqrt(head_dim)
 * ───────────────────────────────────────────────────────────────────────────*/
__global__ void paged_attn_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v,
    const int*    __restrict__ phys_blocks,
    const int*    __restrict__ slots,
    __half*       __restrict__ out,
    int T_q,
    int T_total,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int block_size,
    float scale
) {
    /* ── Block assignment ─────────────────────────────────────────────────── */
    int q_tok  = blockIdx.x;   /* query token index (0 .. T_q-1)      */
    int q_head = blockIdx.y;   /* attention head   (0 .. N_HEADS-1)   */
    int d      = threadIdx.x;  /* head dimension   (0 .. HEAD_DIM-1)  */

    if (d >= head_dim || q_tok >= T_q || q_head >= n_heads) return;

    /* GQA: map query head → KV head */
    int n_rep   = n_heads / n_kv_heads;
    int kv_head = q_head / n_rep;

    /* Absolute sequence position of this query token */
    int q_global = start_pos + q_tok;

    /* ── Shared memory ──────────────────────────────────────────────────────
     *   s_dot[head_dim]  — scratch for parallel dot-product reduction
     *   s_out[head_dim]  — running weighted sum of V
     * ────────────────────────────────────────────────────────────────────── */
    extern __shared__ float smem[];
    float* s_dot = smem;              /* [0 .. head_dim-1]              */
    float* s_out = smem + head_dim;   /* [head_dim .. 2*head_dim-1]     */

    /* ── Load query value (scaled) into register ───────────────────────────*/
    int64_t q_idx = (int64_t)q_tok  * n_heads  * head_dim
                  + (int64_t)q_head * head_dim
                  + d;
    float q_val = __half2float(q[q_idx]) * scale;

    /* ── Initialise accumulator and online softmax state ───────────────────*/
    s_out[d] = 0.0f;
    float running_max = NEG_INF;
    float running_sum = 0.0f;
    __syncthreads();

    /* ── Main loop over KV tokens ──────────────────────────────────────────*/
    for (int t = 0; t < T_total; t++) {

        int phys_id = phys_blocks[t];
        int slot    = slots[t];

        /* Base offset for this (phys_id, slot, kv_head) in the pool */
        int64_t pool_base = (int64_t)phys_id * block_size * n_kv_heads * head_dim
                          + (int64_t)slot    * n_kv_heads * head_dim
                          + (int64_t)kv_head * head_dim;

        /* ── Load K and compute partial dot product ─────────────────────── */
        float k_val = __half2float(pool_k[pool_base + d]);
        s_dot[d] = q_val * k_val;   /* partial: q[d]*k[d], scaled by q_val */
        __syncthreads();

        /* ── Parallel tree reduction → s_dot[0] = dot(q,k)*scale ──────────
         *   All threads participate at every step — no conditional sync.    */
        for (int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) s_dot[d] += s_dot[d + stride];
            __syncthreads();
        }
        /* s_dot[0] now holds the full scaled dot product for this KV token */

        /* ── Causal masking (branch-free) ──────────────────────────────────
         *   All threads read s_dot[0]; masked positions get score = NEG_INF
         *   so their exp() ≈ 0 and they contribute nothing to output/sum.   */
        float score = s_dot[0];
        if (t > q_global) score = NEG_INF;   /* future token → mask out */

        /* ── Online softmax update ──────────────────────────────────────────
         *   All threads compute the same new_max / correction / exp_score,
         *   so running_max and running_sum are identical across threads.    */
        float new_max   = fmaxf(running_max, score);
        float corr      = expf(running_max - new_max); /* rescale old accum */
        float exp_score = expf(score - new_max);

        /* ── Rescale previous output accumulator ────────────────────────── */
        s_out[d] = s_out[d] * corr;
        /* No __syncthreads() needed: each thread d only writes s_out[d]    */

        /* ── Load V and accumulate weighted contribution ─────────────────── */
        float v_val = __half2float(pool_v[pool_base + d]);
        s_out[d] += exp_score * v_val;

        /* Update running stats (identical value in all threads) */
        running_max = new_max;
        running_sum = running_sum * corr + exp_score;

        __syncthreads();   /* sync before next iteration overwrites s_dot    */
    }

    /* ── Normalise and write fp16 output ───────────────────────────────────*/
    float result = (running_sum > 0.0f) ? (s_out[d] / running_sum) : 0.0f;

    int64_t out_idx = (int64_t)q_tok  * n_heads  * head_dim
                    + (int64_t)q_head * head_dim
                    + d;
    out[out_idx] = __float2half(result);
}


/* ─────────────────────────────────────────────────────────────────────────────
 * extern "C" launcher
 *
 * launch_paged_attn
 *   grid  = (T_q, n_heads)
 *   block = (head_dim,)
 *   smem  = 2 * head_dim * sizeof(float)
 * ───────────────────────────────────────────────────────────────────────────*/
extern "C" {

void launch_paged_attn(
    const void* q,
    const void* pool_k,
    const void* pool_v,
    const int*  phys_blocks,
    const int*  slots,
    void*       out,
    int T_q,
    int T_total,
    int start_pos,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int block_size,
    float scale
) {
    dim3 grid(T_q, n_heads);
    dim3 block(head_dim);
    size_t smem_bytes = 2 * head_dim * sizeof(float);

    paged_attn_kernel<<<grid, block, smem_bytes>>>(
        (const __half*)q,
        (const __half*)pool_k,
        (const __half*)pool_v,
        phys_blocks,
        slots,
        (__half*)out,
        T_q, T_total, start_pos,
        n_heads, n_kv_heads, head_dim, block_size,
        scale
    );
}

} /* extern "C" */

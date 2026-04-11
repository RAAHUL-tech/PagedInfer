/*
 *
 * Handles BOTH prefill (T_q > 1 tokens) and decode (T_q = 1 token)
 * sequences in a SINGLE kernel launch — the key operation of continuous
 * batching engines.
 *
 * Kernel: paged_attn_unified
 * ──────────────────────────────────────────────────────────────────────────
 *   Grid:  (TOTAL_Q_TOKENS, N_HEADS)
 *   Block: (HEAD_DIM,)
 *   Smem:  2 * HEAD_DIM * sizeof(float)
 *
 *   Each CUDA block handles one (query_token, head) pair across ALL sequences.
 *
 * Inputs
 * ──────
 *   q            (TOTAL_Q, N_HEADS, HEAD_DIM)  fp16
 *                — all query tokens concatenated: prefill seqs first, then decode seqs
 *
 *   pool_k/v     (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM)  fp16
 *                — paged KV pool (same layout as kv_cache_kernels.cu)
 *
 *   block_table  (N_SEQS, MAX_BLOCKS_PER_SEQ)  int32
 *                — THIS layer's physical block ids per sequence
 *                  block_table[seq_idx][logical_block] = physical_block_id
 *
 *   context_lens (N_SEQS,)  int32
 *                — total KV tokens per sequence AFTER writing current step's K/V
 *                  (includes both past context and the new tokens written this step)
 *
 *   q_lens       (N_SEQS,)  int32
 *                — number of query tokens from each sequence this step
 *                  > 1 during prefill (or chunked prefill)
 *                  = 1 during decode
 *
 *   q_offsets    (N_SEQS,)  int32
 *                — start index of each sequence's queries in the q tensor
 *                  q_offsets[s] = sum(q_lens[0..s-1])
 *
 *   out          (TOTAL_Q, N_HEADS, HEAD_DIM)  fp16 — output
 *
 * Causal masking
 * ──────────────
 *   For query token at local position q_pos within its sequence:
 *     global KV position of this query = (context_len - q_len) + q_pos
 *     attends to KV positions: 0 .. (context_len - q_len + q_pos)  [inclusive]
 *   Decode (q_len=1, q_pos=0):  attends to all context_len KV positions. ✓
 *   Prefill (q_len>1):          attends causally to previous positions.  ✓
 *
 * GQA
 * ───
 *   kv_head = q_head / (n_heads / n_kv_heads)
 *
 * Online softmax
 * ──────────────
 *   Flash-attention style: running max + sum, rescale accumulator on new max.
 *   All threads share the same score (after dot-product reduction) so
 *   running_max and running_sum are identical across threads — no extra sync.
 *
 * Pool layout (identical to kv_cache_kernels.cu)
 * ──────────────────────────────────────────────
 *   pool[phys_id, slot, head, dim]
 *   flat = phys_id * BS * NKV * HD + slot * NKV * HD + head * HD + dim
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#define NEG_INF_F (-1e20f)


/* ─────────────────────────────────────────────────────────────────────────────
 * paged_attn_unified
 * ───────────────────────────────────────────────────────────────────────────*/
__global__ void paged_attn_unified(
    const __half* __restrict__ q,
    const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v,
    const int*    __restrict__ block_table,   /* (N_SEQS, MAX_BLOCKS) */
    const int*    __restrict__ context_lens,  /* (N_SEQS,) */
    const int*    __restrict__ q_lens,        /* (N_SEQS,) */
    const int*    __restrict__ q_offsets,     /* (N_SEQS,) */
    __half*       __restrict__ attn_out,
    int n_seqs,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    /* ── Block assignment ─────────────────────────────────────────────────── */
    int q_tok_global = blockIdx.x;   /* global query token index   */
    int q_head       = blockIdx.y;   /* attention head             */
    int d            = threadIdx.x;  /* head dimension thread      */

    if (d >= head_dim) return;

    /* ── Find which sequence this query token belongs to ─────────────────────
     * Linear scan over q_offsets — valid since N_SEQS is small (≤ 16).    */
    int seq_idx = n_seqs - 1;   /* default: last seq (handles edge case) */
    for (int s = 0; s < n_seqs; s++) {
        if (q_tok_global < q_offsets[s] + q_lens[s]) {
            seq_idx = s;
            break;
        }
    }

    int q_pos_in_seq = q_tok_global - q_offsets[seq_idx];  /* local query pos */
    int ctx_len      = context_lens[seq_idx];  /* total KV tokens this seq    */
    int q_len        = q_lens[seq_idx];        /* query tokens this step      */

    /* KV positions to attend to: 0 .. kv_end-1 (causal)
     * kv_end = (ctx_len - q_len) + q_pos_in_seq + 1
     * Decode (q_len=1, q_pos=0): kv_end = ctx_len (full history)
     * Prefill (q_len>1):         kv_end = past_ctx + q_pos + 1 (causal)    */
    int kv_end = (ctx_len - q_len) + q_pos_in_seq + 1;

    /* GQA: map query head → KV head */
    int n_rep   = n_heads / n_kv_heads;
    int kv_head = q_head / n_rep;

    /* ── Shared memory ──────────────────────────────────────────────────────
     *   s_dot[head_dim]  — scratch for parallel dot-product reduction
     *   s_out[head_dim]  — running weighted V accumulator
     * ────────────────────────────────────────────────────────────────────── */
    extern __shared__ float smem[];
    float* s_dot = smem;
    float* s_out = smem + head_dim;

    /* Load this thread's query value (scaled) */
    int64_t q_idx = (int64_t)q_tok_global * n_heads * head_dim
                  + (int64_t)q_head       * head_dim + d;
    float q_val = __half2float(q[q_idx]);
    float scale = 1.0f / sqrtf((float)head_dim);
    q_val *= scale;

    /* Initialise accumulator and online softmax state */
    s_out[d] = 0.0f;
    float running_max = NEG_INF_F;
    float running_sum = 0.0f;
    __syncthreads();

    /* ── Main loop: iterate over KV positions 0 .. kv_end-1 ─────────────────
     * Physical address computed from block_table:
     *   logical_block = pos / block_size
     *   slot          = pos % block_size
     *   phys_id       = block_table[seq_idx * max_blocks + logical_block]
     * ────────────────────────────────────────────────────────────────────── */
    for (int pos = 0; pos < kv_end; pos++) {
        int logical_block = pos / block_size;
        int slot          = pos % block_size;
        int phys_id       = block_table[seq_idx * max_blocks_per_seq + logical_block];

        int64_t pool_base = (int64_t)phys_id * block_size * n_kv_heads * head_dim
                          + (int64_t)slot    * n_kv_heads * head_dim
                          + (int64_t)kv_head * head_dim;

        /* ── Dot product: each thread contributes q[d]*k[d] then reduce ───── */
        float k_val = __half2float(pool_k[pool_base + d]);
        s_dot[d] = q_val * k_val;
        __syncthreads();

        /* Parallel tree reduction → s_dot[0] = dot(q,k)*scale */
        for (int stride = head_dim >> 1; stride > 0; stride >>= 1) {
            if (d < stride) s_dot[d] += s_dot[d + stride];
            __syncthreads();
        }
        float score = s_dot[0];   /* all threads read the same value */

        /* ── Online softmax update ──────────────────────────────────────────
         * All threads compute the same new_max, corr, exp_score.           */
        float new_max   = fmaxf(running_max, score);
        float corr      = expf(running_max - new_max);
        float exp_score = expf(score - new_max);

        /* Rescale previous accumulator, add new weighted V */
        float v_val = __half2float(pool_v[pool_base + d]);
        s_out[d] = s_out[d] * corr + exp_score * v_val;

        running_max = new_max;
        running_sum = running_sum * corr + exp_score;

        __syncthreads();   /* safe before next s_dot write */
    }

    /* ── Normalise and write fp16 output ─────────────────────────────────── */
    float result = (running_sum > 0.0f) ? (s_out[d] / running_sum) : 0.0f;

    int64_t out_idx = (int64_t)q_tok_global * n_heads * head_dim
                    + (int64_t)q_head       * head_dim + d;
    attn_out[out_idx] = __float2half(result);
}


/* ─────────────────────────────────────────────────────────────────────────────
 * extern "C" launcher
 *
 * launch_paged_attn_unified
 *   grid  = (total_q_tokens, n_heads)
 *   block = (head_dim,)
 *   smem  = 2 * head_dim * sizeof(float)
 * ───────────────────────────────────────────────────────────────────────────*/
extern "C" {

void launch_paged_attn_unified(
    const void* q,
    const void* pool_k,
    const void* pool_v,
    const int*  block_table,
    const int*  context_lens,
    const int*  q_lens,
    const int*  q_offsets,
    void*       attn_out,
    int total_q_tokens,
    int n_seqs,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int block_size,
    int max_blocks_per_seq
) {
    dim3 grid(total_q_tokens, n_heads);
    dim3 block(head_dim);
    size_t smem_bytes = 2 * head_dim * sizeof(float);

    paged_attn_unified<<<grid, block, smem_bytes>>>(
        (const __half*)q,
        (const __half*)pool_k,
        (const __half*)pool_v,
        block_table,
        context_lens,
        q_lens,
        q_offsets,
        (__half*)attn_out,
        n_seqs,
        n_heads, n_kv_heads, head_dim,
        block_size, max_blocks_per_seq
    );
}

} /* extern "C" */

/*
 * kv_cache_kernels.cu — CUDA kernels for paged KV cache I/O
 *
 * Three kernels:
 *
 *   kv_init_blocks
 *     Zero-fill newly allocated physical blocks.
 *     Prevents stale data from previous sequences leaking into new ones.
 *     grid = (n_new_blocks, BLOCK_SIZE, N_KV_HEADS)
 *     block = (HEAD_DIM,)
 *
 *   kv_write
 *     Scatter K and V tokens from a contiguous (T, N_KV_HEADS, HEAD_DIM)
 *     input tensor into their physical (block_id, slot) addresses in the pool.
 *     grid = (T, N_KV_HEADS)
 *     block = (HEAD_DIM,)
 *
 *   kv_read
 *     Gather K and V from scattered physical (block_id, slot) addresses
 *     into a contiguous (T, N_KV_HEADS, HEAD_DIM) output tensor.
 *     grid = (T, N_KV_HEADS)
 *     block = (HEAD_DIM,)
 *
 * Pool memory layout (row-major, all three kernels use the same formula):
 *   pool[phys_id, slot, head, dim]
 *   flat index = phys_id * BLOCK_SIZE * N_KV_HEADS * HEAD_DIM
 *              + slot    * N_KV_HEADS  * HEAD_DIM
 *              + head    * HEAD_DIM
 *              + dim
 *
 * All tensors are fp16 (__half). int64_t indices avoid overflow for large pools.
 *
 * Compiled and loaded at runtime by kernels/kv_cache_kernels.py via nvcc + ctypes.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/* ─────────────────────────────────────────────────────────────────────────────
 * kv_init_blocks
 *
 * Zero-fill `n_new` physical blocks in pool_k and pool_v.
 *
 * Args (device pointers):
 *   pool_k, pool_v  — (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16
 *   block_ids       — (n_new,) int32  physical block ids to initialise
 *   block_size      — tokens per block
 *   n_kv_heads      — number of KV heads
 *   head_dim        — dimension per head
 * ───────────────────────────────────────────────────────────────────────────*/
__global__ void kv_init_blocks(
    __half* __restrict__ pool_k,
    __half* __restrict__ pool_v,
    const int* __restrict__ block_ids,
    int block_size,
    int n_kv_heads,
    int head_dim
) {
    int blk_idx = blockIdx.x;   /* index into block_ids array */
    int slot    = blockIdx.y;   /* token slot within block     */
    int head    = blockIdx.z;   /* KV head index               */
    int dim     = threadIdx.x;  /* head dimension              */

    if (dim >= head_dim) return;

    int phys_id = block_ids[blk_idx];

    int64_t base = (int64_t)phys_id * block_size * n_kv_heads * head_dim
                 + (int64_t)slot    * n_kv_heads  * head_dim
                 + (int64_t)head    * head_dim
                 + dim;

    pool_k[base] = __float2half(0.0f);
    pool_v[base] = __float2half(0.0f);
}


/* ─────────────────────────────────────────────────────────────────────────────
 * kv_write
 *
 * Scatter T tokens' K and V into the paged pool.
 * Each token maps to (phys_blocks[tok], slots[tok]) in the pool.
 *
 * Args:
 *   pool_k, pool_v  — (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16  [write]
 *   k_in,   v_in    — (T, N_KV_HEADS, HEAD_DIM) fp16                   [read]
 *   phys_blocks     — (T,) int32  physical block id for each token
 *   slots           — (T,) int32  slot offset within block for each token
 *   n_kv_heads, head_dim, block_size — pool geometry
 * ───────────────────────────────────────────────────────────────────────────*/
__global__ void kv_write(
    __half* __restrict__       pool_k,
    __half* __restrict__       pool_v,
    const __half* __restrict__ k_in,
    const __half* __restrict__ v_in,
    const int* __restrict__    phys_blocks,
    const int* __restrict__    slots,
    int n_kv_heads,
    int head_dim,
    int block_size
) {
    int tok  = blockIdx.x;   /* token index in the current write batch */
    int head = blockIdx.y;   /* KV head index                          */
    int dim  = threadIdx.x;  /* head dimension                         */

    if (dim >= head_dim) return;

    int phys_id = phys_blocks[tok];
    int slot    = slots[tok];

    /* Source: contiguous (T, N_KV_HEADS, HEAD_DIM) */
    int64_t src = (int64_t)tok  * n_kv_heads * head_dim
                + (int64_t)head * head_dim
                + dim;

    /* Destination: scattered pool layout */
    int64_t dst = (int64_t)phys_id * block_size * n_kv_heads * head_dim
                + (int64_t)slot    * n_kv_heads  * head_dim
                + (int64_t)head    * head_dim
                + dim;

    pool_k[dst] = k_in[src];
    pool_v[dst] = v_in[src];
}


/* ─────────────────────────────────────────────────────────────────────────────
 * kv_read
 *
 * Gather T tokens' K and V from the paged pool into contiguous output buffers.
 * Inverse of kv_write.
 *
 * Args:
 *   pool_k, pool_v  — (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16  [read]
 *   k_out,  v_out   — (T, N_KV_HEADS, HEAD_DIM) fp16                   [write]
 *   phys_blocks     — (T,) int32
 *   slots           — (T,) int32
 *   n_kv_heads, head_dim, block_size — pool geometry
 * ───────────────────────────────────────────────────────────────────────────*/
__global__ void kv_read(
    const __half* __restrict__ pool_k,
    const __half* __restrict__ pool_v,
    __half* __restrict__       k_out,
    __half* __restrict__       v_out,
    const int* __restrict__    phys_blocks,
    const int* __restrict__    slots,
    int n_kv_heads,
    int head_dim,
    int block_size
) {
    int tok  = blockIdx.x;
    int head = blockIdx.y;
    int dim  = threadIdx.x;

    if (dim >= head_dim) return;

    int phys_id = phys_blocks[tok];
    int slot    = slots[tok];

    /* Source: scattered pool layout */
    int64_t src = (int64_t)phys_id * block_size * n_kv_heads * head_dim
                + (int64_t)slot    * n_kv_heads  * head_dim
                + (int64_t)head    * head_dim
                + dim;

    /* Destination: contiguous (T, N_KV_HEADS, HEAD_DIM) */
    int64_t dst = (int64_t)tok  * n_kv_heads * head_dim
                + (int64_t)head * head_dim
                + dim;

    k_out[dst] = pool_k[src];
    v_out[dst] = pool_v[src];
}


/* ─────────────────────────────────────────────────────────────────────────────
 * extern "C" launchers
 *
 * Disables C++ name mangling so ctypes can find symbols by plain name.
 * All tensor arguments arrive as void* (raw GPU device pointers from
 * tensor.data_ptr() in Python).
 * ───────────────────────────────────────────────────────────────────────────*/
extern "C" {

/*
 * launch_init_blocks
 *   grid  = (n_new, block_size, n_kv_heads)
 *   block = (head_dim,)
 */
void launch_init_blocks(
    void*      pool_k,
    void*      pool_v,
    const int* block_ids,
    int n_new,
    int block_size,
    int n_kv_heads,
    int head_dim
) {
    dim3 grid(n_new, block_size, n_kv_heads);
    dim3 block(head_dim);
    kv_init_blocks<<<grid, block>>>(
        (__half*)pool_k,
        (__half*)pool_v,
        block_ids,
        block_size, n_kv_heads, head_dim
    );
}

/*
 * launch_write
 *   grid  = (T, n_kv_heads)
 *   block = (head_dim,)
 */
void launch_write(
    void*       pool_k,
    void*       pool_v,
    const void* k_in,
    const void* v_in,
    const int*  phys_blocks,
    const int*  slots,
    int T,
    int n_kv_heads,
    int head_dim,
    int block_size
) {
    dim3 grid(T, n_kv_heads);
    dim3 block(head_dim);
    kv_write<<<grid, block>>>(
        (__half*)pool_k,       (__half*)pool_v,
        (const __half*)k_in,   (const __half*)v_in,
        phys_blocks, slots,
        n_kv_heads, head_dim, block_size
    );
}

/*
 * launch_read
 *   grid  = (T, n_kv_heads)
 *   block = (head_dim,)
 */
void launch_read(
    const void* pool_k,
    const void* pool_v,
    void*       k_out,
    void*       v_out,
    const int*  phys_blocks,
    const int*  slots,
    int T,
    int n_kv_heads,
    int head_dim,
    int block_size
) {
    dim3 grid(T, n_kv_heads);
    dim3 block(head_dim);
    kv_read<<<grid, block>>>(
        (const __half*)pool_k, (const __half*)pool_v,
        (__half*)k_out,        (__half*)v_out,
        phys_blocks, slots,
        n_kv_heads, head_dim, block_size
    );
}

} /* extern "C" */

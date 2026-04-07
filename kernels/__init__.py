"""
kernels/ — CUDA kernels for PagedInfer.

Kernels
───────
kv_cache_kernels.cu / .py
    Three kernels for paged KV pool I/O:
      kv_init_blocks  — zero-fill newly allocated physical blocks
      kv_write        — scatter K/V tokens into pool (T, NKV, HD) → paged
      kv_read         — gather K/V tokens from pool paged → (T, NKV, HD)

    Python API:
        from kernels.kv_cache_kernels import load_kv_kernels
        ops = load_kv_kernels()   # compiles on first call, cached thereafter
        ops.write_kv_cuda(pool_k, pool_v, k, v, phys_t, slot_t)
        ops.read_kv_cuda(pool_k, pool_v, phys_t, slot_t) -> (k_out, v_out)
        ops.init_blocks_cuda(pool_k, pool_v, block_ids)

paged_attention.cu / paged_attention_kernels.py
    One fused kernel for paged attention computation:
      paged_attn_kernel — computes softmax(Q @ K^T / sqrt(d)) @ V where
                          K/V are stored non-contiguously in the paged pool.
                          Online softmax (flash-attention style numerics).
                          GQA and causal masking handled inside the kernel.
                          No separate gather step needed.

    Grid:  (T_q, N_HEADS)   — one CUDA block per (query_token, head)
    Block: (HEAD_DIM,)      — one thread per output dimension
    Smem:  2 × HEAD_DIM × 4 bytes

    Python API:
        from kernels.paged_attention_kernels import load_paged_attn_kernels
        ops = load_paged_attn_kernels()
        out = ops.paged_attention_cuda(q, pool_k, pool_v, phys_t, slot_t, ...)

Compilation
───────────
Both kernels are compiled at runtime via nvcc the first time they are
needed.  Compiled .so files are cached in /tmp/:
    /tmp/pagedinfer_kv_ops.so
    /tmp/pagedinfer_paged_attn.so

The SM architecture is detected automatically from torch.cuda.get_device_capability().
Re-compilation is triggered when the source .cu file is newer than the .so.
"""

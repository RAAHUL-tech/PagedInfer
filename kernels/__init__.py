"""
kernels/ — CUDA kernels for PagedInfer.
────────────────────────────────────────
Three pairs of .cu (CUDA C) + _ops.py (Python wrapper) files.
Each .cu is compiled once at runtime via nvcc and cached as a .so in /tmp/.

kv_io.cu  +  kv_io_ops.py
    KV pool I/O — the building blocks for reading and writing K/V tensors.
    Three kernels:
      kv_init_blocks  zero-fill newly allocated physical blocks on allocation
      kv_write        scatter K/V from (T, NKV, HD) into paged pool addresses
      kv_read         gather K/V from paged pool into contiguous (T, NKV, HD)
    Used by: GPUPagedKVCache (kv_cache/gpu_paged_kv_cache.py)

    from kernels.kv_io_ops import load_kv_kernels
    ops = load_kv_kernels()
    ops.write_kv_cuda(pool_k, pool_v, k, v, phys_t, slot_t)
    ops.read_kv_cuda(pool_k, pool_v, phys_t, slot_t) -> (k_out, v_out)

attn_decode.cu  +  attn_decode_ops.py
    Single-sequence paged attention — used during single-request GPU generation.
    Reads K/V from the paged pool inside the kernel (no gather step before SDPA).
    Handles GQA, causal masking, and online softmax in one pass.
    Grid: (T_q, N_HEADS)  Block: (HEAD_DIM,)
    Used by: GPUPagedKVCache.compute_paged_attn() (kv_cache/gpu_paged_kv_cache.py)

    from kernels.attn_decode_ops import load_paged_attn_kernels
    ops = load_paged_attn_kernels()
    out = ops.paged_attention_cuda(q, pool_k, pool_v, phys_t, slot_t, ...)

attn_continuous.cu  +  attn_continuous_ops.py
    Multi-sequence batched attention — the core kernel of the continuous batching engine.
    Handles ALL sequences (prefill T_q>1 AND decode T_q=1) in ONE kernel launch.
    Resolves physical KV addresses from block_table[n_seqs, max_blocks] inside the kernel.
    Grid: (TOTAL_Q_TOKENS, N_HEADS)  Block: (HEAD_DIM,)
    Used by: engine forward pass (engine/forward_pass.py)

    from kernels.attn_continuous_ops import load_unified_attn_kernels
    ops = load_unified_attn_kernels()
    out = ops.forward(q, pool_k, pool_v, block_table_t, ctx_t, qlens_t, qoff_t, ...)

Compilation
───────────
All kernels compile on first use via nvcc and are cached in /tmp/:
    /tmp/pagedinfer_kv_io.so
    /tmp/pagedinfer_attn_decode.so
    /tmp/pagedinfer_attn_continuous.so

SM arch is auto-detected from torch.cuda.get_device_capability().
Re-compilation is triggered when the .cu source is newer than the cached .so.
"""

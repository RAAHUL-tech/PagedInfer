"""
kv_cache/ — paged KV cache subsystem for PagedInfer.
──────────────────────────────────────────────────────
Implements the vLLM-style paged KV cache: instead of pre-allocating a fixed
(max_seq_len, n_kv_heads, head_dim) tensor per sequence, memory is divided
into fixed-size physical blocks that are allocated on demand and tracked via
a per-sequence logical→physical mapping.

Components:
    block_allocator.py   BlockAllocator — free-list of integer physical block IDs
    block_table.py       LayeredBlockTable — maps (layer, logical_pos) → (phys_block, slot)
                         Each logical block gets n_layers separate physical IDs
                         so every transformer layer has its own independent storage.
    paged_kv_cache.py    PagedKVCache — CPU-friendly pool (dict of tensors per layer).
                         Allocates tensors on demand; suitable for CPU or small GPU tests.
    gpu_paged_kv_cache.py  GPUPagedKVCache — GPU pool backed by one contiguous tensor
                           (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16.
                           All I/O goes through CUDA kernels (kernels/kv_io_ops.py).
    cpu_attn.py          paged_attention() — Python-level gather + SDPA fallback.
                         Used by PagedAttention when kv_cache is a PagedKVCache.

Backend dispatch (models/attention.py — PagedAttention._compute_attention):
    GPUPagedKVCache → CUDA kernel (attn_decode.cu)  — no explicit gather
    PagedKVCache    → cpu_attn.paged_attention()    — Python gather + SDPA
"""

from .block_allocator import BlockAllocator
from .block_table import LayeredBlockTable
from .cpu_attn import paged_attention
from .paged_kv_cache import PagedKVCache

__all__ = [
    "BlockAllocator",
    "LayeredBlockTable",
    "PagedKVCache",
    "GPUPagedKVCache",
    "paged_attention",
]

# GPUPagedKVCache depends on CUDA kernels — import lazily so CPU-only
# environments can still use the rest of the kv_cache package.
try:
    from .gpu_paged_kv_cache import GPUPagedKVCache
except Exception:
    pass  # CUDA not available or nvcc not found — GPUPagedKVCache unavailable

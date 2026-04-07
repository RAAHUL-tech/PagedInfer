"""
kv_cache/ — paged KV cache subsystem for PagedInfer.

Public API:
    BlockAllocator    — free-list manager for physical block ids
    LayeredBlockTable — per-sequence logical→physical block mapping
    PagedKVCache      — physical KV storage pool (dict of tensors, CPU-friendly)
    GPUPagedKVCache   — contiguous GPU pool backed by CUDA kernels
    paged_attention   — gather-based attention kernel (pure Python / torch)

Backend selection:
    CPU / no CUDA   →  PagedKVCache      (dict of tensors, on-demand allocation)
    CUDA available  →  GPUPagedKVCache   (single cudaMalloc, kernel I/O)

    PagedAttention.forward() in models/attention.py detects which type is
    passed as kv_cache and dispatches accordingly:
      - GPUPagedKVCache → paged_attn CUDA kernel (no separate gather step)
      - PagedKVCache    → Python-level paged_attention() gather + SDPA
"""

from .block_allocator import BlockAllocator
from .block_table import LayeredBlockTable
from .paged_attention import paged_attention
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

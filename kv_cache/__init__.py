"""
kv_cache/ — paged KV cache subsystem for PagedInfer.

Public API:
    BlockAllocator    — free-list manager for physical block ids
    LayeredBlockTable — per-sequence logical→physical block mapping
    PagedKVCache      — physical KV storage pool (dict of tensors per layer)
    paged_attention   — gather-based attention kernel (pure Python / torch)
"""

from .block_allocator import BlockAllocator
from .block_table import LayeredBlockTable
from .paged_attention import paged_attention
from .paged_kv_cache import PagedKVCache

__all__ = [
    "BlockAllocator",
    "LayeredBlockTable",
    "PagedKVCache",
    "paged_attention",
]

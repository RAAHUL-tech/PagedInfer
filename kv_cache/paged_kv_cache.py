"""
PagedKVCache — physical KV storage for the paged attention system.

Storage layout:
    pool_k[layer_idx][block_id] = tensor(block_size, n_kv_heads, head_dim)
    pool_v[layer_idx][block_id] = tensor(block_size, n_kv_heads, head_dim)

Key design choices:
  - Memory is allocated on demand, not at init.  Only tokens that are
    actually generated consume GPU memory.
  - Each (layer, block_id) pair is an independent tensor.  The BlockAllocator
    ensures no two sequences share a block_id within the same layer.
  - Reads scatter K/V from non-contiguous physical blocks into a contiguous
    tensor for the attention kernel.  This is the "gather" step of paged attention.
  - fp16 storage by default; configurable.

This mirrors the core of vLLM's KV cache, minus prefix-caching and
copy-on-write (those are future extensions).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import torch

if TYPE_CHECKING:
    from .block_table import LayeredBlockTable


class PagedKVCache:
    """
    Physical KV storage pool.

    Each block_id belongs to exactly one layer — guaranteed by LayeredBlockTable
    which allocates n_layers separate ids per logical block.  This makes the
    pool a pure flat map with no layer-sharing ambiguity.

    Args:
        n_layers       : Number of transformer layers.
        n_total_blocks : Total physical block slots managed by the allocator.
        block_size     : Tokens per block (must match BlockAllocator / BlockTable).
        n_kv_heads     : Number of KV heads (after GQA reduction).
        head_dim       : Dimension per head.
        dtype          : Storage dtype (float16 recommended for GPU).
        device         : "cuda" or "cpu".
    """

    def __init__(
        self,
        n_layers       : int,
        n_total_blocks : int,
        block_size     : int,
        n_kv_heads     : int,
        head_dim       : int,
        dtype          : torch.dtype = torch.float16,
        device         : str = "cpu",
    ) -> None:
        self.n_layers       = n_layers
        self.n_total_blocks = n_total_blocks
        self.block_size     = block_size
        self.n_kv_heads     = n_kv_heads
        self.head_dim       = head_dim
        self.dtype          = dtype
        self.device         = device

        # Dictionaries — memory only allocated when a block is first used
        self.pool_k: List[Dict[int, torch.Tensor]] = [{} for _ in range(n_layers)]
        self.pool_v: List[Dict[int, torch.Tensor]] = [{} for _ in range(n_layers)]

    # ── Allocation / deallocation ─────────────────────────────────────────────

    def allocate_block_for_layer(self, layer: int, block_id: int) -> None:
        """
        Provision the GPU tensor for one (layer, block_id) slot.
        Called by LayeredBlockTable.append_token() — once per layer per logical block.
        """
        if block_id not in self.pool_k[layer]:
            shape = (self.block_size, self.n_kv_heads, self.head_dim)
            self.pool_k[layer][block_id] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.pool_v[layer][block_id] = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def free_blocks(self, block_ids: List[int]) -> None:
        """
        Release GPU tensors for a list of block ids.
        Each id belongs to exactly one layer; we search all layers but only
        the owning layer will have the id in its dict.
        """
        for bid in block_ids:
            for layer in range(self.n_layers):
                self.pool_k[layer].pop(bid, None)
                self.pool_v[layer].pop(bid, None)

    # ── Write ─────────────────────────────────────────────────────────────────

    def write_slot(
        self,
        layer    : int,
        block_id : int,
        slot     : int,
        k        : torch.Tensor,   # (n_kv_heads, head_dim)
        v        : torch.Tensor,
    ) -> None:
        """Write one token's K and V into a specific slot."""
        self.pool_k[layer][block_id][slot] = k.to(self.dtype)
        self.pool_v[layer][block_id][slot] = v.to(self.dtype)

    def write_tokens(
        self,
        layer       : int,
        block_table : "LayeredBlockTable",
        k_seq       : torch.Tensor,   # (T, n_kv_heads, head_dim)
        v_seq       : torch.Tensor,   # (T, n_kv_heads, head_dim)
        start_pos   : int = 0,
    ) -> None:
        """
        Write a contiguous slice of T token keys and values into the paged pool.
        Each token is placed at its layer-specific (physical_block, slot) address.
        """
        T = k_seq.shape[0]
        for i in range(T):
            phys_block, slot = block_table.translate(layer, start_pos + i)
            self.write_slot(layer, phys_block, slot, k_seq[i], v_seq[i])

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_sequence(
        self,
        layer       : int,
        block_table : "LayeredBlockTable",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K and V for all tokens in the sequence from non-contiguous blocks.

        Returns:
            k_gathered : (T, n_kv_heads, head_dim)
            v_gathered : (T, n_kv_heads, head_dim)

        This is the "gather" step that makes paged attention work — it assembles
        a contiguous view from scattered physical storage for the attention kernel.
        """
        phys_blocks, slots = block_table.get_all_physical_slots(layer)
        k_list = [self.pool_k[layer][pb][s] for pb, s in zip(phys_blocks, slots)]
        v_list = [self.pool_v[layer][pb][s] for pb, s in zip(phys_blocks, slots)]
        return torch.stack(k_list), torch.stack(v_list)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def allocated_block_slots(self) -> int:
        """Total allocated (layer, block_id) tensor pairs."""
        return sum(len(d) for d in self.pool_k)

    def allocated_logical_blocks(self) -> int:
        """Logical blocks = physical slots / n_layers."""
        return self.allocated_block_slots() // max(self.n_layers, 1)

    def memory_used_mb(self) -> float:
        """
        Actual GPU memory consumed by all live tensors (K + V).
        formula: slots × block_size × n_kv_heads × head_dim × dtype_bytes × 2 (K and V)
        """
        slots = self.allocated_block_slots()
        dtype_bytes = torch.finfo(self.dtype).bits // 8
        return slots * self.block_size * self.n_kv_heads * self.head_dim * dtype_bytes * 2 / 1e6

    def __repr__(self) -> str:
        return (
            f"PagedKVCache("
            f"logical_blocks={self.allocated_logical_blocks()}, "
            f"block_slots={self.allocated_block_slots()}, "
            f"memory={self.memory_used_mb():.3f} MB)"
        )

    # ── Static memory calculators ─────────────────────────────────────────────

    @staticmethod
    def flat_cache_memory_mb(
        n_seqs      : int,
        max_seq_len : int,
        n_layers    : int,
        n_kv_heads  : int,
        head_dim    : int,
        dtype_bytes : int = 2,
    ) -> float:
        """
        Memory used by a flat (naive) KV cache that pre-allocates max_seq_len
        for every sequence, regardless of actual generation length.

        This is the baseline all paged systems improve on.
        """
        return (
            2 * n_layers * n_seqs * max_seq_len * n_kv_heads * head_dim * dtype_bytes / 1e6
        )

    @staticmethod
    def paged_cache_memory_mb(
        n_tokens    : int,
        block_size  : int,
        n_layers    : int,
        n_kv_heads  : int,
        head_dim    : int,
        dtype_bytes : int = 2,
    ) -> float:
        """
        Memory used by a paged KV cache for a sequence of n_tokens.
        Includes internal fragmentation from the last (partial) block.
        """
        import math
        n_blocks = math.ceil(n_tokens / block_size)
        return (
            2 * n_layers * n_blocks * block_size * n_kv_heads * head_dim * dtype_bytes / 1e6
        )

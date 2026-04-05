"""
LayeredBlockTable — per-sequence logical-to-physical block mapping.

Each sequence has exactly one BlockTable.  The table maps:

    (layer_idx, logical_token_position)  →  (physical_block_id, slot_offset)

Why layered?
    Each logical block requires n_layers separate physical block ids — one per
    transformer layer.  This means the BlockAllocator tracks truly independent
    1-to-1 units (one block_id == one GPU tensor in exactly one layer), which
    keeps the allocator simple and makes layer isolation exact.

Memory layout example (BLOCK_SIZE=16, n_layers=12, 35 tokens):
    Logical block 0 → allocate(12) ids → [0 .. 11]
        layer 0  → phys_block 0   pool_k[0][0]  = tensor(16, n_kv_heads, head_dim)
        layer 1  → phys_block 1   pool_k[1][1]  = tensor(16, n_kv_heads, head_dim)
        …
        layer 11 → phys_block 11
    Logical block 1 → allocate(12) ids → [12 .. 23]
        layer 0  → phys_block 12
        …
    Logical block 2 (partial, 3 tokens used) → allocate(12) ids → [24 .. 35]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .block_allocator import BlockAllocator
    from .paged_kv_cache import PagedKVCache


class LayeredBlockTable:
    """
    Logical-to-physical block mapping for a single sequence — layer-aware.

    block_ids[layer_idx][logical_block_idx] = physical_block_id

    All bookkeeping is done in CPU memory.  Only the actual K/V tensors
    live on GPU (inside PagedKVCache).
    """

    def __init__(
        self,
        seq_id     : int,
        n_layers   : int,
        block_size : int = 16,
    ) -> None:
        self.seq_id     = seq_id
        self.n_layers   = n_layers
        self.block_size = block_size
        self.num_tokens = 0
        # block_ids[layer_idx][logical_block_idx] = physical_block_id
        self.block_ids: List[List[int]] = [[] for _ in range(n_layers)]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        """Number of logical blocks (same across all layers)."""
        return len(self.block_ids[0])

    @property
    def last_block_is_full(self) -> bool:
        return self.num_tokens > 0 and (self.num_tokens % self.block_size) == 0

    @property
    def last_block_offset(self) -> int:
        """Slot index of the next token to write within the current block."""
        return self.num_tokens % self.block_size

    # ── Token append ─────────────────────────────────────────────────────────

    def append_token(
        self,
        allocator : "BlockAllocator",
        kv_cache  : Optional["PagedKVCache"] = None,
    ) -> None:
        """
        Advance the table by one token.

        When a new logical block is needed (first token, or every block_size
        tokens), allocate n_layers physical ids at once — one per layer —
        and optionally provision the GPU tensor for each via kv_cache.
        """
        if self.num_blocks == 0 or self.last_block_is_full:
            new_ids = allocator.allocate(self.n_layers)
            for layer_idx in range(self.n_layers):
                phys_id = new_ids[layer_idx]
                self.block_ids[layer_idx].append(phys_id)
                if kv_cache is not None:
                    kv_cache.allocate_block_for_layer(layer_idx, phys_id)
        self.num_tokens += 1

    def append_tokens(
        self,
        n         : int,
        allocator : "BlockAllocator",
        kv_cache  : Optional["PagedKVCache"] = None,
    ) -> None:
        for _ in range(n):
            self.append_token(allocator, kv_cache)

    # ── Address translation ───────────────────────────────────────────────────

    def translate(self, layer: int, logical_pos: int) -> Tuple[int, int]:
        """
        Convert a logical token position to (physical_block_id, slot_offset)
        for a specific layer.

        Each layer has its own physical block id for the same logical position,
        ensuring full layer isolation in the KV pool.

        Example (BLOCK_SIZE=16, layer=3, logical_pos=20):
            logical_block = 20 // 16 = 1
            slot_offset   = 20 %  16 = 4
            phys_block    = block_ids[3][1]
            → pool_k[3][phys_block][4]
        """
        logical_block  = logical_pos // self.block_size
        slot_offset    = logical_pos %  self.block_size
        physical_block = self.block_ids[layer][logical_block]
        return physical_block, slot_offset

    def get_all_physical_slots(self, layer: int) -> Tuple[List[int], List[int]]:
        """
        Return (physical_block_ids, slot_offsets) for every token in the
        sequence for the given layer.  Used by PagedKVCache.read_sequence().
        """
        phys_blocks: List[int] = []
        slots:       List[int] = []
        for pos in range(self.num_tokens):
            pb, so = self.translate(layer, pos)
            phys_blocks.append(pb)
            slots.append(so)
        return phys_blocks, slots

    # ── Free ─────────────────────────────────────────────────────────────────

    def free(
        self,
        allocator : "BlockAllocator",
        kv_cache  : Optional["PagedKVCache"] = None,
    ) -> None:
        """
        Return all physical block ids across all layers to the allocator
        and optionally release their GPU tensors from the KV pool.
        """
        all_ids = [bid for layer_ids in self.block_ids for bid in layer_ids]
        if kv_cache is not None:
            kv_cache.free_blocks(all_ids)
        allocator.free(all_ids)
        self.block_ids  = [[] for _ in range(self.n_layers)]
        self.num_tokens = 0

    # ── Memory stats ─────────────────────────────────────────────────────────

    def blocks_needed_for(self, n_tokens: int) -> int:
        """Logical blocks required to hold n_tokens."""
        return math.ceil(n_tokens / self.block_size)

    def __repr__(self) -> str:
        return (
            f"LayeredBlockTable(seq={self.seq_id}, tokens={self.num_tokens}, "
            f"logical_blocks={self.num_blocks}, "
            f"layer0_ids={self.block_ids[0] if self.block_ids else []})"
        )

"""
kv_cache/prefix_block_table.py — block table that can reuse cached prefix blocks.

Extends the LayeredBlockTable concept with one new method:

    attach_cached_prefix(cached_ids, n_tokens)

Instead of allocating fresh physical blocks for the shared prefix portion,
this method maps the cached physical block IDs directly into block_ids[layer].
The allocator ref counts were already bumped by PrefixCache.match(), so the
blocks are pinned — no allocate() call is needed here.

The engine advances seq.prefill_offset past the cached tokens, so no forward
pass is run for those positions.  The first forward pass starts at the suffix.

Lifecycle
─────────
    1. PrefixCache.match() → returns (n_matched, cached_ids_per_layer)
    2. attach_cached_prefix(cached_ids, n_matched * block_size)
       → num_tokens advances, n_cached_blocks recorded
    3. append_tokens(suffix_len, allocator, kv_cache)
       → fresh blocks allocated only for the uncached suffix
    4. On free(): all blocks (cached + new) are freed together via allocator.free()
       → ref counts decremented; cached blocks only return to free-list when
         the prefix cache also evicts them (ref drops from 2 → 1 → 0)

Comparison to LayeredBlockTable (kv_cache/block_table.py)
─────────────────────────────────────────────────────────
    LayeredBlockTable       PrefixAwareBlockTable
    ─────────────────       ──────────────────────
    no prefix reuse         attach_cached_prefix()
    blocks_needed_next()    blocks_needed_for(n) / blocks_needed_next()
    new_blocks_for_tokens() blocks_needed_for(n)  (same logic, different name)
    translate(layer, pos)   translate(layer, pos)  — identical
"""

from __future__ import annotations

import math
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from kv_cache.block_allocator import BlockAllocator
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache


class PrefixAwareBlockTable:
    """
    Per-sequence block table that supports prefix block reuse.

    block_ids[layer][logical_block] = physical_block_id

    For the cached prefix (logical blocks 0 .. n_cached_blocks-1):
        - Physical IDs come from PrefixCache — no allocator.allocate() call
        - Ref counts already bumped by PrefixCache.match()

    For the uncached suffix (logical blocks n_cached_blocks ..):
        - append_token() / append_tokens() work exactly like LayeredBlockTable

    Args:
        seq_id     : Sequence identifier (for debugging).
        n_layers   : Number of transformer layers.
        block_size : Tokens per physical block.
    """

    def __init__(self, seq_id: int, n_layers: int, block_size: int) -> None:
        self.seq_id          = seq_id
        self.n_layers        = n_layers
        self.block_size      = block_size
        self.num_tokens      = 0
        self.block_ids       : List[List[int]] = [[] for _ in range(n_layers)]
        self.n_cached_blocks : int = 0   # leading blocks supplied by prefix cache

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def num_blocks(self) -> int:
        return len(self.block_ids[0])

    @property
    def last_block_is_full(self) -> bool:
        return self.num_tokens > 0 and (self.num_tokens % self.block_size) == 0

    # ── Prefix attachment ─────────────────────────────────────────────────────

    def attach_cached_prefix(
        self,
        cached_ids : List[List[int]],   # [layer][logical_block] → physical_id
        n_tokens   : int,               # tokens covered (= n_cached_blocks * block_size)
    ) -> None:
        """
        Map cached physical blocks into this table without allocating.

        cached_ids[li][i] = physical block id for layer li, logical block i.
        Ref counts were already bumped by PrefixCache.match() — nothing more
        needed here.  num_tokens is advanced so append_token() knows where the
        next free slot starts.

        Args:
            cached_ids : Per-layer list of physical block IDs from the cache.
            n_tokens   : Token positions covered (= n_cached_blocks * block_size).
        """
        n_blocks = len(cached_ids[0]) if cached_ids and cached_ids[0] else 0
        for li in range(self.n_layers):
            self.block_ids[li] = list(cached_ids[li])
        self.num_tokens      = n_tokens
        self.n_cached_blocks = n_blocks

    # ── Block allocation ──────────────────────────────────────────────────────

    def append_token(
        self,
        allocator : "BlockAllocator",
        kv_cache  : "GPUPagedKVCache | None" = None,
    ) -> None:
        """
        Ensure there is a slot for one more token, allocating a new block if needed.

        If the last block is full (or there are no blocks yet), allocate n_layers
        new physical blocks — one per transformer layer — and append them.
        GPUPagedKVCache is a pre-allocated contiguous pool and needs no per-block
        callback — the kv_cache argument is accepted for API compatibility only.
        """
        if self.num_blocks == 0 or self.last_block_is_full:
            new_ids = allocator.allocate(self.n_layers)
            for li in range(self.n_layers):
                self.block_ids[li].append(new_ids[li])
        self.num_tokens += 1

    def append_tokens(
        self,
        n         : int,
        allocator : "BlockAllocator",
        kv_cache  : "GPUPagedKVCache | None" = None,
    ) -> None:
        """Allocate slots for `n` additional tokens."""
        for _ in range(n):
            self.append_token(allocator, kv_cache)

    # ── Address translation ───────────────────────────────────────────────────

    def translate(self, layer: int, pos: int) -> tuple:
        """
        Convert a (layer, absolute_token_position) to (physical_block_id, slot).

        Returns (physical_block_id, slot_within_block).
        """
        return self.block_ids[layer][pos // self.block_size], pos % self.block_size

    def get_all_physical_slots(self, layer: int):
        """
        Return parallel (phys_block_ids, slots) lists for all token positions.

        Used by GPUPagedKVCache.read_sequence() to gather K/V from the pool.
        """
        pbs, ss = [], []
        for pos in range(self.num_tokens):
            pb, s = self.translate(layer, pos)
            pbs.append(pb)
            ss.append(s)
        return pbs, ss

    # ── Block count helpers ───────────────────────────────────────────────────

    def blocks_needed_for(self, n_more_tokens: int) -> int:
        """
        Number of NEW logical blocks needed to store `n_more_tokens` additional tokens.

        0 if there is enough slack in the current last block.
        """
        current_capacity = self.num_blocks * self.block_size
        free_slots       = current_capacity - self.num_tokens
        need_tokens      = max(0, n_more_tokens - free_slots)
        return math.ceil(need_tokens / self.block_size)

    def blocks_needed_next(self) -> int:
        """New logical blocks needed for one more token (0 or 1)."""
        return self.blocks_needed_for(1)

    # ── Deallocation ──────────────────────────────────────────────────────────

    def free(
        self,
        allocator : "BlockAllocator",
        kv_cache  : "GPUPagedKVCache | None" = None,
    ) -> None:
        """
        Free all physical blocks held by this table.

        For cached prefix blocks (n_cached_blocks leading entries), this
        decrements their ref count from 2 → 1.  They stay in the GPU pool
        until the prefix cache also evicts them (ref drops to 0).

        For newly allocated suffix blocks, ref goes 1 → 0 and they return
        immediately to the allocator free-list.
        """
        all_ids = [b for layer_ids in self.block_ids for b in layer_ids]
        allocator.free(all_ids)
        self.block_ids  = [[] for _ in range(self.n_layers)]
        self.num_tokens = 0

    def __repr__(self) -> str:
        return (
            f"PrefixAwareBlockTable(seq={self.seq_id}, "
            f"tokens={self.num_tokens}, blocks={self.num_blocks}, "
            f"cached_blocks={self.n_cached_blocks})"
        )

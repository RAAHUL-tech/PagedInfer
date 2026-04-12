"""
kv_cache/prefix_cache.py — hash-based KV block cache for prefix reuse.

What is prefix caching?
────────────────────────
When multiple requests share a common prefix (e.g. the same system prompt,
few-shot examples, or document context), the transformer has to compute
identical K/V for those tokens every time — wasting GPU compute and memory.

PrefixCache stores the physical KV block IDs for completed logical blocks
after a sequence finishes prefill.  When a new sequence arrives, its token
hash chain is matched against the cache.  Matching blocks are reused directly
— their KV is already in the GPU pool — and the forward pass only runs on the
suffix that differs.

Hash chain
──────────
Each logical block gets a hash that encodes the full token context up to and
including that block:

    h[0] = hash((0,             tuple(tokens[0 : block_size])))
    h[1] = hash((h[0],          tuple(tokens[block_size : 2*block_size])))
    h[2] = hash((h[1],          tuple(tokens[2*block_size : 3*block_size])))
    ...

Two blocks are identical iff their entire preceding context is identical.
Only COMPLETE blocks (length == block_size) are hashed.

LRU eviction
────────────
The cache is bounded by max_blocks.  When it fills up the least-recently-used
block is evicted — but only if the allocator reference count is exactly 1
(the cache is the sole owner).  Blocks shared with a live sequence (ref=2)
are skipped during eviction.

Reference counting
──────────────────
BlockAllocator._ref_count[b] tracks owners:
    1 = owned by one sequence OR by the cache alone
    2 = shared between cache and a live sequence
    0 = free

PrefixCache.match() calls allocator.inc_ref() for every matched block,
pinning them so they cannot be evicted while the new sequence is alive.
When the sequence finishes, block_table.free() decrements the count back.

Integration
───────────
    Called from engine/prefix_engine.py:
        match()                  → at request admission
        insert_sequence_blocks() → immediately after prefill completes
        evict_to_free()          → when allocator is running low
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from kv_cache.block_allocator import BlockAllocator


class PrefixCache:
    """
    vLLM-style hash-based KV block cache with LRU eviction.

    Storage: OrderedDict mapping block_hash → List[int] of length n_layers.
             _cache[h][layer_idx] = physical block id for that layer.
             OrderedDict gives O(1) LRU tracking via move_to_end().

    Args:
        allocator  : Shared BlockAllocator — used for inc_ref and free.
        max_blocks : Maximum number of cached logical blocks.
                     0 means unlimited (evict only when explicitly asked).
    """

    def __init__(self, allocator: "BlockAllocator", max_blocks: int = 0) -> None:
        self.allocator  = allocator
        self.max_blocks = max_blocks
        self._cache: OrderedDict[int, List[int]] = OrderedDict()

        # Telemetry
        self.n_hits    : int = 0
        self.n_misses  : int = 0
        self.n_evicted : int = 0

    # ── Hashing ───────────────────────────────────────────────────────────────

    @staticmethod
    def block_hash(parent_hash: int, block_tokens: tuple) -> int:
        """
        Compute the hash for one logical block.

        parent_hash   — hash of the immediately preceding block (0 for block 0)
        block_tokens  — tuple of token IDs in this block

        Chaining parent_hash means two blocks are equal iff their entire
        preceding context is identical — not just the tokens in this block.
        """
        return hash((parent_hash, block_tokens))

    def compute_hashes(self, token_ids: List[int], block_size: int) -> List[int]:
        """
        Build the full hash chain for a token sequence.

        Only COMPLETE blocks are hashed — partial trailing blocks are excluded
        because their KV slots are not fully written yet.

        Example (block_size=16, 48 tokens → 3 complete blocks):
            h[0] = hash((0,    tuple(tokens[0:16])))
            h[1] = hash((h[0], tuple(tokens[16:32])))
            h[2] = hash((h[1], tuple(tokens[32:48])))
        """
        n_complete = len(token_ids) // block_size
        hashes, parent = [], 0
        for i in range(n_complete):
            toks = tuple(token_ids[i * block_size : (i + 1) * block_size])
            h    = self.block_hash(parent, toks)
            hashes.append(h)
            parent = h
        return hashes

    # ── Lookup ────────────────────────────────────────────────────────────────

    def match(
        self,
        token_ids  : List[int],
        block_size : int,
        n_layers   : int,
    ) -> Tuple[int, List[List[int]]]:
        """
        Find cached prefix blocks for a new request.

        Walks the hash chain and collects matching blocks until the first miss.
        allocator.inc_ref() is called for every matched block to pin them — they
        cannot be evicted while the new sequence holds a reference.

        Returns
        -------
        n_matched : int             — number of logical blocks reused from cache
        per_layer : List[List[int]] — physical block ids indexed [layer][block]
                    per_layer[li][i] = physical block id for layer li, logical block i
        """
        per_layer = [[] for _ in range(n_layers)]
        if not self._cache:
            return 0, per_layer

        hashes    = self.compute_hashes(token_ids, block_size)
        n_matched = 0

        for h in hashes:
            if h not in self._cache:
                self.n_misses += 1
                break
            ids = self._cache[h]   # List[int], length = n_layers
            for li, phys_id in enumerate(ids):
                per_layer[li].append(phys_id)
                self.allocator.inc_ref(phys_id)   # pin: cache + new seq both hold ref
            self._cache.move_to_end(h)            # mark as recently used
            n_matched += 1
            self.n_hits += 1

        return n_matched, per_layer

    # ── Insertion ─────────────────────────────────────────────────────────────

    def insert(self, block_hash: int, ids_per_layer: List[int]) -> None:
        """
        Cache one completed logical block.

        ids_per_layer : List[int] of length n_layers.
            ids_per_layer[li] = physical block id for layer li.

        If the hash is already cached, just bump it to MRU.
        Evicts LRU blocks if over the hard limit (max_blocks > 0).
        """
        if block_hash in self._cache:
            self._cache.move_to_end(block_hash)
            return
        self._cache[block_hash] = ids_per_layer
        if self.max_blocks > 0:
            while len(self._cache) > self.max_blocks:
                self._evict_lru()

    def insert_sequence_blocks(
        self,
        token_ids  : List[int],
        block_table,               # PrefixAwareBlockTable (or LayeredBlockTable)
        block_size : int,
    ) -> None:
        """
        Insert all complete prompt blocks of a finished prefill into the cache.

        Called by PrefixAwareEngine immediately after the last prefill chunk
        completes.  Only fully written blocks are cached — the partial trailing
        block (if any) is excluded.

        Args:
            token_ids   : Prompt token IDs (slice to prompt_len before calling).
            block_table : The sequence's block table after prefill.
            block_size  : Tokens per block.
        """
        hashes = self.compute_hashes(token_ids, block_size)
        for i, h in enumerate(hashes):
            if h not in self._cache:
                ids = [block_table.block_ids[li][i]
                       for li in range(block_table.n_layers)]
                self.insert(h, ids)

    # ── Eviction ──────────────────────────────────────────────────────────────

    def evict_to_free(self, n_logical_blocks: int) -> int:
        """
        Evict up to n_logical_blocks LRU entries to free allocator space.

        Only evicts blocks whose ref count is exactly 1 (cache is the sole owner).
        Blocks with ref > 1 are shared with a live sequence and cannot be evicted.

        Returns the number of logical blocks actually evicted.
        """
        evicted = 0
        while evicted < n_logical_blocks and self._cache:
            if self._evict_lru():
                evicted += 1
            else:
                break   # all remaining blocks have live refs — cannot evict more
        return evicted

    def _evict_lru(self) -> bool:
        """Evict the least-recently-used entry if it has no live references."""
        if not self._cache:
            return False
        lru_hash, ids = next(iter(self._cache.items()))
        # Only evict if the cache is the sole owner (ref_count == 1)
        if all(self.allocator._ref_count[bid] == 1 for bid in ids):
            del self._cache[lru_hash]
            self.allocator.free(ids)
            self.n_evicted += 1
            return True
        # Block shared with a live seq — skip to end, try the next LRU
        self._cache.move_to_end(lru_hash)
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_cached(self) -> int:
        """Number of logical blocks currently in the cache."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate since creation."""
        total = self.n_hits + self.n_misses
        return self.n_hits / total if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"PrefixCache(cached={self.n_cached}, "
            f"hits={self.n_hits}, misses={self.n_misses}, "
            f"hit_rate={self.hit_rate:.1%}, evicted={self.n_evicted})"
        )

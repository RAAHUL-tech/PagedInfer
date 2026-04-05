"""
BlockAllocator — free-list manager for physical KV blocks.

A "physical block" is a fixed-size slot in the KV pool that holds
BLOCK_SIZE tokens worth of keys and values for one layer of the model.

The allocator mirrors the core of vLLM's block manager:
  - All blocks start on a free deque
  - allocate(n) pops n blocks from the front
  - free(ids) appends them back to the rear
  - ref_count bookkeeping is in place for copy-on-write (prefix sharing)

Terminology:
  logical block  — abstract index in a sequence (0, 1, 2, …)
  physical block — integer id in [0, n_total_blocks) referencing a real GPU tensor
"""

from __future__ import annotations

from collections import deque
from typing import List


class BlockAllocator:
    """
    Manages a flat pool of physical KV block ids via a free-list.

    One physical block id maps to exactly ONE (layer, tensor) pair.
    The LayeredBlockTable allocates n_layers ids per logical block so
    every layer has its own isolated physical storage.

    Args:
        n_total_blocks : Total number of physical block slots in the pool.
                         Typically: ceil(MAX_SEQS * MAX_SEQ_LEN / BLOCK_SIZE) * n_layers
    """

    def __init__(self, n_total_blocks: int) -> None:
        self.n_total    = n_total_blocks
        self._free      = deque(range(n_total_blocks))   # all blocks start free
        self._ref_count = [0] * n_total_blocks           # for future copy-on-write

    # ── Allocation ────────────────────────────────────────────────────────────

    def allocate(self, n: int = 1) -> List[int]:
        """
        Pop `n` physical block ids from the free-list.

        Raises:
            RuntimeError: if fewer than `n` blocks are available.
        """
        if len(self._free) < n:
            raise RuntimeError(
                f"Out of KV blocks: requested {n}, "
                f"only {len(self._free)} free out of {self.n_total} total."
            )
        blocks = [self._free.popleft() for _ in range(n)]
        for b in blocks:
            self._ref_count[b] = 1
        return blocks

    def free(self, block_ids: List[int]) -> None:
        """Return a list of physical block ids to the free-list."""
        for b in block_ids:
            if self._ref_count[b] > 0:
                self._ref_count[b] -= 1
            if self._ref_count[b] == 0:
                self._free.append(b)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def n_free(self) -> int:
        return len(self._free)

    @property
    def n_used(self) -> int:
        return self.n_total - self.n_free

    def utilization(self) -> float:
        return self.n_used / self.n_total if self.n_total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(total={self.n_total}, "
            f"free={self.n_free}, used={self.n_used}, "
            f"util={self.utilization():.1%})"
        )

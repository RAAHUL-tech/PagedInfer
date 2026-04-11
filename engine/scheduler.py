"""
engine/scheduler.py — iteration-level scheduler for continuous batching.

Every engine step the Scheduler is called once to decide:
  1. Which DECODING sequences continue (highest priority — always run first)
  2. Which SWAPPED sequences can be restored (second priority)
  3. Which WAITING sequences can be admitted for prefill (lowest priority)
  4. Which sequence to preempt if the pool is exhausted

Scheduling policy
─────────────────
  Priority 1 — DECODING:
    All currently decoding seqs get a decode slot.  If one needs a new block
    and the pool is full, we preempt the decode seq with the fewest generated
    tokens (cheapest to re-serve) and try again.

  Priority 2 — SWAPPED (restore):
    Preempted seqs that have CPU KV cache saved get priority over new requests
    because users are already waiting and no recomputation is needed.

  Priority 3 — WAITING (new prefill):
    New requests are admitted up to max_prefill_seqs and max_batch_size.
    Chunked prefill: if a prompt exceeds chunk_size tokens, only chunk_size
    tokens are processed this step.  The seq stays in PREFILL status and
    resumes next iteration.  This prevents one long prompt from blocking
    all decoding sequences for many steps.

Preemption
──────────
  When memory is exhausted and a decode seq needs a new block:
    - Find the decode seq with the fewest generated tokens
    - Call seq.swap_out() — copies K/V to CPU, frees GPU blocks
    - The freed blocks allow the evicted seq's slot to be taken by another

  The evicted seq goes to the SWAPPED list and will be restored in a
  future iteration once memory is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from kv_cache.block_allocator import BlockAllocator
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
    from engine.sequence import SequenceGroup

# Default tokens per prefill chunk — keep decode latency low
DEFAULT_CHUNK_SIZE = 32


@dataclass
class SchedulerOutput:
    """Decision made by the Scheduler for one engine iteration."""
    prefill_seqs   : List["SequenceGroup"]   # admitted for prefill this step
    decode_seqs    : List["SequenceGroup"]   # continue decoding this step
    preempted_seqs : List["SequenceGroup"]   # evicted from GPU this step
    swapped_in     : List["SequenceGroup"]   # restored from CPU this step


class Scheduler:
    """
    Iteration-level scheduler.

    Args:
        allocator        : BlockAllocator tracking free physical block IDs.
        n_layers         : Transformer depth — each logical block needs
                           n_layers physical IDs.
        block_size       : Tokens per KV block.
        max_batch_size   : Maximum sequences active at once (prefill + decode).
        max_prefill_seqs : Max new sequences admitted per iteration.
        chunk_size       : Max prompt tokens processed per prefill step.
    """

    def __init__(
        self,
        allocator       : "BlockAllocator",
        n_layers        : int,
        block_size      : int,
        max_batch_size  : int = 8,
        max_prefill_seqs: int = 4,
        chunk_size      : int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.allocator        = allocator
        self.n_layers         = n_layers
        self.block_size       = block_size
        self.max_batch_size   = max_batch_size
        self.max_prefill_seqs = max_prefill_seqs
        self.chunk_size       = chunk_size

        # Aggregate stats
        self.n_preemptions : int = 0
        self.n_swaps_in    : int = 0
        self.n_chunked     : int = 0   # iterations where chunked prefill occurred

    def _phys_ids_for_n_logical(self, n_logical: int) -> int:
        """Physical block IDs needed for n_logical new logical blocks."""
        return n_logical * self.n_layers

    def schedule(
        self,
        waiting : List["SequenceGroup"],
        running : List["SequenceGroup"],
        swapped : List["SequenceGroup"],
        kv_cache: "GPUPagedKVCache",
    ) -> SchedulerOutput:
        """
        Produce a schedule for the current iteration.

        Args:
            waiting : Seqs in WAITING or PREFILL status (not yet on GPU).
            running : Seqs in DECODING status (currently on GPU).
            swapped : Seqs in SWAPPED status (KV saved to CPU).
            kv_cache: GPU KV pool (needed for swap_out).

        Returns:
            SchedulerOutput with prefill_seqs, decode_seqs, etc.
        """
        prefill_seqs    : List["SequenceGroup"] = []
        decode_seqs     : List["SequenceGroup"] = []
        preempted_seqs  : List["SequenceGroup"] = []
        swapped_in_seqs : List["SequenceGroup"] = []

        # ── Priority 1: Reserve decode slots for currently running seqs ───────
        # Decode seqs always get first access to the pool.  If one needs a
        # new block but the pool is full, preempt the cheapest running seq.
        confirmed = []
        for seq in running:
            new_logical    = seq.blocks_needed_next()
            phys_needed    = self._phys_ids_for_n_logical(new_logical)

            if phys_needed == 0 or self.allocator.can_allocate(phys_needed):
                confirmed.append(seq)
                decode_seqs.append(seq)
            else:
                # Pool is full — preempt the confirmed seq with fewest tokens
                if confirmed:
                    victim = min(confirmed, key=lambda s: s.n_generated)
                    confirmed.remove(victim)
                    decode_seqs.remove(victim)
                    victim.swap_out(kv_cache, self.allocator)
                    preempted_seqs.append(victim)
                    self.n_preemptions += 1

                # Retry with freed blocks
                if phys_needed == 0 or self.allocator.can_allocate(phys_needed):
                    confirmed.append(seq)
                    decode_seqs.append(seq)
                else:
                    # Still no room — preempt this seq too
                    seq.swap_out(kv_cache, self.allocator)
                    preempted_seqs.append(seq)
                    self.n_preemptions += 1

        # ── Priority 2: Restore swapped seqs (already computed partial KV) ───
        for seq in swapped:
            if seq.cpu_kv_cache is None:
                continue
            n_saved         = seq.cpu_kv_cache[0][0].shape[0]
            logical_needed  = math.ceil(n_saved / self.block_size)
            phys_needed     = self._phys_ids_for_n_logical(logical_needed)
            total_active    = len(decode_seqs) + len(prefill_seqs) + len(swapped_in_seqs)

            if (self.allocator.can_allocate(phys_needed)
                    and total_active < self.max_batch_size):
                seq.swap_in(kv_cache, self.allocator)
                swapped_in_seqs.append(seq)
                decode_seqs.append(seq)
                self.n_swaps_in += 1

        # ── Priority 3: Admit new waiting seqs for prefill ────────────────────
        n_admitted = 0
        for seq in waiting:
            if n_admitted >= self.max_prefill_seqs:
                break
            total_active = len(decode_seqs) + len(prefill_seqs)
            if total_active >= self.max_batch_size:
                break

            # How many tokens to process this chunk?
            chunk_end  = min(seq.prefill_offset + self.chunk_size, seq.prompt_len)
            chunk_len  = chunk_end - seq.prefill_offset
            phys_needed = self._phys_ids_for_n_logical(
                math.ceil(chunk_len / self.block_size)
            )

            if self.allocator.can_allocate(phys_needed):
                if chunk_len < seq.prompt_len:
                    self.n_chunked += 1
                from engine.sequence import SeqStatus
                seq.status = SeqStatus.PREFILL
                prefill_seqs.append(seq)
                n_admitted += 1

        return SchedulerOutput(
            prefill_seqs   = prefill_seqs,
            decode_seqs    = decode_seqs,
            preempted_seqs = preempted_seqs,
            swapped_in     = swapped_in_seqs,
        )

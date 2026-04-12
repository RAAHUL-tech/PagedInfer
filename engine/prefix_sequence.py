"""
engine/prefix_sequence.py — SequenceGroup variant for prefix-caching engine.

PrefixSequenceGroup extends the base SequenceGroup with two additions:

    n_cached_tokens : int
        Set by PrefixAwareEngine._try_prefix_match() when the request's
        leading blocks are found in the PrefixCache.  Records how many
        prompt tokens were skipped (no forward pass needed for those).

    block_table : PrefixAwareBlockTable  (instead of LayeredBlockTable)
        Supports attach_cached_prefix() for zero-copy prefix reuse.
        swap_in() rebuilds using PrefixAwareBlockTable so the sequence
        can be restored without re-prefilling.

Everything else — swap_out, status transitions, timing metrics — is
identical to the base SequenceGroup in engine/sequence.py.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from engine.sequence import SeqStatus

if TYPE_CHECKING:
    from kv_cache.block_allocator import BlockAllocator
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
    from engine.request import Request


class PrefixSequenceGroup:
    """
    Runtime state for one request in the prefix-caching engine.

    Uses PrefixAwareBlockTable so that cached prefix blocks can be attached
    at admission time without calling allocator.allocate().

    Args:
        request    : The originating Request.
        token_ids  : Tokenized prompt (list of int).
        n_layers   : Transformer depth.
        block_size : Tokens per KV block.
    """

    _id_counter: int = 0

    def __init__(
        self,
        request    : "Request",
        token_ids  : List[int],
        n_layers   : int,
        block_size : int,
    ) -> None:
        from kv_cache.prefix_block_table import PrefixAwareBlockTable

        self.seq_id     = PrefixSequenceGroup._id_counter
        PrefixSequenceGroup._id_counter += 1

        self.request    = request
        self.token_ids  = list(token_ids)
        self.prompt_len = len(token_ids)
        self.n_layers   = n_layers
        self.block_size = block_size

        self.n_generated     : int = 0
        self.status          : SeqStatus = SeqStatus.WAITING
        self.prefill_offset  : int = 0
        self.n_cached_tokens : int = 0   # tokens covered by a prefix cache hit

        self.block_table = PrefixAwareBlockTable(
            seq_id     = self.seq_id,
            n_layers   = n_layers,
            block_size = block_size,
        )

        self.cpu_kv_cache     : Optional[Dict[int, Tuple]] = None
        self.arrival_time     : float          = request.arrival_time
        self.prefill_time     : Optional[float] = None
        self.first_token_time : Optional[float] = None
        self.finish_time      : Optional[float] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def max_new_tokens(self) -> int:
        return self.request.max_new_tokens

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_done(self) -> bool:
        return self.status == SeqStatus.DONE

    @property
    def prefill_remaining(self) -> int:
        return self.prompt_len - self.prefill_offset

    @property
    def prefill_complete(self) -> bool:
        return self.prefill_offset >= self.prompt_len

    @property
    def ttft_ms(self) -> Optional[float]:
        if self.first_token_time and self.arrival_time:
            return (self.first_token_time - self.arrival_time) * 1000.0
        return None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        if self.finish_time and self.arrival_time:
            return (self.finish_time - self.arrival_time) * 1000.0
        return None

    def blocks_needed_next(self) -> int:
        """New logical blocks needed for one more token (0 or 1)."""
        return self.block_table.blocks_needed_next()

    # ── Swap-out (preemption) ─────────────────────────────────────────────────

    def swap_out(
        self,
        kv_cache  : "GPUPagedKVCache",
        allocator : "BlockAllocator",
    ) -> None:
        """
        Preemption: copy KV from GPU pool to CPU RAM, then free GPU blocks.

        Identical to SequenceGroup.swap_out() — reads all layers via the
        kv_read CUDA kernel, pins to CPU RAM, then frees GPU blocks.
        Note: the cached prefix blocks' ref count drops from 2 → 1 (still
        owned by PrefixCache); suffix blocks are fully freed.
        """
        import torch
        self.cpu_kv_cache = {}
        for layer in range(self.n_layers):
            k_gpu, v_gpu = kv_cache.read_sequence(layer, self.block_table)
            self.cpu_kv_cache[layer] = (
                k_gpu.cpu().pin_memory(),
                v_gpu.cpu().pin_memory(),
            )
        torch.cuda.synchronize()
        self.block_table.free(allocator, kv_cache)
        self.status = SeqStatus.SWAPPED

    # ── Swap-in (restore) ─────────────────────────────────────────────────────

    def swap_in(
        self,
        kv_cache  : "GPUPagedKVCache",
        allocator : "BlockAllocator",
    ) -> None:
        """
        Restore: allocate fresh GPU blocks and write saved CPU K/V back.

        Unlike the base class, uses PrefixAwareBlockTable on restore so the
        seq can participate in the prefix engine without type mismatches.
        No re-prefill needed — the K/V was saved to CPU.
        """
        import torch
        from kv_cache.prefix_block_table import PrefixAwareBlockTable

        if self.cpu_kv_cache is None:
            self.block_table    = PrefixAwareBlockTable(self.seq_id, self.n_layers, self.block_size)
            self.prefill_offset = 0
            self.status         = SeqStatus.PREFILL
            return

        n_saved = self.cpu_kv_cache[0][0].shape[0]
        self.block_table = PrefixAwareBlockTable(self.seq_id, self.n_layers, self.block_size)
        self.block_table.append_tokens(n_saved, allocator, kv_cache)

        for layer in range(self.n_layers):
            k_cpu, v_cpu = self.cpu_kv_cache[layer]
            k_gpu = k_cpu.to(kv_cache.device, non_blocking=True)
            v_gpu = v_cpu.to(kv_cache.device, non_blocking=True)
            torch.cuda.synchronize()
            kv_cache.write_tokens(layer, self.block_table, k_gpu, v_gpu, start_pos=0)

        self.cpu_kv_cache = None
        self.status       = SeqStatus.DECODING

    def __repr__(self) -> str:
        chunk = (
            f" chunk={self.prefill_offset}/{self.prompt_len}"
            if self.status == SeqStatus.PREFILL else ""
        )
        cached = f" cached={self.n_cached_tokens}" if self.n_cached_tokens > 0 else ""
        return (
            f"PrefixSeq(id={self.seq_id} {self.status.name}{chunk}{cached} "
            f"tokens={self.total_tokens} gen={self.n_generated})"
        )

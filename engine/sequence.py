"""
engine/sequence.py — SeqStatus and SequenceGroup.

SequenceGroup holds all runtime state for one request as it moves through
the continuous batching engine:

  WAITING → PREFILL → DECODING → DONE
                 ↓         ↓
              SWAPPED ←────┘   (preempted: KV moved to CPU)
                 ↓
             DECODING          (restored: KV written back to GPU)

Swap-out / swap-in (preemption with KV save)
────────────────────────────────────────────
  swap_out(): reads the sequence's K/V from GPU pool to CPU RAM, then
              frees the GPU blocks. Used when the pool is full and a lower-
              priority sequence must yield to a higher-priority one.

  swap_in():  allocates fresh GPU blocks and writes the saved CPU K/V back
              to the pool via CUDA kv_write kernels — no recomputation.
              If no cpu_kv_cache was saved (shouldn't happen in normal flow),
              falls back to re-prefill from scratch.

Chunked prefill
───────────────
  Long prompts are split into chunks of CHUNK_SIZE tokens to prevent one
  sequence from blocking all decoding sequences for many iterations.
  prefill_offset tracks how many prompt tokens have been processed.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from kv_cache.block_allocator import BlockAllocator
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
    from engine.request import Request


class SeqStatus(Enum):
    WAITING  = auto()   # in queue, not yet scheduled
    PREFILL  = auto()   # being prefilled (possibly chunked)
    DECODING = auto()   # actively generating tokens
    SWAPPED  = auto()   # preempted — KV saved to CPU RAM
    DONE     = auto()   # finished (EOS or max_tokens reached)


class SequenceGroup:
    """
    Runtime state for a single inference request.

    The engine creates one SequenceGroup per Request when it is admitted.
    The block_table and kv_cache are external — the engine owns allocation
    and passes them to swap_out / swap_in as needed.

    Args:
        request    : The originating Request.
        token_ids  : Tokenized prompt (list of int).
        n_layers   : Transformer depth (used to build LayeredBlockTable).
        block_size : Tokens per KV block (must match pool configuration).
    """

    _id_counter: int = 0

    def __init__(
        self,
        request    : "Request",
        token_ids  : List[int],
        n_layers   : int,
        block_size : int,
    ) -> None:
        from kv_cache.block_table import LayeredBlockTable

        self.seq_id      = SequenceGroup._id_counter
        SequenceGroup._id_counter += 1

        self.request     = request
        self.token_ids   = list(token_ids)
        self.prompt_len  = len(token_ids)
        self.n_layers    = n_layers
        self.block_size  = block_size

        self.n_generated : int = 0
        self.status      : SeqStatus = SeqStatus.WAITING

        self.block_table = LayeredBlockTable(
            seq_id     = self.seq_id,
            n_layers   = n_layers,
            block_size = block_size,
        )

        # Chunked prefill: how many prompt tokens have been processed
        self.prefill_offset: int = 0

        # CPU KV storage used during preemption: {layer_idx: (k_cpu, v_cpu)}
        # Each tensor: (n_saved_tokens, n_kv_heads, head_dim) fp16 on CPU
        self.cpu_kv_cache: Optional[Dict[int, Tuple]] = None

        # Timing metrics
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
        """Number of prompt tokens not yet prefilled."""
        return self.prompt_len - self.prefill_offset

    @property
    def prefill_complete(self) -> bool:
        return self.prefill_offset >= self.prompt_len

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time-to-first-token in milliseconds."""
        if self.first_token_time and self.arrival_time:
            return (self.first_token_time - self.arrival_time) * 1000.0
        return None

    @property
    def e2e_latency_ms(self) -> Optional[float]:
        """End-to-end latency in milliseconds."""
        if self.finish_time and self.arrival_time:
            return (self.finish_time - self.arrival_time) * 1000.0
        return None

    def blocks_needed_next(self) -> int:
        """
        Number of NEW logical blocks needed to accommodate one more token.
        0 if there is free space in the last block, 1 otherwise.
        Multiply by n_layers to get physical block IDs needed.
        """
        return self.block_table.new_blocks_for_tokens(1)

    # ── Swap-out (preemption) ─────────────────────────────────────────────────

    def swap_out(
        self,
        kv_cache  : "GPUPagedKVCache",
        allocator : "BlockAllocator",
    ) -> None:
        """
        Preemption: copy KV from GPU pool to CPU RAM, then free GPU blocks.

        For each layer, kv_cache.read_sequence() gathers this sequence's
        K and V tensors from the paged pool (CUDA kv_read kernel), which
        are then moved to pinned CPU RAM for fast DMA transfer on swap-in.

        After this call:
          - GPU blocks are freed and available for other sequences.
          - cpu_kv_cache stores {layer: (k_cpu, v_cpu)} for all layers.
          - status = SWAPPED.

        CPU memory cost: n_layers × 2 × num_tokens × n_kv_heads × head_dim × 2 B
        """
        import torch
        n_saved = self.block_table.num_tokens
        self.cpu_kv_cache = {}

        for layer in range(self.n_layers):
            k_gpu, v_gpu = kv_cache.read_sequence(layer, self.block_table)
            # Pin memory for faster GPU→CPU→GPU transfers
            self.cpu_kv_cache[layer] = (
                k_gpu.cpu().pin_memory(),   # (n_saved, n_kv_heads, head_dim) fp16
                v_gpu.cpu().pin_memory(),
            )

        torch.cuda.synchronize()  # ensure all reads finished before freeing
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

        For each layer, the saved CPU tensors are moved back to GPU and
        written to the newly allocated blocks via kv_cache.write_tokens()
        (CUDA kv_write kernel).  No re-prefill needed.

        After this call:
          - GPU blocks are allocated for all saved tokens.
          - cpu_kv_cache is cleared.
          - status = DECODING (can continue generating immediately).

        Falls back to re-prefill if cpu_kv_cache is missing (shouldn't happen).
        """
        import torch
        from kv_cache.block_table import LayeredBlockTable

        if self.cpu_kv_cache is None:
            # Fallback: lost the CPU cache, re-prefill from scratch
            self.block_table    = LayeredBlockTable(self.seq_id, self.n_layers, self.block_size)
            self.prefill_offset = 0
            self.status         = SeqStatus.PREFILL
            return

        n_saved = self.cpu_kv_cache[0][0].shape[0]

        # Allocate fresh GPU blocks for all saved tokens
        self.block_table = LayeredBlockTable(self.seq_id, self.n_layers, self.block_size)
        self.block_table.append_tokens(n_saved, allocator, kv_cache)

        # Write each layer's K/V back to the pool
        for layer in range(self.n_layers):
            k_cpu, v_cpu = self.cpu_kv_cache[layer]
            k_gpu = k_cpu.to(kv_cache.device, non_blocking=True)
            v_gpu = v_cpu.to(kv_cache.device, non_blocking=True)
            torch.cuda.synchronize()
            kv_cache.write_tokens(layer, self.block_table, k_gpu, v_gpu, start_pos=0)

        self.cpu_kv_cache = None
        self.status       = SeqStatus.DECODING

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        chunk = (
            f" chunk={self.prefill_offset}/{self.prompt_len}"
            if self.status == SeqStatus.PREFILL else ""
        )
        return (
            f"Seq(id={self.seq_id} {self.status.name}{chunk} "
            f"tokens={self.total_tokens} gen={self.n_generated})"
        )

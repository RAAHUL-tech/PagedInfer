"""
engine/ — Continuous batching inference engine for PagedInfer.
──────────────────────────────────────────────────────────────
Files in this package:

    request.py           Request dataclass + RequestQueue (priority min-heap)
    sequence.py          SeqStatus enum + SequenceGroup (per-request runtime state:
                         token list, block_table, KV swap-out/in, latency metrics)
    scheduler.py         Scheduler — every iteration decides which sequences prefill,
                         decode, get preempted, or get restored from CPU KV swap
    forward_pass.py      forward_unified() — one batched transformer forward pass
                         for ALL active sequences (prefill + decode) per iteration
    continuous_engine.py ContinuousBatchingEngine — top-level engine loop + metrics
    prefix_sequence.py   PrefixSequenceGroup — SequenceGroup variant that uses
                         PrefixAwareBlockTable and tracks n_cached_tokens
    prefix_engine.py     PrefixAwareEngine — extends ContinuousBatchingEngine with
                         hash-based prefix KV reuse; skips forward pass for cached
                         prompt prefixes shared across requests

Sequence lifecycle:
    Request → WAITING → PREFILL → DECODING → DONE
                             ↓         ↓
                          SWAPPED ←────┘   (preempted: K/V moved to CPU RAM)
                             ↓
                          DECODING          (restored: K/V written back to GPU)

CUDA kernel used per iteration:
    attn_continuous.cu  (kernels/attn_continuous_ops.py)
      Grid = (TOTAL_Q_TOKENS, N_HEADS) — one block per (query_token, head)
      Block = (HEAD_DIM,)
      Handles causal masking, GQA, and online softmax inside the kernel.
      Processes ALL sequences (prefill + decode) in one launch.

See inference/serve_continuous.py for a complete runnable example.
"""

from engine.request import Request, RequestQueue
from engine.sequence import SeqStatus, SequenceGroup
from engine.scheduler import Scheduler, SchedulerOutput, DEFAULT_CHUNK_SIZE
from engine.continuous_engine import ContinuousBatchingEngine, EngineMetrics
from engine.prefix_sequence import PrefixSequenceGroup
from engine.prefix_engine import PrefixAwareEngine, PrefixEngineMetrics

__all__ = [
    "Request",
    "RequestQueue",
    "SeqStatus",
    "SequenceGroup",
    "Scheduler",
    "SchedulerOutput",
    "DEFAULT_CHUNK_SIZE",
    "ContinuousBatchingEngine",
    "EngineMetrics",
    "PrefixSequenceGroup",
    "PrefixAwareEngine",
    "PrefixEngineMetrics",
]

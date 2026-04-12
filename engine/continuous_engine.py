"""
engine/continuous_engine.py
────────────────────────────
ContinuousBatchingEngine — the top-level inference engine that keeps the GPU
busy by interleaving prefill and decode across multiple requests simultaneously.

What is continuous batching?
    Traditional (static) batching: process one batch of requests fully before
    starting the next.  Sequences that finish early leave GPU compute wasted.

    Continuous batching: every iteration the scheduler selects a new set of
    active sequences — decode slots are freed the moment a sequence finishes
    and immediately filled by the next waiting request.  GPU utilisation stays
    high throughout.

One engine iteration:
    1. scheduler.schedule()        — decide which seqs prefill / decode / preempt
    2. Block allocation            — append_token() / append_tokens() for active seqs
    3. engine/forward_pass.py      — one batched forward pass for ALL active seqs
                                     (single CUDA kernel for attention, single matmul
                                      for QKV, one K/V write for all tokens)
    4. Sampling                    — temperature / top-k / top-p per sequence
    5. State updates               — advance token lists, mark finished seqs
    6. Metrics                     — throughput, TTFT, E2E latency, preemptions

Tracked metrics (EngineMetrics):
    throughput_tok_s  tokens generated per second
    avg_ttft_ms       average time-to-first-token across finished requests
    avg_e2e_ms        average end-to-end latency across finished requests
    n_preemptions     times a sequence was evicted from GPU to free pool space
    n_swap_ins        times a preempted sequence was restored from CPU KV cache

Usage:
    engine = ContinuousBatchingEngine.build(
        checkpoint  = "model_checkpoint/llama_ckpt.pt",
        eos_token_id= tokenizer.eos_token_id,
        device      = "cuda",
        block_size  = 16,
        max_batch   = 8,
    )
    engine.add_request(request, token_ids)
    finished_seqs = engine.run_until_done()

See inference/serve_continuous.py for a complete runnable example.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from engine.request import Request
from engine.scheduler import DEFAULT_CHUNK_SIZE, Scheduler, SchedulerOutput
from engine.sequence import SeqStatus, SequenceGroup
from engine.forward_pass import forward_unified


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample(
    logits      : torch.Tensor,
    temperature : float = 1.0,
    top_k       : int   = 50,
    top_p       : float = 1.0,
) -> int:
    if temperature < 1e-5:
        return int(logits.argmax().item())
    logits = logits.float() / temperature
    if top_k > 0:
        k     = min(top_k, logits.size(-1))
        thresh, _ = torch.topk(logits, k)
        logits[logits < thresh[-1]] = float("-inf")
    if top_p < 1.0:
        probs        = F.softmax(logits, dim=-1)
        sorted_p, si = torch.sort(probs, descending=True)
        cum          = torch.cumsum(sorted_p, dim=-1)
        mask         = (cum - sorted_p) >= top_p
        sorted_p[mask] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, si, sorted_p)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EngineMetrics:
    """Accumulated statistics for one engine run."""
    iteration           : int   = 0
    total_tokens_gen    : int   = 0
    total_requests_done : int   = 0
    n_preemptions       : int   = 0
    n_swap_ins          : int   = 0
    n_chunked_steps     : int   = 0
    start_time          : float = field(default_factory=time.time)

    ttft_ms_list  : List[float] = field(default_factory=list)
    e2e_ms_list   : List[float] = field(default_factory=list)
    iter_times_s  : List[float] = field(default_factory=list)
    batch_sizes   : List[int]   = field(default_factory=list)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    @property
    def throughput_tok_s(self) -> float:
        return self.total_tokens_gen / max(self.elapsed_s, 1e-6)

    @property
    def avg_ttft_ms(self) -> float:
        return float(sum(self.ttft_ms_list) / len(self.ttft_ms_list)) if self.ttft_ms_list else 0.0

    @property
    def avg_e2e_ms(self) -> float:
        return float(sum(self.e2e_ms_list) / len(self.e2e_ms_list)) if self.e2e_ms_list else 0.0

    def print_summary(self) -> None:
        print("\n" + "=" * 65)
        print("  ENGINE METRICS SUMMARY")
        print("=" * 65)
        print(f"  Total iterations     : {self.iteration}")
        print(f"  Total tokens gen     : {self.total_tokens_gen}")
        print(f"  Total requests done  : {self.total_requests_done}")
        print(f"  Elapsed              : {self.elapsed_s:.2f} s")
        print(f"  Throughput           : {self.throughput_tok_s:.1f} tok/s")
        print(f"  Avg TTFT             : {self.avg_ttft_ms:.1f} ms")
        print(f"  Avg E2E latency      : {self.avg_e2e_ms:.1f} ms")
        print(f"  Preemptions          : {self.n_preemptions}")
        print(f"  Swap-ins             : {self.n_swap_ins}")
        print(f"  Chunked prefill steps: {self.n_chunked_steps}")
        if self.batch_sizes:
            avg_bs = sum(self.batch_sizes) / len(self.batch_sizes)
            print(f"  Avg batch size       : {avg_bs:.1f}")
        print("=" * 65)


# ── Engine ────────────────────────────────────────────────────────────────────

class ContinuousBatchingEngine:
    """
    Continuous batching inference engine.

    Manages the full lifecycle from Request admission to finished text:
      - Request queueing (priority heap)
      - Sequence state machine (WAITING → PREFILL → DECODING → DONE)
      - Block allocation via BlockAllocator
      - GPU KV pool via GPUPagedKVCache
      - Scheduler for iteration-level decisions
      - Unified batched forward pass (prefill + decode in one kernel call)
      - CPU KV swap for preempted sequences

    Create via ContinuousBatchingEngine.build() for the standard GPU setup.
    """

    def __init__(
        self,
        model        ,                 # Transformer
        allocator    ,                 # BlockAllocator
        kv_cache     ,                 # GPUPagedKVCache
        eos_token_id : int,
        n_layers     : int,
        block_size   : int,
        device       : str,
        max_batch    : int = 8,
        max_prefill  : int = 4,
        chunk_size   : int = DEFAULT_CHUNK_SIZE,
        log_every    : int = 10,
    ) -> None:
        from kernels.attn_continuous_ops import load_unified_attn_kernels

        self.model        = model
        self.allocator    = allocator
        self.kv_cache     = kv_cache
        self.eos_token_id = eos_token_id
        self.n_layers     = n_layers
        self.block_size   = block_size
        self.device       = device
        self.chunk_size   = chunk_size
        self.log_every    = log_every

        self.scheduler = Scheduler(
            allocator        = allocator,
            n_layers         = n_layers,
            block_size       = block_size,
            max_batch_size   = max_batch,
            max_prefill_seqs = max_prefill,
            chunk_size       = chunk_size,
        )

        # Compile unified attention kernel
        self.attn_ops = load_unified_attn_kernels(device=device, verbose=True)

        # Queue: (priority, counter, SequenceGroup)
        self._waiting_heap : List = []
        self._heap_counter : int  = 0
        self.running       : List[SequenceGroup] = []
        self.swapped       : List[SequenceGroup] = []
        self.finished      : List[SequenceGroup] = []
        self.metrics       = EngineMetrics()

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        checkpoint   : str,
        eos_token_id : int,
        device       : str   = "cuda",
        block_size   : int   = 16,
        kv_budget_gb : Optional[float] = None,
        max_batch    : int   = 8,
        max_prefill  : int   = 4,
        chunk_size   : int   = DEFAULT_CHUNK_SIZE,
        log_every    : int   = 10,
        verbose      : bool  = True,
    ) -> "ContinuousBatchingEngine":
        """
        Build an engine from a checkpoint file.

        Loads the model, sizes the GPU KV pool from available VRAM, compiles
        CUDA kernels, and returns a ready-to-use engine.

        Args:
            checkpoint   : Path to .pt checkpoint file.
            eos_token_id : Token ID for end-of-sequence detection.
            device       : CUDA device (e.g. "cuda", "cuda:0").
            block_size   : Tokens per KV block.
            kv_budget_gb : GB to allocate for KV pool
                           (default: all free VRAM minus 1.25 GB reserve).
            max_batch    : Max sequences active per iteration.
            max_prefill  : Max new sequences per iteration.
            chunk_size   : Tokens per prefill chunk.
            log_every    : Print progress every N iterations.
            verbose      : Show kernel compilation progress.
        """
        import gc
        from inference.model_loader import _remap_state_dict, _MODEL_FIELDS
        from kv_cache.block_allocator import BlockAllocator
        from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
        from models import ModelConfig, Transformer

        if not torch.cuda.is_available():
            raise RuntimeError("ContinuousBatchingEngine requires CUDA.")

        # Load model
        torch.cuda.empty_cache(); gc.collect()
        raw        = torch.load(checkpoint, map_location=device, weights_only=True)
        raw_config = raw.get("config", {}) if isinstance(raw, dict) else {}
        state_dict = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        step       = raw.get("step", "?") if isinstance(raw, dict) else "?"

        kwargs                  = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
        kwargs["attention_type"] = "paged"
        cfg   = ModelConfig(**kwargs)
        model = Transformer(cfg)
        model.load_state_dict(_remap_state_dict(state_dict))
        model = model.half().to(device).eval()

        if verbose:
            print(f"[Engine] Loaded {checkpoint}  step={step}  "
                  f"{cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h")

        # Size KV pool
        torch.cuda.synchronize(device)
        if kv_budget_gb is not None:
            kv_bytes = int(kv_budget_gb * 1024 ** 3)
        else:
            free, _ = torch.cuda.mem_get_info(device)
            reserve = int(1.5 * 1024 ** 3)   # 1.5 GB: activations + overhead
            kv_bytes = max(0, free - reserve)

        kv_cache = GPUPagedKVCache.from_memory_budget(
            free_gpu_bytes = kv_bytes,
            n_layers       = cfg.n_layers,
            block_size     = block_size,
            n_kv_heads     = cfg.n_kv_heads,
            head_dim       = cfg.head_dim,
            device         = device,
            verbose        = verbose,
        )
        allocator = BlockAllocator(kv_cache.n_phys)

        if verbose:
            print(f"[Engine] KV pool: {kv_cache.memory_pool_mb():.0f} MB  "
                  f"({kv_cache.n_phys} physical blocks, "
                  f"{kv_cache.n_phys // cfg.n_layers} logical)")

        return cls(
            model        = model,
            allocator    = allocator,
            kv_cache     = kv_cache,
            eos_token_id = eos_token_id,
            n_layers     = cfg.n_layers,
            block_size   = block_size,
            device       = device,
            max_batch    = max_batch,
            max_prefill  = max_prefill,
            chunk_size   = chunk_size,
            log_every    = log_every,
        )

    def add_request(self, request: Request, token_ids: List[int]) -> int:
        """
        Enqueue a request.  Returns the sequence id.

        Args:
            request   : Request metadata (prompt text, params, priority).
            token_ids : Pre-tokenized prompt (caller handles tokenization).
        """
        seq = SequenceGroup(
            request    = request,
            token_ids  = token_ids,
            n_layers   = self.n_layers,
            block_size = self.block_size,
        )
        heapq.heappush(
            self._waiting_heap,
            (request.priority, self._heap_counter, seq)
        )
        self._heap_counter += 1
        return seq.seq_id

    def has_work(self) -> bool:
        """True if there are any requests in progress or queued."""
        return bool(self._waiting_heap or self.running or self.swapped)

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self) -> List[Tuple[int, str, List[int]]]:
        """
        Run one engine iteration.

        Returns a list of (seq_id, status, token_ids) tuples for sequences
        that finished this step (status="done"), or empty list if none finished.
        """
        t0 = time.perf_counter()
        self.metrics.iteration += 1

        # Flatten waiting heap to list (sorted by priority)
        waiting_seqs = [item[2] for item in sorted(self._waiting_heap)]

        sched_out = self.scheduler.schedule(
            waiting  = waiting_seqs,
            running  = self.running,
            swapped  = self.swapped,
            kv_cache = self.kv_cache,
        )

        # ── Update swapped list (preemptions happened in scheduler) ───────────
        for seq in sched_out.preempted_seqs:
            if seq in self.running:
                self.running.remove(seq)
            if seq not in self.swapped:
                self.swapped.append(seq)
            self.metrics.n_preemptions += 1

        # ── Remove restored seqs from swapped list ────────────────────────────
        for seq in sched_out.swapped_in:
            if seq in self.swapped:
                self.swapped.remove(seq)
            self.metrics.n_swap_ins += 1

        # ── Remove admitted prefill seqs from waiting heap ────────────────────
        admitted_ids = {s.seq_id for s in sched_out.prefill_seqs}
        self._waiting_heap = [
            item for item in self._waiting_heap
            if item[2].seq_id not in admitted_ids
        ]
        heapq.heapify(self._waiting_heap)

        # ── Allocate decode blocks ─────────────────────────────────────────────
        active_decode: List[SequenceGroup] = []
        for seq in sched_out.decode_seqs:
            if seq.is_done:
                continue
            new_logical = seq.blocks_needed_next()
            phys_needed = new_logical * self.n_layers
            if phys_needed > 0 and not self.allocator.can_allocate(phys_needed):
                # Emergency preemption
                candidates = [s for s in active_decode if not s.is_done]
                if candidates:
                    victim = min(candidates, key=lambda s: s.n_generated)
                    active_decode.remove(victim)
                    victim.swap_out(self.kv_cache, self.allocator)
                    self.swapped.append(victim)
                    if victim in self.running:
                        self.running.remove(victim)
                    self.metrics.n_preemptions += 1
                else:
                    continue   # skip this decode seq this iteration
            seq.block_table.append_token(self.allocator, self.kv_cache)
            active_decode.append(seq)

        # ── Allocate prefill blocks ────────────────────────────────────────────
        active_prefill     : List[SequenceGroup] = []
        prefill_chunk_info : List[Tuple[int,int,bool]] = []  # (start, end, is_last)

        for seq in sched_out.prefill_seqs:
            if seq.prefill_time is None:
                seq.prefill_time = time.time()

            cs      = seq.prefill_offset
            ce      = min(cs + self.chunk_size, seq.prompt_len)
            clen    = ce - cs
            is_last = (ce >= seq.prompt_len)
            phys_needed = math.ceil(clen / self.block_size) * self.n_layers

            if not self.allocator.can_allocate(phys_needed):
                # No room — push back to waiting
                if not any(item[2].seq_id == seq.seq_id for item in self._waiting_heap):
                    heapq.heappush(
                        self._waiting_heap,
                        (seq.request.priority, self._heap_counter, seq)
                    )
                    self._heap_counter += 1
                seq.status = SeqStatus.WAITING
                continue

            seq.block_table.append_tokens(clen, self.allocator, self.kv_cache)
            active_prefill.append(seq)
            prefill_chunk_info.append((cs, ce, is_last))

        if is_chunk := any(not info[2] for info in prefill_chunk_info):
            self.metrics.n_chunked_steps += 1

        # ── ONE unified forward pass ──────────────────────────────────────────
        finished_this_step: List[Tuple[int, str, List[int]]] = []

        if active_prefill or active_decode:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prefill_logits, decode_logits = forward_unified(
                    model        = self.model,
                    prefill_seqs = active_prefill,
                    decode_seqs  = active_decode,
                    kv_cache     = self.kv_cache,
                    attn_ops     = self.attn_ops,
                    device       = self.device,
                    chunk_size   = self.chunk_size,
                )

            # ── Process prefill results ───────────────────────────────────────
            for seq, logits, (cs, ce, is_last) in zip(
                    active_prefill, prefill_logits, prefill_chunk_info):

                seq.prefill_offset = ce

                if is_last:
                    next_tok = _sample(
                        logits[-1],
                        seq.request.temperature,
                        seq.request.top_k,
                        seq.request.top_p,
                    )
                    seq.token_ids.append(next_tok)
                    seq.n_generated      += 1
                    seq.first_token_time  = time.time()
                    seq.status            = SeqStatus.DECODING
                    self.metrics.total_tokens_gen += 1

                    done = (
                        next_tok == self.eos_token_id
                        or seq.n_generated >= seq.max_new_tokens
                    )
                    if done:
                        finished_this_step.append(
                            self._finish(seq)
                        )
                    else:
                        if seq not in self.running:
                            self.running.append(seq)
                else:
                    # More chunks remain — push back to the waiting heap so
                    # it is scheduled again next iteration.  Slightly elevated
                    # priority (- 0.5) so it resumes before brand-new requests.
                    seq.status = SeqStatus.PREFILL
                    if not any(item[2].seq_id == seq.seq_id
                               for item in self._waiting_heap):
                        heapq.heappush(
                            self._waiting_heap,
                            (seq.request.priority - 0.5, self._heap_counter, seq),
                        )
                        self._heap_counter += 1

            # ── Process decode results ────────────────────────────────────────
            for seq, logits in zip(active_decode, decode_logits):
                if seq.is_done:
                    continue
                next_tok = _sample(
                    logits,
                    seq.request.temperature,
                    seq.request.top_k,
                    seq.request.top_p,
                )
                seq.token_ids.append(next_tok)
                seq.n_generated      += 1
                self.metrics.total_tokens_gen += 1

                done = (
                    next_tok == self.eos_token_id
                    or seq.n_generated >= seq.max_new_tokens
                )
                if done:
                    finished_this_step.append(self._finish(seq))

        # ── Logging ───────────────────────────────────────────────────────────
        iter_t = time.perf_counter() - t0
        self.metrics.iter_times_s.append(iter_t)
        batch_sz = len(active_prefill) + len(active_decode)
        self.metrics.batch_sizes.append(batch_sz)

        if self.metrics.iteration % self.log_every == 0:
            print(
                f"[Engine] iter={self.metrics.iteration:4d}  "
                f"batch={batch_sz}  "
                f"running={len(self.running)}  "
                f"waiting={len(self._waiting_heap)}  "
                f"swapped={len(self.swapped)}  "
                f"done={self.metrics.total_requests_done}  "
                f"tok/s={self.metrics.throughput_tok_s:.1f}  "
                f"iter={iter_t*1000:.1f}ms"
            )

        return finished_this_step

    # ── Run-to-completion ─────────────────────────────────────────────────────

    def run_until_done(self) -> List[SequenceGroup]:
        """
        Run the engine until all queued requests are finished.

        Returns:
            List of SequenceGroup objects (in finish order).
            Text is: tokenizer.decode(seq.token_ids[seq.prompt_len:])
        """
        while self.has_work():
            self.step()
        return self.finished

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finish(self, seq: SequenceGroup) -> Tuple[int, str, List[int]]:
        seq.status      = SeqStatus.DONE
        seq.finish_time = time.time()
        seq.block_table.free(self.allocator, self.kv_cache)
        if seq in self.running:
            self.running.remove(seq)
        self.finished.append(seq)
        self.metrics.total_requests_done += 1
        if seq.ttft_ms is not None:
            self.metrics.ttft_ms_list.append(seq.ttft_ms)
        if seq.e2e_latency_ms is not None:
            self.metrics.e2e_ms_list.append(seq.e2e_latency_ms)
        return (seq.seq_id, "done", seq.token_ids)

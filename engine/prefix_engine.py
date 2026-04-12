"""
engine/prefix_engine.py — continuous batching engine with prefix KV caching.

Extends ContinuousBatchingEngine with two hooks that eliminate redundant
prefill compute when multiple requests share a common prefix:

    _try_prefix_match(seq)
    ──────────────────────
    Called once per sequence the first time it is scheduled for prefill
    (prefill_offset == 0 and n_cached_tokens == 0).

    prefix_cache.match(token_ids, block_size, n_layers)
        → (n_cached_blocks, cached_ids_per_layer)

    If n_cached_blocks > 0:
        seq.block_table.attach_cached_prefix(cached_ids, n_cached_blocks * block_size)
        seq.prefill_offset  = n_cached_blocks * block_size   ← skip forward pass
        seq.n_cached_tokens = prefill_offset

    Special case — fully cached prompt:
        If prefill_offset == prompt_len after match, the ENTIRE prompt was in
        the cache.  No forward pass is needed at all.  The seq is moved
        directly to DECODING status with TTFT recorded immediately.

    _insert_into_cache(seq)
    ───────────────────────
    Called once per sequence immediately after the last prefill chunk completes.

    prefix_cache.insert_sequence_blocks(
        seq.token_ids[:seq.prompt_len],
        seq.block_table,
        block_size
    )

    This stores all complete prompt blocks in the cache so future requests
    with the same prefix can skip the forward pass for those tokens.

Prefill path comparison
───────────────────────
    ContinuousBatchingEngine        PrefixAwareEngine
    ────────────────────────        ─────────────────
    allocate blocks for full        1. match prefix cache
    prompt, run forward on all      2. attach cached blocks (no alloc, no fwd)
    prompt tokens                   3. allocate blocks for suffix only
                                    4. run forward on suffix tokens only
                                    5. insert blocks into cache for next time

Build
─────
    engine = PrefixAwareEngine.build(
        checkpoint   = "model_checkpoint/llama_ckpt.pt",
        eos_token_id = tokenizer.eos_token_id,
        device       = "cuda",
        block_size   = 16,
        cache_fraction = 0.5,   # fraction of logical pool for prefix cache
    )
    engine.add_request(request, token_ids)
    finished_seqs = engine.run_until_done()

See inference/generate_prefix_cached.py for a complete runnable example.
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
from engine.sequence import SeqStatus
from engine.prefix_sequence import PrefixSequenceGroup
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
class PrefixEngineMetrics:
    """Accumulated statistics for one PrefixAwareEngine run."""
    iteration           : int   = 0
    total_tokens_gen    : int   = 0
    total_requests_done : int   = 0
    n_preemptions       : int   = 0
    n_swap_ins          : int   = 0
    n_chunked_steps     : int   = 0
    n_fully_cached      : int   = 0   # requests whose entire prompt was in cache
    start_time          : float = field(default_factory=time.time)

    ttft_ms_list  : List[float] = field(default_factory=list)
    e2e_ms_list   : List[float] = field(default_factory=list)
    iter_times_s  : List[float] = field(default_factory=list)
    batch_sizes   : List[int]   = field(default_factory=list)
    cached_tok_counts : List[int] = field(default_factory=list)  # tokens saved per hit

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

    @property
    def total_cached_tokens(self) -> int:
        return sum(self.cached_tok_counts)

    def print_summary(self, prefix_cache=None) -> None:
        print("\n" + "=" * 65)
        print("  PREFIX-CACHING ENGINE METRICS")
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
        print(f"  Fully cached prompts : {self.n_fully_cached}")
        print(f"  Prefix tokens saved  : {self.total_cached_tokens}")
        if self.batch_sizes:
            print(f"  Avg batch size       : {sum(self.batch_sizes)/len(self.batch_sizes):.1f}")
        if prefix_cache is not None:
            print(f"  Cache blocks stored  : {prefix_cache.n_cached}")
            print(f"  Cache hits           : {prefix_cache.n_hits}")
            print(f"  Cache misses         : {prefix_cache.n_misses}")
            print(f"  Cache hit rate       : {prefix_cache.hit_rate:.1%}")
            print(f"  Blocks evicted       : {prefix_cache.n_evicted}")
        print("=" * 65)


# ── Engine ────────────────────────────────────────────────────────────────────

class PrefixAwareEngine:
    """
    Continuous batching engine with hash-based prefix KV caching.

    Manages the full lifecycle from Request admission to finished text,
    with an additional prefix-match step at admission that can skip the
    forward pass for shared prompt prefixes entirely.

    Create via PrefixAwareEngine.build() for the standard GPU setup.
    """

    def __init__(
        self,
        model          ,                  # Transformer
        allocator      ,                  # BlockAllocator
        kv_cache       ,                  # GPUPagedKVCache
        prefix_cache   ,                  # PrefixCache
        eos_token_id   : int,
        n_layers       : int,
        block_size     : int,
        device         : str,
        max_batch      : int   = 8,
        max_prefill    : int   = 4,
        chunk_size     : int   = DEFAULT_CHUNK_SIZE,
        log_every      : int   = 10,
    ) -> None:
        from kernels.attn_continuous_ops import load_unified_attn_kernels

        self.model        = model
        self.allocator    = allocator
        self.kv_cache     = kv_cache
        self.prefix_cache = prefix_cache
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

        self.attn_ops = load_unified_attn_kernels(device=device, verbose=True)

        self._waiting_heap : List = []
        self._heap_counter : int  = 0
        self.running       : List[PrefixSequenceGroup] = []
        self.swapped       : List[PrefixSequenceGroup] = []
        self.finished      : List[PrefixSequenceGroup] = []
        self.metrics       = PrefixEngineMetrics()

    # ── Public API ────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        checkpoint     : str,
        eos_token_id   : int,
        device         : str   = "cuda",
        block_size     : int   = 16,
        kv_budget_gb   : Optional[float] = None,
        cache_fraction : float = 0.5,
        max_batch      : int   = 8,
        max_prefill    : int   = 4,
        chunk_size     : int   = DEFAULT_CHUNK_SIZE,
        log_every      : int   = 10,
        verbose        : bool  = True,
    ) -> "PrefixAwareEngine":
        """
        Build a PrefixAwareEngine from a checkpoint file.

        Args:
            checkpoint     : Path to .pt checkpoint.
            eos_token_id   : Token ID for end-of-sequence detection.
            device         : CUDA device string.
            block_size     : Tokens per KV block.
            kv_budget_gb   : GB for the KV pool (default: free VRAM - 1.5 GB).
            cache_fraction : Fraction of the logical block pool reserved for the
                             prefix cache (default: 0.5 = half the pool).
            max_batch      : Max active sequences per iteration.
            max_prefill    : Max new sequences admitted per iteration.
            chunk_size     : Max prompt tokens per prefill chunk.
            log_every      : Print progress every N iterations.
            verbose        : Show kernel compilation progress.
        """
        import gc
        from inference.model_loader import _remap_state_dict, _MODEL_FIELDS
        from kv_cache.block_allocator import BlockAllocator
        from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
        from kv_cache.prefix_cache import PrefixCache
        from models import ModelConfig, Transformer

        if not torch.cuda.is_available():
            raise RuntimeError("PrefixAwareEngine requires CUDA.")

        torch.cuda.empty_cache(); 
        gc.collect()
        raw        = torch.load(checkpoint, map_location=device, weights_only=True)
        raw_config = raw.get("config", {}) if isinstance(raw, dict) else {}
        state_dict = raw.get("model_state", raw) if isinstance(raw, dict) else raw
        step       = raw.get("step", "?") if isinstance(raw, dict) else "?"

        kwargs                   = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
        kwargs["attention_type"] = "paged"
        cfg   = ModelConfig(**kwargs)
        model = Transformer(cfg)
        model.load_state_dict(_remap_state_dict(state_dict))
        model = model.half().to(device).eval()

        if verbose:
            print(f"[PrefixEngine] Loaded {checkpoint}  step={step}  "
                  f"{cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h")

        torch.cuda.synchronize(device)
        if kv_budget_gb is not None:
            kv_bytes = int(kv_budget_gb * 1024 ** 3)
        else:
            free, _ = torch.cuda.mem_get_info(device)
            reserve  = int(1.5 * 1024 ** 3)
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

        # Prefix cache: at most cache_fraction of logical blocks
        n_logical    = kv_cache.n_phys // cfg.n_layers
        max_cached   = max(1, int(n_logical * cache_fraction))
        prefix_cache = PrefixCache(allocator, max_blocks=max_cached)

        if verbose:
            print(f"[PrefixEngine] KV pool: {kv_cache.memory_pool_mb():.0f} MB  "
                  f"({kv_cache.n_phys} phys blocks, {n_logical} logical)  "
                  f"prefix cache cap: {max_cached} blocks")

        return cls(
            model          = model,
            allocator      = allocator,
            kv_cache       = kv_cache,
            prefix_cache   = prefix_cache,
            eos_token_id   = eos_token_id,
            n_layers       = cfg.n_layers,
            block_size     = block_size,
            device         = device,
            max_batch      = max_batch,
            max_prefill    = max_prefill,
            chunk_size     = chunk_size,
            log_every      = log_every,
        )

    def add_request(self, request: Request, token_ids: List[int]) -> int:
        """Enqueue a request. Returns the sequence id."""
        seq = PrefixSequenceGroup(
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
        return bool(self._waiting_heap or self.running or self.swapped)

    # ── Prefix cache hooks ────────────────────────────────────────────────────

    def _try_prefix_match(self, seq: PrefixSequenceGroup) -> int:
        """
        Check the prefix cache before allocating blocks for a new sequence.

        If matching blocks are found:
          - They are attached to seq.block_table (no allocator.allocate() call)
          - seq.prefill_offset advances past the cached tokens
          - The forward pass will only run on the uncached suffix

        Returns the number of tokens covered by the cache hit (0 if miss).
        """
        n_matched, cached_ids = self.prefix_cache.match(
            seq.token_ids, self.block_size, self.n_layers
        )
        if n_matched > 0:
            n_tok = n_matched * self.block_size
            seq.block_table.attach_cached_prefix(cached_ids, n_tok)
            seq.prefill_offset  = n_tok
            seq.n_cached_tokens = n_tok
            self.metrics.cached_tok_counts.append(n_tok)
        return seq.n_cached_tokens

    def _insert_into_cache(self, seq: PrefixSequenceGroup) -> None:
        """
        Insert all complete prompt blocks into the prefix cache after prefill.

        Only fully written blocks are stored (partial trailing block excluded).
        Future requests sharing this prefix will skip the forward pass for these
        tokens entirely.
        """
        self.prefix_cache.insert_sequence_blocks(
            seq.token_ids[:seq.prompt_len],
            seq.block_table,
            self.block_size,
        )

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self) -> List[Tuple[int, str, List[int]]]:
        """Run one engine iteration. Returns list of (seq_id, 'done', token_ids)."""
        t0 = time.perf_counter()
        self.metrics.iteration += 1

        waiting_seqs = [item[2] for item in sorted(self._waiting_heap)]

        sched_out = self.scheduler.schedule(
            waiting  = waiting_seqs,
            running  = self.running,
            swapped  = self.swapped,
            kv_cache = self.kv_cache,
        )

        # ── Track preemptions ─────────────────────────────────────────────────
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

        # ── Allocate decode slots ─────────────────────────────────────────────
        active_decode: List[PrefixSequenceGroup] = []
        for seq in sched_out.decode_seqs:
            if seq.is_done:
                continue
            new_logical = seq.blocks_needed_next()
            phys_needed = new_logical * self.n_layers
            if phys_needed > 0 and not self.allocator.can_allocate(phys_needed):
                # Try evicting LRU prefix blocks before hard preemption
                freed = self.prefix_cache.evict_to_free(new_logical)
                if freed == 0 and active_decode:
                    victim = min(active_decode, key=lambda s: s.n_generated)
                    active_decode.remove(victim)
                    victim.swap_out(self.kv_cache, self.allocator)
                    self.swapped.append(victim)
                    if victim in self.running:
                        self.running.remove(victim)
                    self.metrics.n_preemptions += 1
                elif freed == 0:
                    continue
            seq.block_table.append_token(self.allocator, self.kv_cache)
            active_decode.append(seq)

        # ── Allocate prefill slots + prefix cache lookup ──────────────────────
        active_prefill     : List[PrefixSequenceGroup] = []
        prefill_chunk_info : List[Tuple[int, int, bool]] = []

        for seq in sched_out.prefill_seqs:
            if seq.prefill_time is None:
                seq.prefill_time = time.time()

            # Prefix cache lookup — only on the first chunk
            if seq.prefill_offset == 0 and seq.n_cached_tokens == 0:
                self._try_prefix_match(seq)

            # ── Fully cached prompt: skip forward pass entirely ───────────────
            if seq.prefill_offset >= seq.prompt_len:
                seq.status           = SeqStatus.DECODING
                seq.first_token_time = time.time()   # TTFT = now (zero compute cost)
                seq.block_table.append_token(self.allocator, self.kv_cache)
                active_decode.append(seq)
                self.metrics.n_fully_cached += 1
                continue

            cs      = seq.prefill_offset
            ce      = min(cs + self.chunk_size, seq.prompt_len)
            clen    = ce - cs
            is_last = (ce >= seq.prompt_len)

            phys_needed = math.ceil(clen / self.block_size) * self.n_layers
            if not self.allocator.can_allocate(phys_needed):
                # Try evicting LRU cache blocks first
                freed = self.prefix_cache.evict_to_free(math.ceil(clen / self.block_size))
                if not self.allocator.can_allocate(phys_needed):
                    # Still no room — defer to next iteration
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

        if any(not info[2] for info in prefill_chunk_info):
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

                    # Insert completed prefix blocks into cache
                    self._insert_into_cache(seq)

                    done = (
                        next_tok == self.eos_token_id
                        or seq.n_generated >= seq.max_new_tokens
                    )
                    if done:
                        finished_this_step.append(self._finish(seq))
                    elif seq not in self.running:
                        self.running.append(seq)
                else:
                    # More chunks remain — push back to waiting heap so it is
                    # scheduled again next iteration.  Slightly elevated priority
                    # (- 0.5) so it resumes before brand-new requests.
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
                elif seq not in self.running:
                    self.running.append(seq)

        # ── Logging ───────────────────────────────────────────────────────────
        iter_t = time.perf_counter() - t0
        self.metrics.iter_times_s.append(iter_t)
        self.metrics.batch_sizes.append(len(active_prefill) + len(active_decode))

        if self.metrics.iteration % self.log_every == 0:
            pc = self.prefix_cache
            print(
                f"[Prefix] iter={self.metrics.iteration:4d}  "
                f"batch={len(active_prefill)+len(active_decode)}  "
                f"run={len(self.running)}  wait={len(self._waiting_heap)}  "
                f"swap={len(self.swapped)}  done={self.metrics.total_requests_done}  "
                f"cache={pc.n_cached}blk hit={pc.hit_rate:.0%}  "
                f"tok/s={self.metrics.throughput_tok_s:.1f}  {iter_t*1000:.1f}ms"
            )

        return finished_this_step

    # ── Run-to-completion ─────────────────────────────────────────────────────

    def run_until_done(self) -> List[PrefixSequenceGroup]:
        """Run until all queued requests finish. Returns finished sequences."""
        while self.has_work():
            self.step()
        return self.finished

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _finish(self, seq: PrefixSequenceGroup) -> Tuple[int, str, List[int]]:
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

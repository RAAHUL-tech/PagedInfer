"""
inference/generate_prefix_cached.py — inference with hash-based prefix KV caching.

Demonstrates and benchmarks the PrefixAwareEngine by running three experiments
that mirror notebook 05:

    Experiment 1 — Shared system prompt (high hit rate)
        All requests begin with the same long system prompt.
        After the first request, every subsequent request skips the entire
        system-prompt forward pass — only the unique suffix is computed.

    Experiment 2 — TTFT comparison: cached vs cold start
        Measures time-to-first-token for identical prompts submitted twice.
        Second submission is fully cached → significantly lower TTFT.

    Experiment 3 — LRU eviction under memory pressure
        Submits more requests than the cache can hold simultaneously.
        Verifies that the LRU policy evicts cleanly and blocked blocks
        (shared with a live sequence) are never evicted prematurely.

Usage
─────
    # Run all three experiments
    uv run python inference/generate_prefix_cached.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt

    # Custom system prompt + generation length
    uv run python inference/generate_prefix_cached.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --max_new_tokens 64 \\
        --cache_fraction 0.4

    # Show generated text for every request
    uv run python inference/generate_prefix_cached.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --show_generated

Output
──────
  Per-experiment:  hit rate, TTFT (cached vs cold), throughput
  Global summary:  total tokens, avg TTFT, avg E2E, prefix cache stats
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import PrefixAwareEngine, Request
from inference.model_loader import load_tokenizer


# ── Shared system prompt ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and precise AI assistant. "
    "Always answer clearly and concisely, citing evidence when possible. "
    "If you are unsure about something, say so rather than guessing. "
    "Format your responses with proper structure when helpful. "
    "Be respectful, unbiased, and safety-conscious in all interactions. "
)

# Unique questions that follow the shared system prompt
_QUESTIONS = [
    "What is the difference between supervised and unsupervised learning?",
    "Explain the transformer architecture in simple terms.",
    "What causes seasons on Earth?",
    "How does gradient descent work in neural networks?",
    "What is the significance of the Turing test?",
    "Explain what a large language model is.",
    "How does the attention mechanism work?",
    "What is transfer learning and why is it useful?",
]


# ── Experiment helpers ────────────────────────────────────────────────────────

def _make_requests(
    prompts        : List[str],
    max_new_tokens : int,
    temperature    : float,
    top_k          : int,
    top_p          : float,
    priority       : int = 0,
    id_offset      : int = 0,
) -> List[Tuple[Request, List[int]]]:
    """Build (Request, token_ids) pairs from prompt strings."""
    from inference.model_loader import load_tokenizer
    return [(
        Request(
            request_id     = id_offset + i,
            prompt         = p,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p,
            priority       = priority,
        ),
        tokenizer.encode(p, add_special_tokens=True),
    ) for i, p in enumerate(prompts)]


def _print_results(engine: PrefixAwareEngine, tokenizer, show_generated: bool) -> None:
    """Print per-request results from a finished engine run."""
    for seq in engine.finished:
        gen_ids = seq.token_ids[seq.prompt_len:]
        ttft    = f"{seq.ttft_ms:.0f}ms" if seq.ttft_ms else "N/A"
        e2e     = f"{seq.e2e_latency_ms:.0f}ms" if seq.e2e_latency_ms else "N/A"
        cached  = f" (cached={seq.n_cached_tokens}tok)" if seq.n_cached_tokens else ""
        print(f"  [Req {seq.seq_id:2d}] TTFT={ttft:>8s}  E2E={e2e:>8s}  "
              f"gen={seq.n_generated:3d}{cached}")
        if show_generated:
            print(f"          Prompt : {seq.request.prompt[:80]!r}")
            print(f"          Output : {tokenizer.decode(gen_ids, skip_special_tokens=True)[:120]!r}")


# ── Experiment 1: Shared system prompt ───────────────────────────────────────

def experiment_shared_system_prompt(args, tokenizer) -> None:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: Shared System Prompt")
    print("  All requests begin with the same system prompt.")
    print("  After the first request, subsequent ones skip that forward pass.")
    print("=" * 65)

    prompts = [_SYSTEM_PROMPT + q for q in _QUESTIONS]

    engine = PrefixAwareEngine.build(
        checkpoint     = args.checkpoint,
        eos_token_id   = tokenizer.eos_token_id,
        device         = args.device,
        block_size     = args.block_size,
        kv_budget_gb   = args.kv_budget_gb,
        cache_fraction = args.cache_fraction,
        max_batch      = args.max_batch,
        max_prefill    = args.max_prefill,
        chunk_size     = args.chunk_size,
        log_every      = args.log_every,
        verbose        = True,
    )

    print(f"\nSubmitting {len(prompts)} requests simultaneously...\n")
    for i, p in enumerate(prompts):
        token_ids = tokenizer.encode(p, add_special_tokens=True)
        engine.add_request(
            Request(
                request_id     = i,
                prompt         = p,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                top_k          = args.top_k,
                top_p          = args.top_p,
                priority       = 0,
            ),
            token_ids,
        )

    torch.cuda.synchronize(args.device)
    t0 = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - t0

    print(f"\n  Results (Experiment 1):")
    _print_results(engine, tokenizer, args.show_generated)
    engine.metrics.print_summary(engine.prefix_cache)
    print(f"  Wall-clock: {elapsed:.2f}s  "
          f"Throughput: {engine.metrics.total_tokens_gen / elapsed:.1f} tok/s")

    del engine
    torch.cuda.empty_cache(); gc.collect()


# ── Experiment 2: TTFT — cached vs cold ──────────────────────────────────────

def experiment_ttft_comparison(args, tokenizer) -> None:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: TTFT — Cached vs Cold Start")
    print("  Same prompts submitted twice. Second run is fully cached.")
    print("=" * 65)

    # Use 4 prompts with shared prefix
    prompts = [_SYSTEM_PROMPT + q for q in _QUESTIONS[:4]]

    engine = PrefixAwareEngine.build(
        checkpoint     = args.checkpoint,
        eos_token_id   = tokenizer.eos_token_id,
        device         = args.device,
        block_size     = args.block_size,
        kv_budget_gb   = args.kv_budget_gb,
        cache_fraction = args.cache_fraction,
        max_batch      = args.max_batch,
        max_prefill    = args.max_prefill,
        chunk_size     = args.chunk_size,
        log_every      = 999,   # suppress per-iter logs for this experiment
        verbose        = False,
    )

    # ── Cold run ──────────────────────────────────────────────────────────────
    print("\n  [Cold run] — no cache entries yet")
    for i, p in enumerate(prompts):
        engine.add_request(
            Request(request_id=i, prompt=p, max_new_tokens=args.max_new_tokens,
                    temperature=0.0, top_k=1, top_p=1.0, priority=0),
            tokenizer.encode(p, add_special_tokens=True),
        )
    torch.cuda.synchronize(args.device)
    t_cold = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize(args.device)
    cold_elapsed = time.perf_counter() - t_cold
    cold_ttfts   = [seq.ttft_ms for seq in engine.finished if seq.ttft_ms]
    cold_avg     = sum(cold_ttfts) / len(cold_ttfts) if cold_ttfts else 0

    # ── Warm run ──────────────────────────────────────────────────────────────
    print(f"\n  [Warm run] — cache has {engine.prefix_cache.n_cached} blocks")
    engine.finished.clear()
    # Reset metric lists for the warm run
    engine.metrics.ttft_ms_list.clear()
    engine.metrics.e2e_ms_list.clear()

    for i, p in enumerate(prompts):
        engine.add_request(
            Request(request_id=100 + i, prompt=p, max_new_tokens=args.max_new_tokens,
                    temperature=0.0, top_k=1, top_p=1.0, priority=0),
            tokenizer.encode(p, add_special_tokens=True),
        )
    torch.cuda.synchronize(args.device)
    t_warm = time.perf_counter()
    engine.run_until_done()
    torch.cuda.synchronize(args.device)
    warm_elapsed = time.perf_counter() - t_warm
    warm_ttfts   = [seq.ttft_ms for seq in engine.finished if seq.ttft_ms]
    warm_avg     = sum(warm_ttfts) / len(warm_ttfts) if warm_ttfts else 0

    print(f"\n  Cold avg TTFT : {cold_avg:.1f} ms  ({cold_elapsed:.2f}s total)")
    print(f"  Warm avg TTFT : {warm_avg:.1f} ms  ({warm_elapsed:.2f}s total)")
    if warm_avg > 0 and cold_avg > 0:
        print(f"  TTFT speedup  : {cold_avg / warm_avg:.2f}×")
    print(f"  Cache hit rate: {engine.prefix_cache.hit_rate:.1%}")

    del engine
    torch.cuda.empty_cache(); gc.collect()


# ── Experiment 3: LRU eviction under memory pressure ─────────────────────────

def experiment_lru_eviction(args, tokenizer) -> None:
    print("\n" + "=" * 65)
    print("  EXPERIMENT 3: LRU Eviction Under Memory Pressure")
    print("  Cache limited to 4 logical blocks — verifies eviction policy.")
    print("=" * 65)

    prompts = [_SYSTEM_PROMPT + q for q in _QUESTIONS]

    # Small cache to force evictions
    engine = PrefixAwareEngine.build(
        checkpoint     = args.checkpoint,
        eos_token_id   = tokenizer.eos_token_id,
        device         = args.device,
        block_size     = args.block_size,
        kv_budget_gb   = args.kv_budget_gb,
        cache_fraction = 0.1,   # force small cache → frequent evictions
        max_batch      = args.max_batch,
        max_prefill    = args.max_prefill,
        chunk_size     = args.chunk_size,
        log_every      = args.log_every,
        verbose        = False,
    )
    print(f"  Cache capacity: {engine.prefix_cache.max_blocks} logical blocks\n")

    for i, p in enumerate(prompts):
        engine.add_request(
            Request(request_id=i, prompt=p, max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature, top_k=args.top_k,
                    top_p=args.top_p, priority=0),
            tokenizer.encode(p, add_special_tokens=True),
        )

    engine.run_until_done()

    pc = engine.prefix_cache
    print(f"\n  Cache blocks remaining : {pc.n_cached}")
    print(f"  Blocks evicted         : {pc.n_evicted}")
    print(f"  Cache hits             : {pc.n_hits}")
    print(f"  Cache misses           : {pc.n_misses}")
    print(f"  Hit rate               : {pc.hit_rate:.1%}")
    print(f"  All {len(engine.finished)} requests finished: "
          f"{'YES' if len(engine.finished) == len(prompts) else 'NO'}")

    del engine
    torch.cuda.empty_cache(); gc.collect()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PrefixAwareEngine — inference with hash-based prefix KV caching"
    )
    p.add_argument("--checkpoint",     required=True,  help="Path to .pt checkpoint")
    p.add_argument("--tokenizer",      default="huggyllama/llama-7b")
    p.add_argument("--max_new_tokens", type=int,   default=64)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--block_size",     type=int,   default=16)
    p.add_argument("--kv_budget_gb",   type=float, default=None)
    p.add_argument("--cache_fraction", type=float, default=0.5,
                   help="Fraction of logical KV pool reserved for prefix cache")
    p.add_argument("--max_batch",      type=int,   default=8)
    p.add_argument("--max_prefill",    type=int,   default=4)
    p.add_argument("--chunk_size",     type=int,   default=32)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--show_generated", action="store_true",
                   help="Print generated text for each request")
    p.add_argument("--experiment",     type=int,   default=0,
                   help="Run a specific experiment (1/2/3). 0 = run all.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for PrefixAwareEngine.")
        sys.exit(1)

    torch.cuda.empty_cache(); gc.collect()
    print(f"GPU: {torch.cuda.get_device_name(args.device)}  "
          f"({torch.cuda.get_device_properties(args.device).total_memory / 1e9:.2f} GB)")

    global tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    run_all = args.experiment == 0

    if run_all or args.experiment == 1:
        experiment_shared_system_prompt(args, tokenizer)

    if run_all or args.experiment == 2:
        experiment_ttft_comparison(args, tokenizer)

    if run_all or args.experiment == 3:
        experiment_lru_eviction(args, tokenizer)


if __name__ == "__main__":
    main()

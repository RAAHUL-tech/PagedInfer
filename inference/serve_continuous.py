"""
inference/serve_continuous.py — CLI test harness for the continuous batching engine.

Submits multiple prompts concurrently and measures throughput vs. static
batching (sequential processing, one request at a time).

Usage
────────
    # Basic: run a set of built-in test prompts
    uv run python inference/serve_continuous.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt

    # Custom prompts (--prompts is a newline-separated file or inline text)
    uv run python inference/serve_continuous.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --max_new_tokens 128 \\
        --max_batch 8

    # Memory-pressure experiment: force preemptions by limiting pool size
    uv run python inference/serve_continuous.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --kv_budget_gb 1.0

Output
────────
  Per-request:  prompt, generated text, TTFT, E2E latency
  Summary:      throughput, avg TTFT, avg E2E, preemptions, swap-ins
  Comparison:   continuous batching tok/s vs. static sequential tok/s
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine import (
    ContinuousBatchingEngine,
    Request,
)
from inference.model_loader import load_tokenizer


# ── Default test prompts ──────────────────────────────────────────────────────

_DEFAULT_PROMPTS: List[Tuple[str, int]] = [
    ("Once upon a time there was a little dragon who",       200),
    ("The scientist looked at the data and realised",        200),
    ("In the year 2075, humanity had finally",               100),
    ("The recipe for the perfect chocolate cake requires",    500),
    ("Write a haiku about the autumn moon:",                  300),
    ("The key insight of the transformer architecture is",    90),
    ("To debug a segmentation fault in C, you should",        150),
    ("The difference between supervised and unsupervised",    450),
]


# ── Static batching baseline ──────────────────────────────────────────────────

def _static_baseline(
    checkpoint   : str,
    prompts      : List[Tuple[str, int]],
    tokenizer,
    device       : str,
    block_size   : int,
    kv_budget_gb : Optional[float],
) -> float:
    """
    Sequential (static) baseline: process each request one at a time.

    Returns total time in seconds for all requests.
    """
    from inference.model_loader import _remap_state_dict, _MODEL_FIELDS
    from kv_cache import BlockAllocator, LayeredBlockTable
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
    from models import ModelConfig, Transformer
    import torch.nn.functional as F

    print("\n[Static] Running static batching baseline...")

    raw        = torch.load(checkpoint, map_location=device, weights_only=True)
    raw_config = raw.get("config", {}) if isinstance(raw, dict) else {}
    state_dict = raw.get("model_state", raw) if isinstance(raw, dict) else raw
    kwargs     = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
    kwargs["attention_type"] = "paged"
    cfg   = ModelConfig(**kwargs)
    model = Transformer(cfg)
    model.load_state_dict(_remap_state_dict(state_dict))
    model = model.half().to(device).eval()

    torch.cuda.synchronize(device)
    if kv_budget_gb is not None:
        kv_bytes = int(kv_budget_gb * 1024 ** 3)
    else:
        free, _ = torch.cuda.mem_get_info(device)
        kv_bytes = max(0, free - int(1.5 * 1024 ** 3))

    kv_cache = GPUPagedKVCache.from_memory_budget(
        free_gpu_bytes=kv_bytes, n_layers=cfg.n_layers, block_size=block_size,
        n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim, device=device, verbose=False,
    )

    total_toks = 0
    t0 = time.perf_counter()

    for prompt, max_new in prompts:
        allocator   = BlockAllocator(kv_cache.n_phys)
        block_table = LayeredBlockTable(0, cfg.n_layers, block_size)
        tokens      = tokenizer.encode(prompt, add_special_tokens=True)

        block_table.append_tokens(len(tokens), allocator, kv_cache)
        idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            logits  = model(idx, start_pos=0, block_table=block_table, kv_cache=kv_cache)
        next_id = int(logits[0, -1].argmax())
        tokens.append(next_id)
        total_toks += 1

        for i in range(max_new - 1):
            block_table.append_token(allocator, kv_cache)
            tok_t = torch.tensor([[next_id]], dtype=torch.long, device=device)
            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                logits = model(tok_t, start_pos=len(tokens) - 1,
                               block_table=block_table, kv_cache=kv_cache)
            next_id = int(logits[0, -1].argmax())
            tokens.append(next_id)
            total_toks += 1
            if next_id == tokenizer.eos_token_id:
                break

        block_table.free(allocator, kv_cache)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    print(f"[Static] {len(prompts)} requests, {total_toks} tokens in {elapsed:.2f}s  "
          f"→  {total_toks/elapsed:.1f} tok/s")
    return elapsed


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ContinuousBatchingEngine — multi-request inference server"
    )
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--tokenizer",      default="huggyllama/llama-7b")
    p.add_argument("--max_new_tokens", type=int,   default=100)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--block_size",     type=int,   default=16)
    p.add_argument("--kv_budget_gb",   type=float, default=None)
    p.add_argument("--max_batch",      type=int,   default=8)
    p.add_argument("--max_prefill",    type=int,   default=4)
    p.add_argument("--chunk_size",     type=int,   default=32)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--log_every",      type=int,   default=5)
    p.add_argument("--compare_static", action="store_true",
                   help="Also run static batching baseline for comparison")
    p.add_argument("--show_generated", action="store_true",
                   help="Print generated text for each request")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for the continuous batching engine.")
        sys.exit(1)

    torch.cuda.empty_cache(); gc.collect()
    device = args.device

    print(f"GPU: {torch.cuda.get_device_name(device)}  "
          f"({torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB)")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id    = tokenizer.eos_token_id

    # ── Build engine ──────────────────────────────────────────────────────────
    engine = ContinuousBatchingEngine.build(
        checkpoint   = args.checkpoint,
        eos_token_id = eos_id,
        device       = device,
        block_size   = args.block_size,
        kv_budget_gb = args.kv_budget_gb,
        max_batch    = args.max_batch,
        max_prefill  = args.max_prefill,
        chunk_size   = args.chunk_size,
        log_every    = args.log_every,
        verbose      = True,
    )

    # ── Add test requests ─────────────────────────────────────────────────────
    prompts = _DEFAULT_PROMPTS
    print(f"\nSubmitting {len(prompts)} requests simultaneously...\n")

    for i, (prompt, max_new) in enumerate(prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
        engine.add_request(
            Request(
                request_id     = i,
                prompt         = prompt,
                max_new_tokens = max_new,
                temperature    = args.temperature,
                top_k          = args.top_k,
                top_p          = args.top_p,
                priority       = 0,
            ),
            token_ids = token_ids,
        )

    # ── Run to completion ─────────────────────────────────────────────────────
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    engine.run_until_done()

    torch.cuda.synchronize(device)
    cb_elapsed = time.perf_counter() - t_start

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  CONTINUOUS BATCHING RESULTS")
    print(f"{'='*65}")

    for seq in engine.finished:
        prompt  = seq.request.prompt
        gen_ids = seq.token_ids[seq.prompt_len:]
        gen_txt = tokenizer.decode(gen_ids, skip_special_tokens=True)

        print(f"\n[Req {seq.seq_id}]  TTFT={seq.ttft_ms:.0f}ms  "
              f"E2E={seq.e2e_latency_ms:.0f}ms  "
              f"gen={seq.n_generated} tokens")
        print(f"  Prompt : {prompt!r}")
        if args.show_generated:
            print(f"  Output : {gen_txt!r}")

    # ── Engine metrics ────────────────────────────────────────────────────────
    engine.metrics.print_summary()

    print(f"\n  Continuous batching : {engine.metrics.total_tokens_gen / cb_elapsed:.1f} tok/s")

    # ── Static baseline comparison ────────────────────────────────────────────
    if args.compare_static:
        torch.cuda.empty_cache(); gc.collect()
        static_elapsed = _static_baseline(
            args.checkpoint, prompts, tokenizer, device,
            args.block_size, args.kv_budget_gb,
        )
        static_toks = sum(n for _, n in prompts)
        print(f"  Static batching     : {static_toks / static_elapsed:.1f} tok/s")
        speedup = (engine.metrics.total_tokens_gen / cb_elapsed) / (static_toks / static_elapsed)
        print(f"  Speedup (CB / static): {speedup:.2f}×")


if __name__ == "__main__":
    main()

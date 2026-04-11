"""
inference/generate_flat_kv.py — inference with a flat (non-paged) KV cache.

Two-phase generation (mirrors notebook cell 36):

  Prefill  — forward the full prompt once; every attention layer stores its
             K and V tensors in a flat (concatenation) cache.
             O(T_prompt²) compute, done once.

  Decode   — feed a single new token per step; each layer reads from the
             cache and appends only the new K/V pair.
             O(T_total) per step instead of O(T_total²) — this is the speedup.

The flat cache (simple torch.cat along the time axis) is the baseline
implementation.  It will be replaced by the paged block allocator once
kv_cache/ is built; the generate interface stays the same.

Sampling strategy (mirrors notebook cell 36):
  1. Temperature scaling
  2. Top-k truncation
  3. Categorical sample

Usage:
    uv run python inference/generate_flat_kv.py \
    --checkpoint model_checkpoint/llama_ckpt.pt \
    --prompt "Once upon a time" \
    --max_new_tokens 200 \
    --temperature 0.8 \
    --top_k 50

    # compare against no-cache for a speed benchmark
    python inference/generate_flat_kv.py \\
        --checkpoint llama_ckpt.pt --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ModelConfig, Transformer
from inference.model_loader import load_model, load_tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_topk(
    logits      : torch.Tensor,   # (1, vocab_size) float32
    temperature : float,
    top_k       : Optional[int],
) -> int:
    """Temperature scaling → top-k truncation → categorical sample."""

    if temperature < 1e-5:
        return int(logits.argmax(dim=-1).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        logits[logits < threshold[:, [-1]]] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_kv_cache(
    model          : Transformer,
    tokenizer,
    prompt         : str,
    max_new_tokens : int   = 200,
    temperature    : float = 0.8,
    top_k          : Optional[int] = 50,
    device         : str   = "cpu",
) -> tuple[str, float, float]:
    """
    Two-phase KV-cache generation.

    Phase 1 — Prefill:
        Forward the full prompt with use_cache=True, start_pos=0.
        Each attention layer fills cache_k and cache_v with the prompt's K/V.
        Sample the first generated token from the last prompt position.

    Phase 2 — Decode:
        For each subsequent token, pass only that single token with
        use_cache=True and start_pos = len(prompt) + i.
        The attention layer appends the new K/V to the cache and computes
        attention over the full history without recomputing prompt K/V.

    Args:
        model          : Transformer in eval mode (kv cache cleared on entry)
        tokenizer      : HuggingFace tokenizer
        prompt         : Input text
        max_new_tokens : Maximum tokens to generate after the prompt
        temperature    : Softmax temperature (0 = greedy)
        top_k          : Keep only the top-k logits before sampling
        device         : "cpu" or "cuda"

    Returns:
        (generated_text, prefill_time_s, decode_tok_per_sec)
    """
    model.eval()
    model.clear_kv_cache()

    tokens  = tokenizer.encode(prompt, add_special_tokens=True)
    T_start = len(tokens)
    idx     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # ── Phase 1: Prefill ──────────────────────────────────────────────────────
    # Process the full prompt in one forward pass; cache all K/V pairs.
    t_prefill_start = time.perf_counter()

    logits   = model(idx, use_cache=True, start_pos=0)   # (1, T_prompt, vocab)
    logits   = logits[:, -1, :].float()                  # last position
    next_id  = _sample_topk(logits, temperature, top_k)
    generated = [next_id]

    t_prefill_end = time.perf_counter()
    prefill_time  = t_prefill_end - t_prefill_start

    if next_id == tokenizer.eos_token_id:
        text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
        model.clear_kv_cache()
        return text, prefill_time, float("inf")

    # ── Phase 2: Decode ───────────────────────────────────────────────────────
    # Feed one token at a time; attention reads from the growing cache.
    t_decode_start = time.perf_counter()

    for i in range(max_new_tokens - 1):
        # Pass ONLY the newly generated token
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits   = model(
            next_tok,
            use_cache=True,
            start_pos=T_start + i,   # absolute position for RoPE
        )                                                 # (1, 1, vocab)
        logits   = logits[:, -1, :].float()
        next_id  = _sample_topk(logits, temperature, top_k)
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    t_decode_end   = time.perf_counter()
    decode_elapsed = t_decode_end - t_decode_start
    n_decode_toks  = len(generated) - 1    # first token was sampled in prefill
    decode_tok_s   = n_decode_toks / decode_elapsed if decode_elapsed > 0 else float("inf")

    model.clear_kv_cache()

    text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
    return text, prefill_time, decode_tok_s


# ── Benchmark helper ──────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark(
    model     : Transformer,
    tokenizer,
    prompt    : str,
    n_tokens  : int  = 100,
    device    : str  = "cpu",
) -> None:
    """
    Side-by-side speed comparison: no-cache vs KV-cache.
    Imports generate from generate.py so the comparison is apples-to-apples.
    """
    from inference.generate import generate as generate_no_cache

    print(f"\nBenchmark: {n_tokens} tokens  |  prompt={prompt!r}")
    print("-" * 60)

    # No-cache baseline
    t0 = time.perf_counter()
    _, tok_s_no_cache = generate_no_cache(
        model, tokenizer, prompt,
        max_new_tokens=n_tokens,
        temperature=0.8, top_k=50, top_p=0.9,
        device=device,
    )
    t1 = time.perf_counter()
    print(f"No KV cache : {t1-t0:.2f}s  →  {tok_s_no_cache:.1f} tok/s")

    # KV-cache
    t2 = time.perf_counter()
    _, prefill_t, decode_tok_s = generate_kv_cache(
        model, tokenizer, prompt,
        max_new_tokens=n_tokens,
        temperature=0.8, top_k=50,
        device=device,
    )
    t3 = time.perf_counter()
    print(f"KV cache    : {t3-t2:.2f}s  →  decode {decode_tok_s:.1f} tok/s  "
          f"(prefill {prefill_t*1000:.1f} ms)")

    if tok_s_no_cache > 0:
        print(f"Speedup     : {decode_tok_s / tok_s_no_cache:.1f}×")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaMA inference with flat KV cache (prefill + decode)"
    )
    p.add_argument("--checkpoint",      required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--prompt",          default="Once upon a time",
                   help="Prompt text")
    p.add_argument("--max_new_tokens",  type=int,   default=200)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top_k",           type=int,   default=50,
                   help="Top-k truncation (0 to disable)")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer",       default="huggyllama/llama-7b")
    p.add_argument("--dtype",           default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--benchmark",       action="store_true",
                   help="Run side-by-side speed comparison with no-cache generate")
    p.add_argument("--benchmark_tokens", type=int, default=100,
                   help="Tokens to generate in benchmark mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {
        "float32":  torch.float32,
        "float16":  torch.float16,
        "bfloat16": torch.bfloat16,
    }

    model, cfg = load_model(args.checkpoint, device=args.device, dtype=dtype_map[args.dtype])
    tokenizer  = load_tokenizer(args.tokenizer)

    if args.benchmark:
        benchmark(model, tokenizer, args.prompt, args.benchmark_tokens, args.device)
        return

    print(f"\nPrompt : {args.prompt!r}")
    print(f"Config : temperature={args.temperature}, top_k={args.top_k}")
    print(f"Device : {args.device}  dtype={args.dtype}")
    print("-" * 60)

    top_k = args.top_k if args.top_k > 0 else None
    text, prefill_t, decode_tok_s = generate_kv_cache(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        device=args.device,
    )

    print(text)
    print("-" * 60)
    print(f"Prefill : {prefill_t * 1000:.1f} ms")
    print(f"Decode  : {decode_tok_s:.1f} tok/s  (flat KV cache — O(T) per step)")


if __name__ == "__main__":
    main()

"""
Inference — no KV cache.

Every generation step runs a full forward pass over the entire sequence grown
so far.  Simple and correct; progressively slower as the sequence grows because
attention is O(T²) and there is no caching.

Sampling strategy (mirrors notebook cell 31):
  1. Temperature scaling   : logits /= temperature
  2. Top-k truncation      : keep only the k highest-logit tokens
  3. Top-p (nucleus)       : further restrict to the smallest set whose
                             cumulative probability ≥ p
  4. Categorical sample    : draw one token from the remaining distribution

Usage:
    python inference/generate.py \\
        --checkpoint llama_ckpt.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 200 \\
        --temperature 0.8 \\
        --top_k 50 \\
        --top_p 0.9

    # greedy decode (deterministic)
    python inference/generate.py --checkpoint llama_ckpt.pt \\
        --prompt "Once upon a time" --temperature 0.0
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
from inference._load import load_model, load_tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample(
    logits      : torch.Tensor,   # (1, vocab_size) float32
    temperature : float,
    top_k       : Optional[int],
    top_p       : float,
) -> int:
    """Apply temperature → top-k → top-p, then sample one token id."""

    # Greedy if temperature is effectively zero
    if temperature < 1e-5:
        return int(logits.argmax(dim=-1).item())

    logits = logits / temperature

    # Top-k: zero out everything below the k-th largest logit
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        logits[logits < threshold[:, [-1]]] = float("-inf")

    # Top-p (nucleus): keep the smallest set of tokens whose cumulative
    # probability mass is at least p, discard the rest
    probs = F.softmax(logits, dim=-1)                           # (1, vocab)
    sorted_p, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_p, dim=-1)

    # Tokens beyond the nucleus: cumulative - token_prob > p
    remove = cumulative - sorted_p > top_p
    sorted_p[remove] = 0.0
    sorted_p /= sorted_p.sum(dim=-1, keepdim=True)             # renormalize

    # Scatter filtered probs back to original vocab order
    probs = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_p)

    return int(torch.multinomial(probs, num_samples=1).item())


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model          : Transformer,
    tokenizer,
    prompt         : str,
    max_new_tokens : int   = 200,
    temperature    : float = 0.8,
    top_k          : Optional[int] = 50,
    top_p          : float = 0.9,
    device         : str   = "cpu",
) -> tuple[str, float]:
    """
    Autoregressive generation without KV cache.

    Each step:
      1. Forward the entire sequence (prompt + generated so far)
      2. Take logits at the last position
      3. Sample the next token
      4. Append and repeat

    Args:
        model          : Transformer in eval mode
        tokenizer      : HuggingFace tokenizer
        prompt         : Input text
        max_new_tokens : Maximum tokens to generate
        temperature    : Softmax temperature (0 = greedy)
        top_k          : Keep only the top-k logits before sampling
        top_p          : Nucleus sampling probability threshold
        device         : "cpu" or "cuda"

    Returns:
        (generated_text, tokens_per_second)
    """
    model.eval()
    max_seq_len = model.cfg.max_seq_len

    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    idx    = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    t_start = time.perf_counter()

    for _ in range(max_new_tokens):
        # Crop to context window — older tokens are discarded
        idx_cond = idx[:, -max_seq_len:]

        # Full forward pass over the entire current sequence — no cache
        logits = model(idx_cond)                  # (1, T, vocab_size)
        logits = logits[:, -1, :].float()         # (1, vocab_size), last position

        next_id = _sample(logits, temperature, top_k, top_p)
        idx     = torch.cat(
            [idx, torch.tensor([[next_id]], device=device)], dim=1
        )

        if next_id == tokenizer.eos_token_id:
            break

    elapsed      = time.perf_counter() - t_start
    n_generated  = idx.shape[1] - len(tokens)
    tok_per_sec  = n_generated / elapsed if elapsed > 0 else float("inf")

    text = tokenizer.decode(idx[0].tolist(), skip_special_tokens=True)
    return text, tok_per_sec


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaMA inference — no KV cache (O(T²) per step)"
    )
    p.add_argument("--checkpoint",      required=True,       help="Path to .pt checkpoint")
    p.add_argument("--prompt",          default="Once upon a time",
                   help="Prompt text")
    p.add_argument("--max_new_tokens",  type=int,   default=200)
    p.add_argument("--temperature",     type=float, default=0.8,
                   help="Sampling temperature (0 = greedy)")
    p.add_argument("--top_k",           type=int,   default=50,
                   help="Top-k truncation (0 to disable)")
    p.add_argument("--top_p",           type=float, default=0.9,
                   help="Nucleus sampling threshold")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer",       default="hf-internal-testing/llama-tokenizer",
                   help="HuggingFace tokenizer id or local path")
    p.add_argument("--dtype",           default="float32",
                   choices=["float32", "float16", "bfloat16"],
                   help="Model weight dtype")
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

    print(f"\nPrompt : {args.prompt!r}")
    print(f"Config : temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print(f"Device : {args.device}  dtype={args.dtype}")
    print("-" * 60)

    top_k = args.top_k if args.top_k > 0 else None
    text, tok_s = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
        top_p=args.top_p,
        device=args.device,
    )

    print(text)
    print("-" * 60)
    print(f"Speed  : {tok_s:.1f} tok/s  (no KV cache — O(T²) per step)")


if __name__ == "__main__":
    main()

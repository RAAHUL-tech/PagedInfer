"""
Inference — paged KV cache (vLLM-style).

Two-phase generation identical to generate_kv_cache.py in structure, but
KV state is stored in a block-paged pool instead of flat tensors:

  Prefill  — process the full prompt with block_table + kv_cache;
             each attention layer writes K/V into physical blocks.
  Decode   — one token per step; each layer appends a new slot to the page.

Memory comparison printed at the end:
  Flat (naive) KV cache  — pre-allocates max_seq_len for every sequence,
                            regardless of actual generation length.
  Paged KV cache         — allocates only the blocks actually used, rounded
                            up to block_size granularity (slight fragmentation
                            in the last block only).

Usage:
    python inference/generate_paged_kv_cache.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --prompt "Once upon a time" \\
        --max_new_tokens 200

    # explicit paged config
    python inference/generate_paged_kv_cache.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --prompt "Once upon a time" \\
        --block_size 16 --n_logical_blocks 256

    # memory-only report (no generation)
    python inference/generate_paged_kv_cache.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --memory_report
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import BlockAllocator, LayeredBlockTable, PagedKVCache
from models import ModelConfig, Transformer
from inference._load import load_tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample_topk(logits: torch.Tensor, temperature: float, top_k: Optional[int]) -> int:
    if temperature < 1e-5:
        return int(logits.argmax(dim=-1).item())
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        threshold, _ = torch.topk(logits, k)
        logits[logits < threshold[:, [-1]]] = float("-inf")
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_paged(
    model          : Transformer,
    tokenizer,
    prompt         : str,
    max_new_tokens : int   = 200,
    temperature    : float = 0.8,
    top_k          : Optional[int] = 50,
    block_size     : int   = 16,
    n_logical_blocks: int  = 256,
    device         : str   = "cpu",
) -> tuple[str, float, float, dict]:
    """
    Two-phase generation with paged KV cache.

    Phase 1 — Prefill:
        Allocate blocks for all prompt tokens.
        Forward full prompt; each layer writes K/V into the paged pool.

    Phase 2 — Decode:
        For each new token, append one slot to the block table (allocating
        a new block if the last one is full), then forward the single token.

    Args:
        model            : Transformer with attention_type="paged"
        tokenizer        : HuggingFace tokenizer
        prompt           : Input text
        max_new_tokens   : Max tokens to generate
        temperature      : Sampling temperature
        top_k            : Top-k truncation
        block_size       : Tokens per KV block (must match training assumption)
        n_logical_blocks : Max logical blocks in the pool (pool size = this × n_layers)
        device           : "cpu" or "cuda"

    Returns:
        (text, prefill_time_s, decode_tok_per_sec, memory_report_dict)
    """
    model.eval()
    cfg      = model.cfg
    n_layers = cfg.n_layers

    # ── Allocator + cache (one per request in this script) ───────────────────
    n_total_blocks = n_logical_blocks * n_layers
    allocator      = BlockAllocator(n_total_blocks)
    kv_cache       = PagedKVCache(
        n_layers       = n_layers,
        n_total_blocks = n_total_blocks,
        block_size     = block_size,
        n_kv_heads     = cfg.n_kv_heads,
        head_dim       = cfg.head_dim,
        dtype          = torch.float16 if device != "cpu" else torch.float32,
        device         = device,
    )
    block_table = LayeredBlockTable(seq_id=0, n_layers=n_layers, block_size=block_size)

    # Encode prompt
    tokens  = tokenizer.encode(prompt, add_special_tokens=True)
    T_start = len(tokens)
    idx     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # ── Phase 1: Prefill ──────────────────────────────────────────────────────
    block_table.append_tokens(T_start, allocator, kv_cache)
    mem_after_prefill = kv_cache.memory_used_mb()

    t_prefill_start = time.perf_counter()
    logits   = model(idx, start_pos=0, block_table=block_table, kv_cache=kv_cache)
    logits   = logits[:, -1, :].float()
    next_id  = _sample_topk(logits, temperature, top_k)
    generated = [next_id]
    t_prefill_end = time.perf_counter()
    prefill_time  = t_prefill_end - t_prefill_start

    if next_id == tokenizer.eos_token_id:
        text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
        mem_report = _memory_report(cfg, T_start, block_size, kv_cache)
        block_table.free(allocator, kv_cache)
        return text, prefill_time, float("inf"), mem_report

    # ── Phase 2: Decode ───────────────────────────────────────────────────────
    t_decode_start = time.perf_counter()

    for i in range(max_new_tokens - 1):
        # Advance block table by one token (allocates a new block when needed)
        block_table.append_token(allocator, kv_cache)

        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits   = model(
            next_tok,
            start_pos   = T_start + i,
            block_table = block_table,
            kv_cache    = kv_cache,
        )
        logits  = logits[:, -1, :].float()
        next_id = _sample_topk(logits, temperature, top_k)
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    t_decode_end   = time.perf_counter()
    decode_elapsed = t_decode_end - t_decode_start
    n_decode_toks  = len(generated) - 1
    decode_tok_s   = n_decode_toks / decode_elapsed if decode_elapsed > 0 else float("inf")

    # ── Memory report ─────────────────────────────────────────────────────────
    total_tokens  = T_start + len(generated)
    mem_report    = _memory_report(cfg, total_tokens, block_size, kv_cache)

    # Cleanup
    block_table.free(allocator, kv_cache)
    assert kv_cache.memory_used_mb() == 0.0, "Memory leak after free!"

    text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
    return text, prefill_time, decode_tok_s, mem_report


# ── Memory report helpers ─────────────────────────────────────────────────────

def _memory_report(
    cfg         : ModelConfig,
    n_tokens    : int,
    block_size  : int,
    kv_cache    : PagedKVCache,
) -> dict:
    """Build the memory comparison dict for the given sequence length."""
    dtype_bytes  = 2   # fp16

    # Flat cache: allocates max_seq_len for every sequence upfront
    flat_mb = PagedKVCache.flat_cache_memory_mb(
        n_seqs      = 1,
        max_seq_len = cfg.max_seq_len,
        n_layers    = cfg.n_layers,
        n_kv_heads  = cfg.n_kv_heads,
        head_dim    = cfg.head_dim,
        dtype_bytes = dtype_bytes,
    )

    # Paged cache: actual tokens generated, rounded up to block granularity
    paged_mb = PagedKVCache.paged_cache_memory_mb(
        n_tokens    = n_tokens,
        block_size  = block_size,
        n_layers    = cfg.n_layers,
        n_kv_heads  = cfg.n_kv_heads,
        head_dim    = cfg.head_dim,
        dtype_bytes = dtype_bytes,
    )

    n_blocks     = math.ceil(n_tokens / block_size)
    fragmented   = n_blocks * block_size - n_tokens  # wasted slots in last block
    savings_pct  = (1 - paged_mb / flat_mb) * 100 if flat_mb > 0 else 0.0

    return {
        "n_tokens"         : n_tokens,
        "block_size"       : block_size,
        "n_logical_blocks" : n_blocks,
        "fragmented_slots" : fragmented,
        "flat_mb"          : flat_mb,
        "paged_mb"         : paged_mb,
        "savings_pct"      : savings_pct,
    }


def print_memory_report(report: dict, n_layers: int, n_kv_heads: int, head_dim: int) -> None:
    print("\n" + "=" * 60)
    print("  KV CACHE MEMORY COMPARISON")
    print("=" * 60)
    print(f"  Sequence length : {report['n_tokens']} tokens")
    print(f"  Block size      : {report['block_size']} tokens/block")
    print(f"  Logical blocks  : {report['n_logical_blocks']}  "
          f"(last block {report['fragmented_slots']} slot(s) wasted)")
    print()
    print(f"  Flat (naive) KV cache : {report['flat_mb']:.2f} MB")
    print(f"    — pre-allocates max_seq_len for every request")
    print()
    print(f"  Paged KV cache        : {report['paged_mb']:.2f} MB")
    print(f"    — allocates only blocks actually written")
    print()
    print(f"  Memory saved          : {report['savings_pct']:.1f}%")
    print("=" * 60)


# ── Memory-only report (no generation needed) ────────────────────────────────

def print_memory_curve(cfg: ModelConfig, block_size: int = 16) -> None:
    """Print memory usage at various generation lengths."""
    print("\n" + "=" * 60)
    print("  MEMORY vs. SEQUENCE LENGTH  (1 sequence, fp16)")
    print("=" * 60)
    print(f"  {'tokens':>8}  {'flat MB':>10}  {'paged MB':>10}  {'savings':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for n in [32, 64, 128, 256, 512]:
        flat_mb  = PagedKVCache.flat_cache_memory_mb(1, cfg.max_seq_len,
                       cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
        paged_mb = PagedKVCache.paged_cache_memory_mb(n, block_size,
                       cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
        savings  = (1 - paged_mb / flat_mb) * 100
        print(f"  {n:>8}  {flat_mb:>10.2f}  {paged_mb:>10.2f}  {savings:>7.1f}%")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaMA inference with paged KV cache + memory comparison"
    )
    p.add_argument("--checkpoint",        required=True)
    p.add_argument("--prompt",            default="Once upon a time")
    p.add_argument("--max_new_tokens",    type=int,   default=200)
    p.add_argument("--temperature",       type=float, default=0.8)
    p.add_argument("--top_k",             type=int,   default=50)
    p.add_argument("--block_size",        type=int,   default=16,
                   help="Tokens per KV block")
    p.add_argument("--n_logical_blocks",  type=int,   default=256,
                   help="Max logical blocks in pool (pool slots = this × n_layers)")
    p.add_argument("--device",            default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tokenizer",         default="huggyllama/llama-7b")
    p.add_argument("--memory_report",     action="store_true",
                   help="Print memory curve without generating text")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Always force attention_type=paged before building the model.
    # load_model() reads config from the checkpoint then builds Transformer —
    # we intercept by loading the raw checkpoint ourselves, patching the config,
    # then delegating the weight mapping to _remap_state_dict.
    from inference._load import _remap_state_dict, _MODEL_FIELDS

    raw        = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    raw_config = raw.get("config", {}) if isinstance(raw, dict) else {}
    state_dict = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    step       = raw.get("step", "?") if isinstance(raw, dict) else "?"

    model_kwargs = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
    model_kwargs["attention_type"] = "paged"   # always override
    cfg   = ModelConfig(**model_kwargs)
    model = Transformer(cfg)
    model.load_state_dict(_remap_state_dict(state_dict))
    model.eval().to(args.device)
    print(f"Loaded checkpoint: {args.checkpoint}  (step={step}, "
          f"{cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h, attention=paged)")

    if args.memory_report:
        print_memory_curve(cfg, args.block_size)
        return

    tokenizer = load_tokenizer(args.tokenizer)
    top_k     = args.top_k if args.top_k > 0 else None

    print(f"\nPrompt : {args.prompt!r}")
    print(f"Config : temperature={args.temperature}, top_k={top_k}, "
          f"block_size={args.block_size}")
    print(f"Device : {args.device}")
    print("-" * 60)

    text, prefill_t, decode_tok_s, mem_report = generate_paged(
        model, tokenizer, args.prompt,
        max_new_tokens    = args.max_new_tokens,
        temperature       = args.temperature,
        top_k             = top_k,
        block_size        = args.block_size,
        n_logical_blocks  = args.n_logical_blocks,
        device            = args.device,
    )

    print(text)
    print("-" * 60)
    print(f"Prefill : {prefill_t * 1000:.1f} ms")
    print(f"Decode  : {decode_tok_s:.1f} tok/s  (paged KV cache)")

    print_memory_report(mem_report, cfg.n_layers, cfg.n_kv_heads, cfg.head_dim)
    print_memory_curve(cfg, args.block_size)


if __name__ == "__main__":
    main()

"""
inference/generate_paged_gpu.py — inference with GPU-backed paged KV cache and CUDA kernels.

Full end-to-end inference using a contiguous GPU KV pool and custom CUDA
kernels for K/V read, write, and paged attention computation.

Pipeline
────────
  1. Load model weights → measure GPU memory consumed
  2. Compute remaining free VRAM → carve into KV blocks (GPUPagedKVCache)
  3. Compile CUDA kernels (once, cached in /tmp/) via nvcc
  4. Two-phase generation:
       Prefill  — full prompt in one forward pass
                  each attention layer writes K/V into pool via kv_write kernel
       Decode   — one token per step
                  each step appends one slot; PagedAttention calls the
                  paged_attn CUDA kernel (reads K/V from pool, online softmax)
  5. Print memory report: model VRAM, pool size, logical blocks

Backends
────────
  PagedAttention.forward() automatically dispatches based on kv_cache type:
    GPUPagedKVCache → paged_attn_kernel (no Python-level gather, fused kernel)
    PagedKVCache    → Python-level paged_attention() + SDPA

Usage
────────
    uv run python inference/generate_gpu.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --prompt "Once upon a time"

    # Override KV budget (default: use all free VRAM minus reserves)
    uv run python inference/generate_gpu.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --kv_budget_gb 4.0 \\
        --block_size 16

    # Memory-only report without generation
    uv run python inference/generate_gpu.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --memory_report
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import BlockAllocator, LayeredBlockTable
from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
from models import ModelConfig, Transformer
from inference.model_loader import load_tokenizer


# ── Sampling ──────────────────────────────────────────────────────────────────

def _sample(
    logits      : torch.Tensor,
    temperature : float = 1.0,
    top_k       : Optional[int] = 50,
    top_p       : float = 1.0,
) -> int:
    if temperature < 1e-5:
        return int(logits.argmax(dim=-1).item())
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh, _ = torch.topk(logits, k)
        logits[logits < thresh[:, [-1]]] = float("-inf")
    if top_p < 1.0:
        probs  = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask   = (cumsum - sorted_probs) >= top_p
        sorted_probs[mask] = 0.0
        probs  = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())


# ── GPU memory helpers ────────────────────────────────────────────────────────

def _gpu_memory_mb(device: str = "cuda") -> dict:
    """Return a snapshot of GPU memory stats in MB."""
    free, total = torch.cuda.mem_get_info(device)
    allocated   = torch.cuda.memory_allocated(device)
    reserved    = torch.cuda.memory_reserved(device)
    return {
        "total_mb"    : total     / 1e6,
        "free_mb"     : free      / 1e6,
        "allocated_mb": allocated / 1e6,
        "reserved_mb" : reserved  / 1e6,
    }


def _compute_kv_budget(
    device             : str,
    activation_reserve_gb : float = 1.0,
    safety_margin_gb   : float = 0.25,
) -> int:
    """
    Compute available GPU bytes for the KV pool after accounting for
    model weights (already loaded) and forward-pass activation headroom.

    Returns free_bytes available for KV cache.
    """
    torch.cuda.synchronize(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    reserve = int((activation_reserve_gb + safety_margin_gb) * 1024 ** 3)
    kv_bytes = max(0, free_bytes - reserve)
    return kv_bytes


# ── Model loader ──────────────────────────────────────────────────────────────

def _load_model_gpu(checkpoint: str, device: str) -> tuple[Transformer, dict]:
    """
    Load checkpoint, override attention_type='paged', build Transformer on GPU.

    Returns (model, raw_config_dict).
    """
    from inference.model_loader import _remap_state_dict, _MODEL_FIELDS

    torch.cuda.empty_cache()
    gc.collect()

    raw        = torch.load(checkpoint, map_location=device, weights_only=True)
    raw_config = raw.get("config", {}) if isinstance(raw, dict) else {}
    state_dict = raw["model_state"] if isinstance(raw, dict) and "model_state" in raw else raw
    step       = raw.get("step", "?") if isinstance(raw, dict) else "?"

    model_kwargs                  = {k: v for k, v in raw_config.items() if k in _MODEL_FIELDS}
    model_kwargs["attention_type"] = "paged"

    cfg   = ModelConfig(**model_kwargs)
    model = Transformer(cfg)
    model.load_state_dict(_remap_state_dict(state_dict))

    # fp16 on GPU for memory efficiency
    model = model.half().to(device).eval()

    print(f"Loaded : {checkpoint}")
    print(f"        step={step}, {cfg.n_layers}L/{cfg.dim}d/{cfg.n_heads}h  "
          f"n_kv={cfg.n_kv_heads}  head_dim={cfg.head_dim}")
    return model, raw_config


# ── KV cache builder ──────────────────────────────────────────────────────────

def _build_gpu_cache(
    cfg         : ModelConfig,
    device      : str,
    block_size  : int,
    kv_budget_gb: Optional[float],
    verbose     : bool = True,
) -> tuple[GPUPagedKVCache, BlockAllocator]:
    """
    Allocate the contiguous GPU KV pool and the block allocator.

    If kv_budget_gb is given, the pool is sized to fit within that budget.
    Otherwise the pool uses all free VRAM minus safety reserves.
    """
    if kv_budget_gb is not None:
        kv_bytes = int(kv_budget_gb * 1024 ** 3)
    else:
        kv_bytes = _compute_kv_budget(device)

    if verbose:
        print(f"KV budget: {kv_bytes / 1e6:.0f} MB")

    kv_cache  = GPUPagedKVCache.from_memory_budget(
        free_gpu_bytes = kv_bytes,
        n_layers       = cfg.n_layers,
        block_size     = block_size,
        n_kv_heads     = cfg.n_kv_heads,
        head_dim       = cfg.head_dim,
        device         = device,
        verbose        = verbose,
    )
    allocator = BlockAllocator(kv_cache.n_phys)
    return kv_cache, allocator


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_gpu(
    model          : Transformer,
    tokenizer,
    kv_cache       : GPUPagedKVCache,
    allocator      : BlockAllocator,
    prompt         : str,
    max_new_tokens : int   = 200,
    temperature    : float = 0.8,
    top_k          : Optional[int] = 50,
    top_p          : float = 1.0,
    device         : str   = "cuda",
) -> tuple[str, float, float]:
    """
    Two-phase generation using GPUPagedKVCache + CUDA paged attention kernel.

    Returns:
        (generated_text, prefill_time_s, decode_tok_per_sec)
    """
    cfg      = model.cfg
    n_layers = cfg.n_layers
    block_size = kv_cache.block_size

    block_table = LayeredBlockTable(
        seq_id     = 0,
        n_layers   = n_layers,
        block_size = block_size,
    )

    # Encode prompt
    tokens  = tokenizer.encode(prompt, add_special_tokens=True)
    T_start = len(tokens)
    idx     = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    # ── Phase 1: Prefill ──────────────────────────────────────────────────────
    # Allocate blocks for all prompt tokens — triggers init_blocks_cuda per block
    block_table.append_tokens(T_start, allocator, kv_cache)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits  = model(idx, start_pos=0, block_table=block_table, kv_cache=kv_cache)
    logits  = logits[:, -1, :].float()
    next_id = _sample(logits, temperature, top_k, top_p)

    torch.cuda.synchronize(device)
    prefill_time = time.perf_counter() - t0
    generated    = [next_id]

    if next_id == tokenizer.eos_token_id:
        text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
        block_table.free(allocator, kv_cache)
        return text, prefill_time, float("inf")

    # ── Phase 2: Decode ───────────────────────────────────────────────────────
    torch.cuda.synchronize(device)
    t_decode = time.perf_counter()

    for i in range(max_new_tokens - 1):
        block_table.append_token(allocator, kv_cache)
        next_tok = torch.tensor([[next_id]], dtype=torch.long, device=device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                next_tok,
                start_pos   = T_start + i,
                block_table = block_table,
                kv_cache    = kv_cache,
            )
        logits  = logits[:, -1, :].float()
        next_id = _sample(logits, temperature, top_k, top_p)
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize(device)
    decode_elapsed = time.perf_counter() - t_decode
    n_decode_toks  = len(generated) - 1
    decode_tok_s   = n_decode_toks / decode_elapsed if decode_elapsed > 0 else float("inf")

    # Free blocks
    block_table.free(allocator, kv_cache)

    text = tokenizer.decode(tokens + generated, skip_special_tokens=True)
    return text, prefill_time, decode_tok_s


# ── Memory report ─────────────────────────────────────────────────────────────

def print_gpu_memory_report(
    cfg        : ModelConfig,
    kv_cache   : GPUPagedKVCache,
    device     : str,
    model_mb   : float,
    block_size : int,
) -> None:
    """
    Print GPU memory breakdown:
      - Model weights
      - KV pool (total allocated)
      - Free VRAM remaining
      - Logical capacity vs flat cache comparison
    """
    torch.cuda.synchronize(device)
    mem = _gpu_memory_mb(device)

    pool_mb      = kv_cache.memory_pool_mb()
    n_logical    = kv_cache.n_phys // max(cfg.n_layers, 1)
    max_tokens   = n_logical * block_size
    dtype_bytes  = 2  # fp16

    flat_mb = (
        2 * cfg.n_layers * 1 * cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim
        * dtype_bytes / 1e6
    )
    savings_pct = (1 - pool_mb / flat_mb) * 100 if flat_mb > 0 else 0.0

    print()
    print("=" * 65)
    print("  GPU MEMORY REPORT")
    print("=" * 65)
    print(f"  GPU                 : {mem['total_mb']:.0f} MB total")
    print(f"  Model weights (fp16): {model_mb:.0f} MB")
    print(f"  KV pool (K+V fp16)  : {pool_mb:.0f} MB")
    print(f"    Physical blocks   : {kv_cache.n_phys}")
    print(f"    Logical blocks    : {n_logical}  "
          f"({n_logical * block_size} max tokens)")
    print(f"  Free VRAM remaining : {mem['free_mb']:.0f} MB")
    print()
    print("  KV capacity vs. flat cache (1 sequence, max_seq_len):")
    print(f"    Flat (pre-alloc)  : {flat_mb:.1f} MB  "
          f"({cfg.max_seq_len} tokens, always)")
    print(f"    Paged pool        : {pool_mb:.1f} MB  "
          f"({max_tokens} token capacity)")
    print(f"    Pool vs flat      : {savings_pct:+.1f}%  "
          f"({'more' if savings_pct < 0 else 'less'} memory)")
    print("=" * 65)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLaMA GPU inference — contiguous KV pool + CUDA paged attention"
    )
    p.add_argument("--checkpoint",   required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--prompt",       default="Once upon a time",
                   help="Input prompt text")
    p.add_argument("--max_new_tokens", type=int,   default=200)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50,
                   help="Top-k sampling (0 = disabled)")
    p.add_argument("--top_p",          type=float, default=1.0,
                   help="Nucleus sampling threshold (1.0 = disabled)")
    p.add_argument("--block_size",     type=int,   default=16,
                   help="Tokens per KV block")
    p.add_argument("--kv_budget_gb",   type=float, default=None,
                   help="GB to reserve for KV pool "
                        "(default: all free VRAM minus 1.25 GB reserve)")
    p.add_argument("--activation_reserve_gb", type=float, default=1.0,
                   help="GB to reserve for forward-pass activations")
    p.add_argument("--safety_margin_gb",      type=float, default=0.25,
                   help="Additional CUDA overhead / fragmentation margin")
    p.add_argument("--device",   default="cuda",
                   help="CUDA device (e.g. 'cuda', 'cuda:0')")
    p.add_argument("--tokenizer", default="huggyllama/llama-7b")
    p.add_argument("--memory_report", action="store_true",
                   help="Print GPU memory report only (no generation)")
    p.add_argument("--verbose",   action="store_true", default=True,
                   help="Show CUDA kernel compilation progress")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use generate_paged_cpu.py for CPU inference.")
        sys.exit(1)

    device = args.device

    # ── 1. Measure baseline GPU memory ───────────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()
    free_before, total = torch.cuda.mem_get_info(device)
    print(f"GPU: {torch.cuda.get_device_name(device)}  "
          f"({total / 1e9:.2f} GB total, {free_before / 1e9:.2f} GB free)")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    model, _ = _load_model_gpu(args.checkpoint, device)
    cfg      = model.cfg

    torch.cuda.synchronize(device)
    free_after_model, _ = torch.cuda.mem_get_info(device)
    model_mb = (free_before - free_after_model) / 1e6
    print(f"Model loaded: {model_mb:.0f} MB on GPU")

    # ── 3. Allocate KV pool ───────────────────────────────────────────────────
    if args.kv_budget_gb is not None:
        kv_bytes = int(args.kv_budget_gb * 1024 ** 3)
    else:
        kv_bytes = _compute_kv_budget(
            device                = device,
            activation_reserve_gb = args.activation_reserve_gb,
            safety_margin_gb      = args.safety_margin_gb,
        )
    print(f"KV budget: {kv_bytes / 1e6:.0f} MB  "
          f"(reserves: {args.activation_reserve_gb + args.safety_margin_gb:.2f} GB)")

    kv_cache, allocator = _build_gpu_cache(
        cfg          = cfg,
        device       = device,
        block_size   = args.block_size,
        kv_budget_gb = args.kv_budget_gb,
        verbose      = args.verbose,
    )

    if args.memory_report:
        print_gpu_memory_report(cfg, kv_cache, device, model_mb, args.block_size)
        return

    # ── 4. Load tokenizer ─────────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    top_k     = args.top_k if args.top_k > 0 else None

    print(f"\nPrompt    : {args.prompt!r}")
    print(f"Sampling  : temperature={args.temperature}, "
          f"top_k={top_k}, top_p={args.top_p}")
    print(f"KV kernel : CUDA paged_attn_kernel  "
          f"(grid=(T_q,{cfg.n_heads}), block=({cfg.head_dim},))")
    print("-" * 65)

    # ── 5. Generate ───────────────────────────────────────────────────────────
    text, prefill_t, decode_tok_s = generate_gpu(
        model          = model,
        tokenizer      = tokenizer,
        kv_cache       = kv_cache,
        allocator      = allocator,
        prompt         = args.prompt,
        max_new_tokens = args.max_new_tokens,
        temperature    = args.temperature,
        top_k          = top_k,
        top_p          = args.top_p,
        device         = device,
    )

    print(text)
    print("-" * 65)
    print(f"Prefill : {prefill_t * 1000:.1f} ms")
    print(f"Decode  : {decode_tok_s:.1f} tok/s  (GPU paged KV cache + CUDA kernel)")

    print_gpu_memory_report(cfg, kv_cache, device, model_mb, args.block_size)


if __name__ == "__main__":
    main()

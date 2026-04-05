"""
paged_attention() — gather-based attention kernel for paged KV cache.

Unlike standard attention that reads from a contiguous (B, n_heads, T, head_dim)
tensor, this function:

  1. Gathers K and V token-by-token from the physical block pool using the
     block table's logical→physical address translation.
  2. Reshapes the gathered tensors to the layout expected by the dot-product.
  3. Runs standard scaled dot-product attention over the gathered K/V.

This is the functional core of vLLM's paged attention, expressed as a plain
Python function operating on torch tensors.  The future CUDA kernel in
kernels/paged_attention.cu will replace step 1-3 with a fused gather+attention
pass that avoids materialising the full K/V tensor in HBM.

Shapes convention (single sequence, B=1):
    q           : (1, n_heads,    T_q, head_dim)
    k_gathered  : (T_kv, n_kv_heads, head_dim)   ← gathered from paged pool
    k (reshaped): (1, n_kv_heads, T_kv, head_dim)
    output      : (1, T_q, dim)                  ← merged heads
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .block_table import LayeredBlockTable
    from .paged_kv_cache import PagedKVCache


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads for GQA: (B, n_kv, T, hd) → (B, n_heads, T, hd)."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


def paged_attention(
    q           : torch.Tensor,          # (1, n_heads, T_q, head_dim)
    layer_idx   : int,
    block_table : "LayeredBlockTable",
    kv_cache    : "PagedKVCache",
    n_heads     : int,
    n_kv_heads  : int,
    head_dim    : int,
    start_pos   : int,
    causal_mask : Optional[torch.Tensor] = None,   # (1, 1, max_T, max_T) additive
    dropout_p   : float = 0.0,
) -> torch.Tensor:
    """
    Paged attention for a single sequence.

    Steps:
      1. Gather K, V from scattered physical blocks → (T_kv, n_kv_heads, head_dim)
      2. Permute / unsqueeze to attention layout   → (1, n_kv_heads, T_kv, head_dim)
      3. GQA expansion: repeat KV heads            → (1, n_heads,    T_kv, head_dim)
      4. Scaled dot-product: Q @ K^T / sqrt(d)
      5. Causal mask (applied during prefill only; skipped when T_q == 1)
      6. Softmax → weighted sum over V
      7. Merge heads                               → (1, T_q, n_heads * head_dim)

    Args:
        q           : Query tensor, already RoPE-rotated.
        layer_idx   : Which transformer layer is calling (determines physical blocks).
        block_table : Sequence's logical→physical block map.
        kv_cache    : Physical KV pool to gather from.
        n_heads     : Number of query heads.
        n_kv_heads  : Number of KV heads (≤ n_heads for GQA).
        head_dim    : Dimension per head.
        start_pos   : Absolute position of q[0] in the sequence (for causal mask).
        causal_mask : Full (1, 1, max_T, max_T) additive mask buffer.
        dropout_p   : Attention weight dropout probability.

    Returns:
        out : (1, T_q, n_heads * head_dim)
    """
    T_q   = q.shape[2]
    n_rep = n_heads // n_kv_heads

    # ── 1. Gather K, V from paged pool ───────────────────────────────────────
    # k_gathered : (T_kv, n_kv_heads, head_dim)
    k_gathered, v_gathered = kv_cache.read_sequence(layer_idx, block_table)
    T_kv = k_gathered.shape[0]

    # ── 2. Reshape to (1, n_kv_heads, T_kv, head_dim) ────────────────────────
    k = k_gathered.permute(1, 0, 2).unsqueeze(0).to(q.dtype)
    v = v_gathered.permute(1, 0, 2).unsqueeze(0).to(q.dtype)

    # ── 3. GQA expansion ─────────────────────────────────────────────────────
    k = repeat_kv(k, n_rep)   # (1, n_heads, T_kv, head_dim)
    v = repeat_kv(v, n_rep)

    # ── 4. Scaled dot-product ─────────────────────────────────────────────────
    scale  = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (1, n_heads, T_q, T_kv)

    # ── 5. Causal mask ────────────────────────────────────────────────────────
    # During decode: T_q == 1 — the single query can attend to all past tokens,
    # no future tokens exist, so no masking is needed.
    # During prefill: T_q > 1 — apply the standard upper-triangular causal mask.
    if causal_mask is not None and T_q > 1:
        scores = scores + causal_mask[:, :, start_pos : start_pos + T_q, :T_kv]

    # ── 6. Softmax + weighted sum ─────────────────────────────────────────────
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p)
    out = torch.matmul(weights, v)                           # (1, n_heads, T_q, head_dim)

    # ── 7. Merge heads ────────────────────────────────────────────────────────
    out = out.transpose(1, 2).contiguous()                   # (1, T_q, n_heads, head_dim)
    out = out.view(1, T_q, n_heads * head_dim)               # (1, T_q, dim)
    return out

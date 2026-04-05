"""
Positional encodings applied inside attention.

The BasePositionalEncoding interface operates on (Q, K) tensors after head
splitting, so every implementation sees:
  q : (B, n_heads,    T, head_dim)
  k : (B, n_kv_heads, T, head_dim)
  start_pos : int   — first absolute position in this forward pass (for KV-cache decoding)

Current implementations:
  - RotaryEmbedding (RoPE) : LLaMA / Llama-2 / Mistral default
  - NoPositionalEncoding   : pass-through; position handled elsewhere (e.g. ALiBi in attention)

Planned / stub:
  - ALiBiEncoding          : additive linear bias on attention scores (not Q/K rotation)

Adding a new encoding:
  1. Subclass BasePositionalEncoding
  2. Register in build_pos_encoding()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from .config import ModelConfig


class BasePositionalEncoding(nn.Module, ABC):
    """
    Positional encoding applied to Q and K inside attention.

    Implementations must be stateless w.r.t. sequence position so they work
    with KV-cache decoding (start_pos > 0).
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,          # (B, n_heads,    T, head_dim)
        k: torch.Tensor,          # (B, n_kv_heads, T, head_dim)
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


# ── RoPE ─────────────────────────────────────────────────────────────────────

class RotaryEmbedding(BasePositionalEncoding):
    """
    Rotary Positional Embedding (Su et al., 2021) as used in LLaMA.

    Idea: rotate Q and K by a position-dependent angle in each 2D subspace of
    head_dim.  The inner product q·k then depends only on the *relative*
    distance (m - n), giving the model implicit relative position information
    without any learned parameters (only the fixed frequency table).

    Frequencies:  θ_i = 1 / (theta ^ (2i / head_dim))   for i in 0..head_dim/2
    Rotation at pos m: q_rot = q*cos(m·θ) + rotate_half(q)*sin(m·θ)

    The cos/sin tables are precomputed up to max_seq_len and registered as
    non-persistent buffers so they move with .to(device) / .half() calls.
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        cos_table, sin_table = _precompute_rope_freqs(head_dim, max_seq_len, theta)
        # persistent=False: not saved in state_dict (recomputed from config)
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        cos = self.cos_table[start_pos : start_pos + T]   # (T, head_dim)
        sin = self.sin_table[start_pos : start_pos + T]
        return _apply_rope(q, k, cos, sin)


class NoPositionalEncoding(BasePositionalEncoding):
    """
    Pass-through positional encoding.

    Use when:
      - The model has no positional encoding (rare)
      - Position is encoded via attention bias (e.g. ALiBi) rather than Q/K rotation
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return q, k


# ── Internal helpers (not exported) ──────────────────────────────────────────

def _precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin tables of shape (max_seq_len, head_dim).

    Each pair of dimensions (2i, 2i+1) shares frequency θ_i.
    repeat_interleave doubles the (T, head_dim/2) table to (T, head_dim) so
    the same cos/sin value applies to both elements of each pair.
    """
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    # θ_i = 1 / theta^(2i/head_dim),  shape (head_dim/2,)
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    # positions 0..max_seq_len-1
    t = torch.arange(max_seq_len, dtype=torch.float32)
    # outer product → (max_seq_len, head_dim/2)
    freqs = torch.outer(t, freqs)
    # duplicate each frequency for both elements of each 2D subspace
    cos_table = torch.cos(freqs).repeat_interleave(2, dim=-1)  # (T, head_dim)
    sin_table = torch.sin(freqs).repeat_interleave(2, dim=-1)
    return cos_table, sin_table


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate each 2D subspace by 90°:
      [..., x_2i, x_{2i+1}, ...] → [..., -x_{2i+1}, x_2i, ...]

    This is the "rotate_half" trick: pairing even/odd indices and flipping sign
    gives the 90° rotation without an explicit complex-number representation.
    """
    x_even = x[..., 0::2]   # (*, head_dim/2) — dimensions 0, 2, 4, ...
    x_odd  = x[..., 1::2]   # (*, head_dim/2) — dimensions 1, 3, 5, ...
    return torch.stack([-x_odd, x_even], dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,    # (B, n_heads,    T, head_dim)
    k: torch.Tensor,    # (B, n_kv_heads, T, head_dim)
    cos: torch.Tensor,  # (T, head_dim)
    sin: torch.Tensor,  # (T, head_dim)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding:  x_rot = x*cos + rotate_half(x)*sin
    cos/sin are broadcast over the batch and head dimensions.
    Output dtype matches input dtype (computation done in input precision).
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot.type_as(q), k_rot.type_as(k)


# ── Factory ──────────────────────────────────────────────────────────────────

def build_pos_encoding(cfg: ModelConfig) -> BasePositionalEncoding:
    """
    Instantiate the positional encoding specified in cfg.pos_encoding.

    Usage:
        pe = build_pos_encoding(cfg)
    """
    if cfg.pos_encoding == "rope":
        return RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
    if cfg.pos_encoding == "none":
        return NoPositionalEncoding()
    raise ValueError(
        f"Unknown pos_encoding: {cfg.pos_encoding!r}. "
        f"Available: 'rope', 'none'"
    )

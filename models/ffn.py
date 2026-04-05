"""
Feed-Forward Network (FFN) sub-layers.

Current implementations:
  - SwiGLUFFN  : LLaMA default — SiLU gated linear unit with 3 weight matrices
  - GeGLUFFN   : GEGLU variant (Shazeer 2020) — GeLU instead of SiLU
  - StandardMLP: classic 2-layer MLP with GeLU (GPT-2 style)

Adding a new FFN:
  1. Subclass BaseFeedForward
  2. Register in _FFN_REGISTRY and build_ffn()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class BaseFeedForward(nn.Module, ABC):
    """Interface contract for all FFN variants."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class SwiGLUFFN(BaseFeedForward):
    """
    SwiGLU Feed-Forward Network (Shazeer 2020, used in LLaMA).

    Architecture:
        gate  = SiLU(x · W_gate)          (B, T, hidden)
        up    =      x · W_up             (B, T, hidden)
        out   = (gate ⊙ up) · W_down      (B, T, dim)

    Three weight matrices instead of two; no bias anywhere (LLaMA convention).
    Hidden dim is set to ffn_dim_mult * dim, rounded to a multiple of 256.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.ffn_hidden_dim
        self.w_gate = nn.Linear(cfg.dim, h, bias=False)
        self.w_up   = nn.Linear(cfg.dim, h, bias=False)
        self.w_down = nn.Linear(h, cfg.dim, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GeGLUFFN(BaseFeedForward):
    """
    GEGLU Feed-Forward Network (Shazeer 2020).

    Same structure as SwiGLU but uses GeLU instead of SiLU for the gate.
    Empirically similar quality; swap in by setting ffn_type: "geglu".
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = cfg.ffn_hidden_dim
        self.w_gate = nn.Linear(cfg.dim, h, bias=False)
        self.w_up   = nn.Linear(cfg.dim, h, bias=False)
        self.w_down = nn.Linear(h, cfg.dim, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x)))


class StandardMLP(BaseFeedForward):
    """
    Classic 2-layer MLP with GeLU activation (GPT-2 / BERT style).

    hidden = 4 * dim by default (matches GPT-2 convention).
    Use for baselines or when porting non-LLaMA architectures.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        h = 4 * cfg.dim
        self.fc1  = nn.Linear(cfg.dim, h)
        self.fc2  = nn.Linear(h, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ── Factory ──────────────────────────────────────────────────────────────────

_FFN_REGISTRY: dict[str, type[BaseFeedForward]] = {
    "swiglu": SwiGLUFFN,
    "geglu":  GeGLUFFN,
    "mlp":    StandardMLP,
}


def build_ffn(cfg: ModelConfig) -> BaseFeedForward:
    """
    Instantiate the FFN specified in cfg.ffn_type.

    Usage:
        ffn = build_ffn(cfg)
    """
    ffn_cls = _FFN_REGISTRY.get(cfg.ffn_type)
    if ffn_cls is None:
        raise ValueError(
            f"Unknown ffn_type: {cfg.ffn_type!r}. "
            f"Available: {list(_FFN_REGISTRY)}"
        )
    return ffn_cls(cfg)

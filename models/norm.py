"""
Normalization layers.

Current implementations:
  - RMSNorm   : LLaMA-style, no mean subtraction, no bias; faster than LayerNorm
  - LayerNorm : standard torch.nn.LayerNorm, available as a drop-in via build_norm()

Adding a new norm (e.g. DeepNorm, CRMSNorm):
  1. Subclass BaseNorm
  2. Register in build_norm()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .config import ModelConfig


class BaseNorm(nn.Module, ABC):
    """Interface contract for all normalization layers used inside the transformer."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


class RMSNorm(BaseNorm):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    RMS(x) = sqrt(mean(x²) + ε)
    output  = γ * (x / RMS(x))

    Advantages over LayerNorm:
      - No mean subtraction  → fewer ops
      - No β bias parameter  → fewer parameters
      - Numerically well-behaved for fp16 / bf16

    Shape: (*, dim) → (*, dim)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 for numerical stability, then cast back to input dtype
        return self.gamma * self._norm(x.float()).type_as(x)


class LayerNorm(BaseNorm):
    """
    Standard LayerNorm wrapper that satisfies BaseNorm.
    Useful when comparing architectures or porting GPT-2-style models.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self._norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x)


# ── Factory ──────────────────────────────────────────────────────────────────

_NORM_REGISTRY: dict[str, type[BaseNorm]] = {
    "rmsnorm": RMSNorm,
    "layernorm": LayerNorm,
}


def build_norm(cfg: ModelConfig) -> BaseNorm:
    """
    Instantiate the norm layer specified in cfg.norm_type.

    Usage:
        norm = build_norm(cfg)         # uses cfg.norm_type, cfg.dim, cfg.norm_eps
    """
    norm_cls = _NORM_REGISTRY.get(cfg.norm_type)
    if norm_cls is None:
        raise ValueError(
            f"Unknown norm_type: {cfg.norm_type!r}. "
            f"Available: {list(_NORM_REGISTRY)}"
        )
    return norm_cls(cfg.dim, cfg.norm_eps)

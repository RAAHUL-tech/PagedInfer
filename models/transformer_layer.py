"""
Single transformer decoder layer (LLaMA-style).

Layout:
    x ──► RMSNorm ──► Attention ──►(+)──► RMSNorm ──► FFN ──►(+)──►
    │                               ▲                           ▲
    └───────────────────────────────┘   ───────────────────────┘
    (pre-norm + residual for both sub-layers)

Both the norm variant and the FFN variant are read from ModelConfig, so the
layer transparently uses whatever is registered in norm.py and ffn.py.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import BaseAttention, build_attention
from .config import ModelConfig
from .ffn import BaseFeedForward, build_ffn
from .norm import BaseNorm, build_norm


class TransformerLayer(nn.Module):
    """
    One decoder layer: pre-norm attention + pre-norm FFN, both with residuals.

    All sub-components are built from ModelConfig so the layer is fully
    config-driven.  Swap attention type, norm, and FFN in the YAML without
    touching this file.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        self.attention_norm: BaseNorm        = build_norm(cfg)
        self.attention:      BaseAttention   = build_attention(cfg, layer_idx)
        self.ffn_norm:       BaseNorm        = build_norm(cfg)
        self.ffn:            BaseFeedForward = build_ffn(cfg)

    def forward(
        self,
        x         : torch.Tensor,                   # (B, T, dim)
        mask      : Optional[torch.Tensor] = None,  # (1, 1, max_T, max_T) additive
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        # Attention sub-layer: pre-norm → attention → residual
        h = x + self.attention(
            self.attention_norm(x),
            mask=mask,
            use_cache=use_cache,
            start_pos=start_pos,
        )
        # FFN sub-layer: pre-norm → FFN → residual
        return h + self.ffn(self.ffn_norm(h))

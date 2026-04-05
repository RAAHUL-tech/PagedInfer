"""
Token embedding table.

PagedInfer uses a decoder-only architecture with no separate positional
embedding table — positions are encoded via RoPE inside each attention layer.
The token embedding weight is tied to the LM head (see Transformer.__init__).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Maps integer token IDs → dense vectors of shape (B, T, dim).

    Weight tying:
        The embedding weight matrix is shared with the final lm_head projection
        (Press & Wolf, 2017).  The Transformer class sets
            lm_head.weight = token_emb.embedding.weight
        after constructing both modules.

    Initialization:
        Normal(0, 0.02) — same as GPT-2 / LLaMA weight init.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    @property
    def weight(self) -> torch.Tensor:
        """Expose weight directly so Transformer can tie it to lm_head."""
        return self.embedding.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx : (B, T)  — integer token IDs
        Returns:
            x   : (B, T, dim) — token embeddings
        """
        return self.embedding(idx)

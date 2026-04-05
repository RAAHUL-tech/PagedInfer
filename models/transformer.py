"""
Full LLaMA-style decoder-only transformer.

Design decisions (inherited from the notebook, preserved for inference):
  - No positional embedding table — positions are encoded by RoPE inside attention
  - RMSNorm everywhere (no mean subtraction, no beta bias)
  - SwiGLU FFN (3 weight matrices, no bias)
  - No bias in any linear layer
  - Weight tying: token_emb.weight == lm_head.weight  (Press & Wolf 2017)
  - Pre-norm architecture (norm before sub-layer, not after)

Inference paths:
  forward(idx)                       — plain forward pass, no cache
  forward(idx, use_cache=True)       — prefill: fills flat KV caches in all layers
  forward(idx, use_cache=True,
          start_pos=n)               — decode step: single token, reads from cache

KV cache lifecycle:
  clear_kv_cache() between requests; each Attention layer holds its own
  flat (cache_k, cache_v) tensors.  These will be replaced by the paged
  block allocator once kv_cache/ is implemented — the interface here does
  not change (use_cache / start_pos / clear_kv_cache remain the same).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from kv_cache.block_table import LayeredBlockTable
    from kv_cache.paged_kv_cache import PagedKVCache

from .config import ModelConfig
from .embeddings import TokenEmbedding
from .norm import BaseNorm, build_norm
from .transformer_layer import TransformerLayer


class Transformer(nn.Module):
    """
    Decoder-only LLaMA-style transformer for inference.

    Args:
        cfg : ModelConfig — all architecture and backend settings
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Token embedding (no positional embedding table — RoPE handles position)
        self.token_emb = TokenEmbedding(cfg)

        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(cfg, layer_idx=i)
            for i in range(cfg.n_layers)
        ])

        # Final normalisation before the LM head
        self.norm: BaseNorm = build_norm(cfg)

        # Language model head: dim → vocab_size (no bias)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Weight tying: embedding and LM head share the same matrix
        self.lm_head.weight = self.token_emb.weight

        # Causal mask (additive): 0 for allowed positions, -inf for future ones.
        # Shape (1, 1, max_seq_len, max_seq_len) — broadcast over batch and heads.
        # Registered as a buffer so it moves to the right device with .to().
        mask = torch.full(
            (1, 1, cfg.max_seq_len, cfg.max_seq_len), float("-inf")
        )
        mask = torch.triu(mask, diagonal=1)   # upper triangle = future tokens
        self.register_buffer("causal_mask", mask)

        self.apply(self._init_weights)

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        idx         : torch.Tensor,
        use_cache   : bool = False,
        start_pos   : int  = 0,
        block_table : Optional["LayeredBlockTable"] = None,
        kv_cache    : Optional["PagedKVCache"] = None,
    ) -> torch.Tensor:
        """
        Args:
            idx         : (B, T) integer token IDs.
            use_cache   : populate / read flat KV caches (gqa / vanilla path).
            start_pos   : absolute position of idx[0] in the sequence.
            block_table : per-sequence block map (paged attention path).
            kv_cache    : physical KV pool (paged attention path).
        Returns:
            logits : (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        x    = self.token_emb(idx)
        mask = self.causal_mask

        for layer in self.layers:
            x = layer(
                x,
                mask=mask,
                use_cache=use_cache,
                start_pos=start_pos,
                block_table=block_table,
                kv_cache=kv_cache,
            )

        x      = self.norm(x)
        logits = self.lm_head(x)
        return logits

    # ── Utility ───────────────────────────────────────────────────────────────

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def clear_kv_cache(self) -> None:
        """
        Reset the flat KV cache in every attention layer.
        Call between requests to avoid cache contamination.
        """
        for layer in self.layers:
            layer.attention.clear_cache()

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "Transformer":
        return cls(cfg)

    @classmethod
    def from_yaml(cls, path: str) -> "Transformer":
        cfg = ModelConfig.from_yaml(path)
        return cls(cfg)

    def load_weights(self, path: str, device: Optional[str] = None) -> None:
        """
        Load a checkpoint saved with torch.save(model.state_dict(), path).

        Args:
            path   : Path to the .pt / .pth checkpoint file.
            device : Target device string, e.g. "cuda" or "cpu".
                     Defaults to the device of the first parameter.
        """
        if device is None:
            device = next(self.parameters()).device.type
        state = torch.load(path, map_location=device)
        # Support both bare state_dicts and dicts with a "model_state" key
        if "model_state" in state:
            state = state["model_state"]
        self.load_state_dict(state)

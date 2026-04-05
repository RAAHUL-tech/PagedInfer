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

from typing import Optional, Tuple

import torch
import torch.nn as nn

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
        idx       : torch.Tensor,   # (B, T)  integer token IDs
        use_cache : bool = False,    # True → populate / read flat KV caches
        start_pos : int  = 0,       # absolute position of the first token in idx
    ) -> torch.Tensor:              # (B, T, vocab_size) logits
        """
        Args:
            idx       : Token IDs, shape (B, T).
            use_cache : If True, each Attention layer stores K/V for future
                        decode steps.  Set start_pos accordingly.
            start_pos : Index of the first token in `idx` within the full
                        sequence.  0 for prefill; len(prompt) for the first
                        decode step.
        Returns:
            logits    : (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        )

        # 1. Token embeddings — no positional offset; RoPE handles position
        x = self.token_emb(idx)                       # (B, T, dim)

        # 2. Full causal mask — each Attention layer slices the rows/cols it
        #    needs using start_pos, so the full mask must be passed here.
        mask = self.causal_mask   # (1, 1, max_seq_len, max_seq_len)

        # 3. Pass through all transformer layers
        for layer in self.layers:
            x = layer(x, mask=mask, use_cache=use_cache, start_pos=start_pos)

        # 4. Final normalisation
        x = self.norm(x)                              # (B, T, dim)

        # 5. Project to vocabulary
        logits = self.lm_head(x)                      # (B, T, vocab_size)
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

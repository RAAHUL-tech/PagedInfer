from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """
    All architecture hyperparameters for the transformer model.

    Loaded from configs/model_config.yaml; every field maps 1-to-1 to a YAML key
    under the `model:` section.

    Pluggable backend fields let you swap implementations without touching model code:
      - attention_type : "vanilla" | "gqa" | "flash" | "paged"
      - pos_encoding   : "rope"    | "none"
      - norm_type      : "rmsnorm" | "layernorm"
      - ffn_type       : "swiglu"  | "geglu"  | "mlp"
    """

    # Vocabulary
    vocab_size: int = 32000

    # Architecture
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 6
    max_seq_len: int = 512
    ffn_dim_mult: float = 2.6875
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    dropout: float = 0.0

    # Pluggable backends
    attention_type: str = "gqa"       # "vanilla" | "gqa" | "flash" | "paged"
    pos_encoding: str = "rope"        # "rope" | "none"
    norm_type: str = "rmsnorm"        # "rmsnorm" | "layernorm"
    ffn_type: str = "swiglu"          # "swiglu" | "geglu" | "mlp"

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.dim // self.n_heads

    @property
    def ffn_hidden_dim(self) -> int:
        """SwiGLU hidden dimension, rounded up to the nearest multiple of 256."""
        raw = int(self.ffn_dim_mult * self.dim)
        return (raw + 255) // 256 * 256

    @property
    def n_kv_rep(self) -> int:
        """How many times each KV head is repeated to match Q heads (GQA)."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads

    # ── Construction helpers ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        """Load config from a YAML file (reads the `model:` section)."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        model_data = data.get("model", {})
        return cls(**model_data)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        return cls(**d)

    def __post_init__(self) -> None:
        # Eagerly validate so errors surface at construction, not forward pass
        _ = self.head_dim
        _ = self.n_kv_rep

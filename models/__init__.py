"""
models/ — PagedInfer model components.

Public API:
    ModelConfig      — all hyperparameters (load from configs/model_config.yaml)
    Transformer      — full decoder-only LLaMA-style model

Internal modules (import directly when extending):
    norm.py          — RMSNorm, LayerNorm, build_norm(), BaseNorm
    rope.py          — RotaryEmbedding, build_pos_encoding(), BasePositionalEncoding
    embeddings.py    — TokenEmbedding
    ffn.py           — SwiGLUFFN, GeGLUFFN, StandardMLP, build_ffn(), BaseFeedForward
    attention.py     — Attention, StandardAttention, FlashAttentionBackend,
                       PagedAttentionBackend, build_attention_backend(), AttentionBackend
    transformer_layer.py — TransformerLayer
    transformer.py   — Transformer
"""

from .config import ModelConfig
from .transformer import Transformer

__all__ = ["ModelConfig", "Transformer"]

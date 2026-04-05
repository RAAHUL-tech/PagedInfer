"""
Attention implementations for PagedInfer.

Class hierarchy:
  BaseAttention          — abstract interface every attention class must satisfy
    VanillaAttention     — standard MHA (n_kv_heads == n_heads)
    GroupedQueryAttention— GQA / MQA  (n_kv_heads <= n_heads)
    FlashAttention       — stub; will use fused CUDA kernel
    PagedAttention       — stub; will use paged KV-cache block allocator

All classes share the same forward() signature so TransformerLayer is
backend-agnostic.  Swap by changing cfg.attention_type (see model_config.yaml).

Adding a new attention variant:
  1. Subclass BaseAttention
  2. Implement forward() and clear_cache()
  3. Register in _ATTENTION_REGISTRY and build_attention()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .rope import BasePositionalEncoding, build_pos_encoding


# ── Shared helper ─────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match the number of Q heads (GQA / MQA).

    Args:
        x     : (B, n_kv_heads, T, head_dim)
        n_rep : n_heads // n_kv_heads
    Returns:
              : (B, n_heads, T, head_dim)
    No-op when n_rep == 1 (standard MHA).
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]              # (B, n_kv, 1,     T, hd)
        .expand(B, n_kv, n_rep, T, hd)  # (B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseAttention(nn.Module, ABC):
    """
    Interface every attention implementation must satisfy.

    forward() contract:
        x         : (B, T, dim)           — input hidden states
        mask      : (1, 1, max_T, max_T)  — additive causal mask (0 / -inf)
        use_cache : bool                  — populate / read flat KV cache
        start_pos : int                   — absolute position of x[0] in the sequence
        Returns   : (B, T, dim)           — output hidden states

    clear_cache() contract:
        Reset any stored KV state. Called between requests.
    """

    @abstractmethod
    def forward(
        self,
        x         : torch.Tensor,
        mask      : Optional[torch.Tensor] = None,
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def clear_cache(self) -> None:
        ...


# ── Vanilla Multi-Head Attention (MHA) ───────────────────────────────────────

class VanillaAttention(BaseAttention):
    """
    Standard Multi-Head Attention (Vaswani et al. 2017).

    n_kv_heads == n_heads — every head has its own K and V projection.
    No GQA expansion; simpler and faster for small models.

    Forward:
        Q = x · Wq       (B, n_heads, T, head_dim)
        K = x · Wk       (B, n_heads, T, head_dim)
        V = x · Wv       (B, n_heads, T, head_dim)
        scores = Q @ K^T / sqrt(head_dim) + mask
        out    = softmax(scores) @ V
        return out · Wo
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        assert cfg.n_heads == cfg.n_kv_heads, (
            f"VanillaAttention requires n_heads == n_kv_heads, "
            f"got {cfg.n_heads} vs {cfg.n_kv_heads}. "
            f"Use GroupedQueryAttention for GQA / MQA."
        )
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout  = cfg.dropout
        self.layer_idx = layer_idx

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)

        self.pos_enc: BasePositionalEncoding = build_pos_encoding(cfg)

        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self,
        x         : torch.Tensor,
        mask      : Optional[torch.Tensor] = None,
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.pos_enc(q, k, start_pos)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        T_k = k.shape[2]
        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            attn_mask = mask[:, :, start_pos : start_pos + T, :T_k]

        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

    def clear_cache(self) -> None:
        self.cache_k = None
        self.cache_v = None


# ── Grouped Query Attention (GQA / MQA) ──────────────────────────────────────

class GroupedQueryAttention(BaseAttention):
    """
    Grouped Query Attention (Ainslie et al. 2023).

    n_kv_heads <= n_heads.
    Each KV head is shared by (n_heads // n_kv_heads) query heads.

    Special cases:
      n_kv_heads == n_heads  →  standard MHA  (same as VanillaAttention)
      n_kv_heads == 1        →  Multi-Query Attention (MQA)

    Memory savings: KV cache is (n_kv_heads / n_heads) × smaller than MHA.
    Compute:        Q still has n_heads projections; K/V projections shrink.

    Forward:
        Q = x · Wq       (B, n_heads,    T, head_dim)
        K = x · Wk       (B, n_kv_heads, T, head_dim)
        V = x · Wv       (B, n_kv_heads, T, head_dim)
        K, V = repeat_kv(K, n_rep), repeat_kv(V, n_rep)   → (B, n_heads, T, head_dim)
        scores = Q @ K^T / sqrt(head_dim) + mask
        out    = softmax(scores) @ V
        return out · Wo
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.n_heads    = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep      = cfg.n_kv_rep       # n_heads // n_kv_heads
        self.head_dim   = cfg.head_dim
        self.dropout    = cfg.dropout
        self.layer_idx  = layer_idx

        self.wq = nn.Linear(cfg.dim, cfg.n_heads    * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim,    bias=False)

        self.pos_enc: BasePositionalEncoding = build_pos_encoding(cfg)

        # Flat KV cache — replaced by paged allocator when kv_cache/ lands
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(
        self,
        x         : torch.Tensor,
        mask      : Optional[torch.Tensor] = None,
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project
        q = self.wq(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Positional encoding
        q, k = self.pos_enc(q, k, start_pos)

        # Flat KV cache
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        # Expand KV heads to match Q heads
        k = repeat_kv(k, self.n_rep)   # (B, n_heads, T_k, head_dim)
        v = repeat_kv(v, self.n_rep)

        T_k = k.shape[2]
        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            attn_mask = mask[:, :, start_pos : start_pos + T, :T_k]

        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        if self.dropout > 0.0 and self.training:
            weights = F.dropout(weights, p=self.dropout)
        out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)

    def clear_cache(self) -> None:
        self.cache_k = None
        self.cache_v = None


# ── Flash Attention ───────────────────────────────────────────────────────────

class FlashAttention(BaseAttention):
    """
    Flash Attention (Dao et al. 2022).

    Fused CUDA kernel that computes attention in SRAM tiles, avoiding the
    O(T²) HBM reads/writes of vanilla attention.  Reduces memory bandwidth
    by ~3-5× and enables longer context lengths.

    Implementation plan:
      - Write fused QKV kernel in kernels/flash_attention.cu
      - Replace the body of forward() with a call to that kernel
      - Support GQA (n_kv_heads <= n_heads) natively in the kernel

    Until then: pass.
    """

    def forward(
        self,
        x         : torch.Tensor,
        mask      : Optional[torch.Tensor] = None,
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"FlashAttention not yet implemented "
            f"(x={tuple(x.shape)}, use_cache={use_cache}, start_pos={start_pos}, "
            f"mask={'yes' if mask is not None else 'none'}). "
            f"See kernels/flash_attention.cu."
        )

    def clear_cache(self) -> None:
        pass


# ── Paged Attention ───────────────────────────────────────────────────────────

class PagedAttention(BaseAttention):
    """
    Paged Attention (Kwon et al. 2023 — vLLM).

    KV cache is split into fixed-size blocks (pages) managed by a block
    allocator in kv_cache/.  Non-contiguous physical pages are gathered by
    the attention kernel, eliminating fragmentation and enabling fine-grained
    memory sharing across requests (prefix caching).

    Implementation plan:
      - Wire the block allocator from kv_cache/block_allocator.py
      - Write paged KV gather/scatter kernel in kernels/paged_attention.cu
      - Replace clear_cache() with block table deallocation

    Until then: stub.
    """

    def forward(
        self,
        x         : torch.Tensor,
        mask      : Optional[torch.Tensor] = None,
        use_cache : bool = False,
        start_pos : int  = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"PagedAttention not yet implemented "
            f"(x={tuple(x.shape)}, use_cache={use_cache}, start_pos={start_pos}, "
            f"mask={'yes' if mask is not None else 'none'}). "
            f"See kv_cache/ and kernels/paged_attention.cu."
        )

    def clear_cache(self) -> None:
        pass


# ── Factory ───────────────────────────────────────────────────────────────────

_ATTENTION_REGISTRY: dict[str, type[BaseAttention]] = {
    "vanilla": VanillaAttention,
    "gqa":     GroupedQueryAttention,
    "flash":   FlashAttention,
    "paged":   PagedAttention,
}


def build_attention(cfg: ModelConfig, layer_idx: int = 0) -> BaseAttention:
    """
    Instantiate the attention class specified in cfg.attention_type.

    Usage:
        attn = build_attention(cfg, layer_idx=i)
    """
    cls = _ATTENTION_REGISTRY.get(cfg.attention_type)
    if cls is None:
        raise ValueError(
            f"Unknown attention_type: {cfg.attention_type!r}. "
            f"Available: {list(_ATTENTION_REGISTRY)}"
        )
    return cls(cfg, layer_idx)

"""
engine/forward_pass.py
───────────────────────
One batched transformer forward pass for ALL active sequences in an engine
iteration — both prefill (T_q > 1 tokens) and decode (T_q = 1 token).

Why one pass instead of per-sequence passes?
    Naive approach:   N_prefill passes + N_decode passes per iteration
    This approach:    1 pass — all token embeddings concatenated into one tensor

    Per layer this means:
      ① QKV projections  — one matmul across all (total_q) tokens
      ② Batched RoPE     — per-token absolute positions from a lookup table
      ③ Batch K/V write  — one kv_write CUDA kernel for all tokens across all seqs
                           (kv_io_ops.KVKernelOps.write_kv_cuda)
      ④ Unified attention— ONE paged_attn_unified CUDA kernel for all seqs + heads
                           (attn_continuous_ops.UnifiedAttnKernelOps.forward)
                           • Prefill seqs: causal mask applied inside kernel
                           • Decode seqs:  attends to full context (q_len=1)
      ⑤ Output proj + residual + FFN — one call for all tokens

    Result: constant number of kernel launches per layer regardless of batch size.

Output:
    prefill_logits : List[(chunk_len, vocab)] — last row = first generated token
    decode_logits  : List[(vocab,)]           — one per decode sequence

Called by: engine/continuous_engine.py — ContinuousBatchingEngine.step()
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kv_cache.gpu_paged_kv_cache import GPUPagedKVCache
    from kernels.attn_continuous_ops import UnifiedAttnKernelOps
    from models.transformer import Transformer
    from engine.sequence import SequenceGroup


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the head dimensions — matches models/rope.py convention."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope_batched(
    q       : torch.Tensor,    # (total_q, n_heads, head_dim)
    k       : torch.Tensor,    # (total_q, n_kv_heads, head_dim)
    cos_tbl : torch.Tensor,    # (max_seq_len, head_dim)
    sin_tbl : torch.Tensor,    # (max_seq_len, head_dim)
    pos_ids : torch.Tensor,    # (total_q,) int64 — absolute position per token
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE with per-token positions.

    Selects cos/sin rows by position index then broadcasts over heads.

    Returns q_rot, k_rot with same shapes as inputs.
    """
    cos = cos_tbl[pos_ids].unsqueeze(1)   # (total_q, 1, head_dim)
    sin = sin_tbl[pos_ids].unsqueeze(1)   # (total_q, 1, head_dim)
    q_r = (q * cos + _rotate_half(q) * sin).to(torch.float16)
    k_r = (k * cos + _rotate_half(k) * sin).to(torch.float16)
    return q_r, k_r


@torch.no_grad()
def forward_unified(
    model        : "Transformer",
    prefill_seqs : List["SequenceGroup"],
    decode_seqs  : List["SequenceGroup"],
    kv_cache     : "GPUPagedKVCache",
    attn_ops     : "UnifiedAttnKernelOps",
    device       : str,
    chunk_size   : int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    One batched forward pass for all active sequences.

    Args:
        model        : Transformer model (weights shared, no copy).
        prefill_seqs : Sequences being prefilled this iteration.
        decode_seqs  : Sequences doing one-token decode this iteration.
        kv_cache     : GPU paged KV pool.
        attn_ops     : Compiled unified attention kernel wrapper.
        device       : CUDA device string.
        chunk_size   : Max tokens per prefill chunk.

    Returns:
        (prefill_logits, decode_logits)
        prefill_logits : List of (chunk_len, vocab_size) float tensors.
                         Last row = logits for the first generated token.
        decode_logits  : List of (vocab_size,) float tensors.
    """
    all_seqs = prefill_seqs + decode_seqs
    N        = len(all_seqs)
    if N == 0:
        return [], []

    cfg = model.cfg

    # ── Access RoPE cos/sin tables from the model's positional encoding ───────
    # model.layers[0].attention.pos_enc has _cos_cached / _sin_cached buffers
    # pre-computed up to max_seq_len in RotaryEmbedding.__init__
    rope = model.layers[0].attention.pos_enc
    cos_tbl = rope._cos_cached.to(device)   # (max_seq_len, head_dim)
    sin_tbl = rope._sin_cached.to(device)

    # ── Per-seq metadata ──────────────────────────────────────────────────────
    q_lens_list    : List[int] = []
    q_offsets_list : List[int] = []
    start_pos_list : List[int] = []   # absolute position of first query token

    offset = 0
    for seq in all_seqs:
        if seq in prefill_seqs:
            chunk_start = seq.prefill_offset
            chunk_end   = min(chunk_start + chunk_size, seq.prompt_len)
            t = chunk_end - chunk_start
        else:
            t = 1   # decode: always 1 token
        q_lens_list.append(t)
        q_offsets_list.append(offset)
        # start_pos = tokens already in KV before this call
        start_pos_list.append(seq.block_table.num_tokens - t)
        offset += t

    total_q = offset

    # ── Collect all token ids ─────────────────────────────────────────────────
    all_tok_ids: List[int] = []
    for seq, t, sp in zip(all_seqs, q_lens_list, start_pos_list):
        if seq in prefill_seqs:
            cs = seq.prefill_offset
            all_tok_ids.extend(seq.token_ids[cs : cs + t])
        else:
            all_tok_ids.append(seq.last_token)

    idx = torch.tensor(all_tok_ids, dtype=torch.long, device=device)  # (total_q,)

    # ── Per-token absolute positions for RoPE ─────────────────────────────────
    positions: List[int] = []
    for sp, t in zip(start_pos_list, q_lens_list):
        positions.extend(range(sp, sp + t))
    pos_ids = torch.tensor(positions, dtype=torch.long, device=device)  # (total_q,)

    # ── Build index tensors for the attention kernel ──────────────────────────
    q_lens_t    = torch.tensor(q_lens_list,    dtype=torch.int32, device=device)
    q_offsets_t = torch.tensor(q_offsets_list, dtype=torch.int32, device=device)

    # ── Embedding ─────────────────────────────────────────────────────────────
    x = model.token_emb(idx)   # (total_q, dim)

    # ── Transformer layers ────────────────────────────────────────────────────
    for li, layer in enumerate(model.layers):
        attn   = layer.attention
        x_norm = layer.attention_norm(x)   # (total_q, dim)

        # QKV projections — one matmul for all tokens
        q_proj = attn.wq(x_norm).view(total_q, cfg.n_heads,    cfg.head_dim)
        k_proj = attn.wk(x_norm).view(total_q, cfg.n_kv_heads, cfg.head_dim)
        v_proj = attn.wv(x_norm).view(total_q, cfg.n_kv_heads, cfg.head_dim)

        # Batched RoPE with per-token positions
        q_r, k_r = _apply_rope_batched(q_proj, k_proj, cos_tbl, sin_tbl, pos_ids)
        # q_r: (total_q, n_heads, head_dim) fp16
        # k_r: (total_q, n_kv_heads, head_dim) fp16

        # ── Build per-layer phys/slot arrays for batch K/V write ──────────────
        all_phys : List[int] = []
        all_slots: List[int] = []
        for si, seq in enumerate(all_seqs):
            t  = q_lens_list[si]
            sp = start_pos_list[si]
            for local_pos in range(t):
                pb, slot = seq.block_table.translate(li, sp + local_pos)
                all_phys.append(pb)
                all_slots.append(slot)

        phys_t  = torch.tensor(all_phys,  dtype=torch.int32, device=device)
        slots_t = torch.tensor(all_slots, dtype=torch.int32, device=device)

        # Batch K/V write — one kernel call for all tokens across all seqs
        v_proj_h = v_proj.half().contiguous()
        kv_cache.batch_write_kv(k_r, v_proj_h, phys_t, slots_t)

        # ── Build per-layer block_table tensor for unified attention ──────────
        max_blks  = max(max(seq.block_table.num_blocks, 1) for seq in all_seqs)
        bt_rows   : List[List[int]] = []
        ctx_lens  : List[int] = []
        for seq in all_seqs:
            row = list(seq.block_table.block_ids[li])
            bt_rows.append(row + [0] * (max_blks - len(row)))
            ctx_lens.append(seq.block_table.num_tokens)

        bt_tensor  = torch.tensor(bt_rows,  dtype=torch.int32, device=device)
        ctx_t      = torch.tensor(ctx_lens, dtype=torch.int32, device=device)

        # ── Unified attention — ONE kernel for ALL seqs (prefill + decode) ────
        attn_out = attn_ops.forward(
            q            = q_r,         # (total_q, n_heads, head_dim)
            pool_k       = kv_cache.pool_k,
            pool_v       = kv_cache.pool_v,
            block_table_t= bt_tensor,   # (n_seqs, max_blks)
            context_lens_t= ctx_t,
            q_lens_t     = q_lens_t,
            q_offsets_t  = q_offsets_t,
            total_q      = total_q,
            n_seqs       = N,
            n_heads      = cfg.n_heads,
            n_kv_heads   = cfg.n_kv_heads,
            head_dim     = cfg.head_dim,
            block_size   = kv_cache.block_size,
        )   # (total_q, n_heads, head_dim)

        # ── Output projection + residual + FFN ───────────────────────────────
        attn_flat = attn_out.reshape(total_q, cfg.n_heads * cfg.head_dim)
        x = x + attn.wo(attn_flat.to(x.dtype))
        x = x + layer.ffn(layer.ffn_norm(x))

    # ── Final norm + lm_head ─────────────────────────────────────────────────
    logits_all = model.lm_head(model.norm(x)).float()   # (total_q, vocab)

    # ── Split back into per-seq logit tensors ─────────────────────────────────
    prefill_logits: List[torch.Tensor] = []
    decode_logits : List[torch.Tensor] = []

    for si, seq in enumerate(all_seqs):
        off       = q_offsets_list[si]
        t         = q_lens_list[si]
        seq_logits = logits_all[off : off + t]   # (t, vocab)
        if seq in prefill_seqs:
            prefill_logits.append(seq_logits)    # last row = first generated token
        else:
            decode_logits.append(seq_logits[0])  # (vocab,) for single decode token

    return prefill_logits, decode_logits

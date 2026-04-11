"""
kernels/attn_continuous_ops.py
──────────────────────────────
Compiles attn_continuous.cu and exposes a Python wrapper for the batched
paged attention kernel used by the continuous batching engine (engine/).

This kernel is the key operation that makes continuous batching efficient:
it processes ALL active sequences — both prefill (T_q > 1 tokens) and
decode (T_q = 1 token) — in a SINGLE CUDA kernel launch per layer,
instead of one launch per sequence.

  paged_attn_unified
    - Prefill sequences: query tokens attend causally to their growing context
    - Decode sequences:  single query token attends to the full KV history
    - Causal masking:    kv_end = (ctx_len - q_len) + q_pos_in_seq + 1
    - GQA:               kv_head = q_head / (n_heads / n_kv_heads)
    - Numerics:          online softmax, fp32 accumulation

    K/V are resolved from block_table[seq_idx][logical_block] at each KV
    position inside the kernel — no pre-built phys/slots arrays needed for
    attention (phys/slots are only built for the K/V write step).

    Grid  = (TOTAL_Q_TOKENS, N_HEADS)  — one block per (query_token, head)
    Block = (HEAD_DIM,)

Difference from attn_decode_ops (single-sequence):
  - Accepts block_table tensor (n_seqs, max_blocks) instead of flat phys/slots
  - Resolves which sequence a query token belongs to via q_offsets[]
  - Handles mixed prefill + decode batches in one launch

Public API:
    UnifiedAttnKernelOps.forward(
        q, pool_k, pool_v,
        block_table_t, context_lens_t, q_lens_t, q_offsets_t,
        total_q, n_seqs, n_heads, n_kv_heads, head_dim, block_size
    ) -> (total_q, n_heads, head_dim) fp16

    load_unified_attn_kernels(device, verbose) -> UnifiedAttnKernelOps  # singleton

Compilation:
    nvcc compiles attn_continuous.cu → /tmp/pagedinfer_attn_continuous.so on first use.

Usage:
    from kernels.attn_continuous_ops import load_unified_attn_kernels
    ops = load_unified_attn_kernels()
    out = ops.forward(q, pool_k, pool_v, bt, ctx, qlens, qoff, ...)
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Optional

import torch

_KERNEL_SRC = Path(__file__).parent / "attn_continuous.cu"
_SO_PATH    = Path("/tmp/pagedinfer_attn_continuous.so")


def _so_is_stale() -> bool:
    if not _SO_PATH.exists():
        return True
    return _SO_PATH.stat().st_mtime < _KERNEL_SRC.stat().st_mtime


def _compile(verbose: bool = True) -> None:
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"-arch=sm_{major}{minor}"
    cmd = [
        "nvcc", arch,
        "-O2", "--use_fast_math",
        "-shared",
        "-Xcompiler", "-fPIC",
        "-o", str(_SO_PATH),
        str(_KERNEL_SRC),
    ]
    if verbose:
        print(f"[unified_attn] Compiling for SM {major}.{minor}  →  {_SO_PATH}")
        print(f"[unified_attn] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")
    if verbose:
        sz = _SO_PATH.stat().st_size // 1024
        print(f"[unified_attn] Compiled OK ({sz} KB)")


def _load_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_SO_PATH), mode=ctypes.RTLD_GLOBAL)

    # launch_paged_attn_unified(
    #   q*, pool_k*, pool_v*,
    #   block_table*, context_lens*, q_lens*, q_offsets*,
    #   attn_out*,
    #   total_q_tokens, n_seqs,
    #   n_heads, n_kv_heads, head_dim, block_size, max_blocks_per_seq
    # )
    lib.launch_paged_attn_unified.argtypes = [
        ctypes.c_void_p,   # q
        ctypes.c_void_p,   # pool_k
        ctypes.c_void_p,   # pool_v
        ctypes.c_void_p,   # block_table   (int32*)
        ctypes.c_void_p,   # context_lens  (int32*)
        ctypes.c_void_p,   # q_lens        (int32*)
        ctypes.c_void_p,   # q_offsets     (int32*)
        ctypes.c_void_p,   # attn_out
        ctypes.c_int,      # total_q_tokens
        ctypes.c_int,      # n_seqs
        ctypes.c_int,      # n_heads
        ctypes.c_int,      # n_kv_heads
        ctypes.c_int,      # head_dim
        ctypes.c_int,      # block_size
        ctypes.c_int,      # max_blocks_per_seq
    ]
    lib.launch_paged_attn_unified.restype = None
    return lib


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    assert t.is_contiguous(), "tensor must be contiguous before CUDA kernel"
    return ctypes.c_void_p(t.data_ptr())


class UnifiedAttnKernelOps:
    """
    Python wrapper around paged_attn_unified.

    Handles all sequences (prefill + decode) in one CUDA kernel call.
    Physical addresses for K/V are looked up from block_table inside the
    kernel — no Python-level phys/slot pre-computation needed for attention.

    GQA, causal masking, and online softmax are handled inside the kernel.
    """

    def __init__(self, lib: ctypes.CDLL, device: str) -> None:
        self._lib    = lib
        self._device = device

    def forward(
        self,
        q               : torch.Tensor,   # (total_q, n_heads, head_dim) fp16
        pool_k          : torch.Tensor,   # (N_PHYS, BS, NKV, HD) fp16
        pool_v          : torch.Tensor,
        block_table_t   : torch.Tensor,   # (n_seqs, max_blocks) int32
        context_lens_t  : torch.Tensor,   # (n_seqs,) int32
        q_lens_t        : torch.Tensor,   # (n_seqs,) int32
        q_offsets_t     : torch.Tensor,   # (n_seqs,) int32
        total_q         : int,
        n_seqs          : int,
        n_heads         : int,
        n_kv_heads      : int,
        head_dim        : int,
        block_size      : int,
    ) -> torch.Tensor:
        """
        Run unified paged attention for a mixed prefill+decode batch.

        Returns:
            attn_out: (total_q, n_heads, head_dim) fp16
        """
        attn_out = torch.empty(
            (total_q, n_heads, head_dim),
            dtype=torch.float16,
            device=self._device,
        )

        if total_q == 0 or n_seqs == 0:
            return attn_out

        max_blocks = int(block_table_t.shape[1])

        q_c  = q.contiguous().half()
        bt_c = block_table_t.contiguous().int()
        cl_c = context_lens_t.contiguous().int()
        ql_c = q_lens_t.contiguous().int()
        qo_c = q_offsets_t.contiguous().int()

        self._lib.launch_paged_attn_unified(
            _ptr(q_c),
            _ptr(pool_k),
            _ptr(pool_v),
            _ptr(bt_c),
            _ptr(cl_c),
            _ptr(ql_c),
            _ptr(qo_c),
            _ptr(attn_out),
            ctypes.c_int(total_q),
            ctypes.c_int(n_seqs),
            ctypes.c_int(n_heads),
            ctypes.c_int(n_kv_heads),
            ctypes.c_int(head_dim),
            ctypes.c_int(block_size),
            ctypes.c_int(max_blocks),
        )
        return attn_out


# ── Singleton loader ──────────────────────────────────────────────────────────

_ops: Optional[UnifiedAttnKernelOps] = None


def load_unified_attn_kernels(
    device: Optional[str] = None,
    verbose: bool = True,
) -> UnifiedAttnKernelOps:
    """
    Compile (if needed) and load the unified paged attention kernel.
    Returns a UnifiedAttnKernelOps singleton (per process).
    """
    global _ops
    if _ops is not None:
        return _ops

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Unified attention kernel requires a GPU."
        )

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    if _so_is_stale():
        _compile(verbose=verbose)

    lib  = _load_lib()
    _ops = UnifiedAttnKernelOps(lib, device)

    if verbose:
        print(f"[unified_attn] Loaded on {device}")

    return _ops

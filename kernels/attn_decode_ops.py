"""
kernels/attn_decode_ops.py
──────────────────────────
Compiles attn_decode.cu and exposes a Python wrapper for the single-sequence
paged attention CUDA kernel used during autoregressive generation
(generate_gpu.py — one request at a time).

The kernel reads K and V DIRECTLY from the paged pool during attention
computation — no separate kv_read gather step is needed before SDPA.
(phys_blocks, slots) index arrays tell each thread where its KV token lives.

  paged_attn_kernel
    Computes:  softmax(Q @ K^T / sqrt(d)) @ V
    K/V source: fetched inside the kernel from the paged pool
    Causal mask: token t masked when t > start_pos + q_tok
    GQA:        kv_head = q_head / (n_heads / n_kv_heads)
    Numerics:   online softmax (flash-attention style), fp32 accumulation

    Grid  = (T_q, N_HEADS)    — one CUDA block per (query_token, head)
    Block = (HEAD_DIM,)       — one thread per output dimension
    Smem  = 2 × HEAD_DIM × 4 bytes

This kernel handles ONLY ONE sequence at a time.  For multi-sequence
continuous batching use attn_continuous_ops (attn_continuous.cu) instead.

Public API:
    PagedAttnKernelOps.paged_attention_cuda(
        q, pool_k, pool_v, phys_blocks, slots,
        T_q, T_total, start_pos,
        n_heads, n_kv_heads, head_dim, block_size, scale
    ) -> (T_q, N_HEADS, HEAD_DIM) fp16

    load_paged_attn_kernels(device, verbose) -> PagedAttnKernelOps  # singleton

Compilation:
    nvcc compiles attn_decode.cu → /tmp/pagedinfer_attn_decode.so on first use.

Usage:
    from kernels.attn_decode_ops import load_paged_attn_kernels
    ops = load_paged_attn_kernels()
    out = ops.paged_attention_cuda(q, pool_k, pool_v, phys_t, slot_t, ...)
"""

from __future__ import annotations

import ctypes
import math
import subprocess
from pathlib import Path
from typing import List, Optional

import torch

_KERNEL_SRC = Path(__file__).parent / "attn_decode.cu"
_SO_PATH    = Path("/tmp/pagedinfer_attn_decode.so")


def _so_is_stale() -> bool:
    if not _SO_PATH.exists():
        return True
    return _SO_PATH.stat().st_mtime < _KERNEL_SRC.stat().st_mtime


def _compile(verbose: bool = True) -> None:
    """Compile attn_decode.cu → /tmp/pagedinfer_attn_decode.so via nvcc."""
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
        print(f"[paged_attn] Compiling for SM {major}.{minor}  →  {_SO_PATH}")
        print(f"[paged_attn] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr}")

    if verbose:
        sz = _SO_PATH.stat().st_size // 1024
        print(f"[paged_attn] Compiled OK ({sz} KB)")


def _load_lib() -> ctypes.CDLL:
    """Load the .so and declare argtypes for launch_paged_attn."""
    lib = ctypes.CDLL(str(_SO_PATH), mode=ctypes.RTLD_GLOBAL)

    # launch_paged_attn(
    #   q*, pool_k*, pool_v*, phys_blocks*, slots*, out*,
    #   T_q, T_total, start_pos,
    #   n_heads, n_kv_heads, head_dim, block_size,
    #   scale
    # )
    lib.launch_paged_attn.argtypes = [
        ctypes.c_void_p,   # q
        ctypes.c_void_p,   # pool_k
        ctypes.c_void_p,   # pool_v
        ctypes.c_void_p,   # phys_blocks (int32*)
        ctypes.c_void_p,   # slots       (int32*)
        ctypes.c_void_p,   # out
        ctypes.c_int,      # T_q
        ctypes.c_int,      # T_total
        ctypes.c_int,      # start_pos
        ctypes.c_int,      # n_heads
        ctypes.c_int,      # n_kv_heads
        ctypes.c_int,      # head_dim
        ctypes.c_int,      # block_size
        ctypes.c_float,    # scale
    ]
    lib.launch_paged_attn.restype = None
    return lib


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    assert t.is_contiguous(), "tensor must be contiguous before CUDA kernel call"
    return ctypes.c_void_p(t.data_ptr())


class PagedAttnKernelOps:
    """
    Python wrapper around the paged_attn_kernel CUDA kernel.

    Computes paged attention entirely on GPU — no Python-level KV gather.
    K and V are fetched directly from the paged pool inside the kernel
    using (phys_blocks, slots) index arrays.

    GQA:       kv_head = q_head / (n_heads / n_kv_heads)
    Causal:    token t is masked if t > start_pos + q_tok
    Numerics:  online softmax (flash-attention style) — fp32 accumulation
    """

    def __init__(self, lib: ctypes.CDLL, device: str) -> None:
        self._lib    = lib
        self._device = device

    def paged_attention_cuda(
        self,
        q           : torch.Tensor,   # (T_q, N_HEADS, HEAD_DIM) fp16
        pool_k      : torch.Tensor,   # (N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM) fp16
        pool_v      : torch.Tensor,   # same
        phys_blocks : torch.Tensor,   # (T_total,) int32  — physical block per KV pos
        slots       : torch.Tensor,   # (T_total,) int32  — slot offset per KV pos
        T_q         : int,
        T_total     : int,
        start_pos   : int,
        n_heads     : int,
        n_kv_heads  : int,
        head_dim    : int,
        block_size  : int,
        scale       : float,
    ) -> torch.Tensor:
        """
        Run the paged attention kernel.

        Returns:
            out: (T_q, N_HEADS, HEAD_DIM) fp16
        """
        out = torch.empty(
            (T_q, n_heads, head_dim),
            dtype=torch.float16,
            device=self._device,
        )

        if T_q == 0 or T_total == 0:
            return out

        q_c  = q.contiguous().half()
        pb   = phys_blocks.contiguous().int()
        sl   = slots.contiguous().int()

        self._lib.launch_paged_attn(
            _ptr(q_c),
            _ptr(pool_k),
            _ptr(pool_v),
            _ptr(pb),
            _ptr(sl),
            _ptr(out),
            ctypes.c_int(T_q),
            ctypes.c_int(T_total),
            ctypes.c_int(start_pos),
            ctypes.c_int(n_heads),
            ctypes.c_int(n_kv_heads),
            ctypes.c_int(head_dim),
            ctypes.c_int(block_size),
            ctypes.c_float(scale),
        )
        return out


# ── Singleton loader ──────────────────────────────────────────────────────────

_ops: Optional[PagedAttnKernelOps] = None


def load_paged_attn_kernels(
    device: Optional[str] = None,
    verbose: bool = True,
) -> PagedAttnKernelOps:
    """
    Compile (if needed) and load the paged attention CUDA kernel.

    Compilation is skipped if the .so exists and is newer than the .cu source.
    Returns a PagedAttnKernelOps instance (singleton per process).

    Args:
        device  : CUDA device string, e.g. "cuda" or "cuda:0".
        verbose : Print compilation progress.

    Raises:
        RuntimeError: if CUDA is not available or nvcc is not found.
    """
    global _ops
    if _ops is not None:
        return _ops

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Paged attention CUDA kernel requires a GPU."
        )

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    if _so_is_stale():
        _compile(verbose=verbose)

    lib  = _load_lib()
    _ops = PagedAttnKernelOps(lib, device)

    if verbose:
        print(f"[paged_attn] Loaded on {device}")

    return _ops

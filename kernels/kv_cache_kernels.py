"""
kernels/kv_cache_kernels.py — compile kv_cache_kernels.cu and expose Python wrappers.

Only runs on CUDA. Import guard at the top of this file prevents errors on CPU machines.

Public API (only callable after load_kv_kernels()):
    init_blocks_cuda(pool_k, pool_v, block_ids)
    write_kv_cuda(pool_k, pool_v, k_in, v_in, phys_blocks, slots)
    read_kv_cuda(pool_k, pool_v, phys_blocks, slots) -> (k_out, v_out)
    clear_blocks_cuda(pool_k, pool_v, block_ids)    # alias for init (zero-fill)

Compilation:
    nvcc is invoked once per process (or when the .so is missing / stale).
    The compiled .so is written to /tmp/pagedinfer_kv_ops.so.
    SM arch is detected automatically from the current GPU.

Usage:
    from kernels.kv_cache_kernels import load_kv_kernels
    ops = load_kv_kernels()          # compiles if needed, returns KVKernelOps
    ops.write_kv_cuda(pool_k, pool_v, k, v, phys_t, slot_t)
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Optional

import torch

_KERNEL_SRC = Path(__file__).parent / "kv_cache_kernels.cu"
_SO_PATH    = Path("/tmp/pagedinfer_kv_ops.so")


def _so_is_stale() -> bool:
    """Return True if .so doesn't exist or is older than the .cu source."""
    if not _SO_PATH.exists():
        return True
    return _SO_PATH.stat().st_mtime < _KERNEL_SRC.stat().st_mtime


def _compile(verbose: bool = True) -> None:
    """Compile kv_cache_kernels.cu → /tmp/pagedinfer_kv_ops.so via nvcc."""
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
        print(f"[kv_kernels] Compiling for SM {major}.{minor}  →  {_SO_PATH}")
        print(f"[kv_kernels] {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc compilation failed:\n{result.stderr}"
        )
    if verbose:
        sz = _SO_PATH.stat().st_size // 1024
        print(f"[kv_kernels] Compiled OK ({sz} KB)")


def _load_lib() -> ctypes.CDLL:
    """Load the .so and declare all argtypes."""
    lib = ctypes.CDLL(str(_SO_PATH), mode=ctypes.RTLD_GLOBAL)

    # launch_init_blocks(pool_k, pool_v, block_ids*, n_new, block_size, n_kv_heads, head_dim)
    lib.launch_init_blocks.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,  # pool_k, pool_v
        ctypes.c_void_p,                    # block_ids (int32* on GPU)
        ctypes.c_int,                       # n_new
        ctypes.c_int,                       # block_size
        ctypes.c_int,                       # n_kv_heads
        ctypes.c_int,                       # head_dim
    ]
    lib.launch_init_blocks.restype = None

    # launch_write(pool_k, pool_v, k_in, v_in, phys_blocks*, slots*, T, nkv, hd, bs)
    lib.launch_write.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,  # pool_k, pool_v
        ctypes.c_void_p, ctypes.c_void_p,  # k_in, v_in
        ctypes.c_void_p, ctypes.c_void_p,  # phys_blocks, slots
        ctypes.c_int,                       # T
        ctypes.c_int,                       # n_kv_heads
        ctypes.c_int,                       # head_dim
        ctypes.c_int,                       # block_size
    ]
    lib.launch_write.restype = None

    # launch_read(pool_k, pool_v, k_out, v_out, phys_blocks*, slots*, T, nkv, hd, bs)
    lib.launch_read.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,  # pool_k, pool_v
        ctypes.c_void_p, ctypes.c_void_p,  # k_out, v_out
        ctypes.c_void_p, ctypes.c_void_p,  # phys_blocks, slots
        ctypes.c_int,                       # T
        ctypes.c_int,                       # n_kv_heads
        ctypes.c_int,                       # head_dim
        ctypes.c_int,                       # block_size
    ]
    lib.launch_read.restype = None

    return lib


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    """Raw GPU device pointer of a contiguous CUDA tensor."""
    assert t.is_contiguous(), "tensor must be contiguous before passing to CUDA kernel"
    return ctypes.c_void_p(t.data_ptr())


class KVKernelOps:
    """
    Thin Python wrappers around the three CUDA kernels.

    All tensor arguments must be on the same CUDA device.
    pool_k / pool_v must be fp16 and contiguous.
    phys_blocks / slots must be int32 and contiguous.
    """

    def __init__(self, lib: ctypes.CDLL, device: str) -> None:
        self._lib    = lib
        self._device = device

    def init_blocks_cuda(
        self,
        pool_k    : torch.Tensor,
        pool_v    : torch.Tensor,
        block_ids : List[int],
    ) -> None:
        """
        Zero-fill the given physical blocks in pool_k and pool_v.
        Called when LayeredBlockTable allocates a new logical block.

        Kernel: kv_init_blocks
          grid  = (n_new, BLOCK_SIZE, N_KV_HEADS)
          block = (HEAD_DIM,)
        """
        if not block_ids:
            return
        ids_t = torch.tensor(block_ids, dtype=torch.int32, device=self._device).contiguous()
        self._lib.launch_init_blocks(
            _ptr(pool_k), _ptr(pool_v),
            _ptr(ids_t),
            ctypes.c_int(len(block_ids)),
            ctypes.c_int(int(pool_k.shape[1])),   # BLOCK_SIZE
            ctypes.c_int(int(pool_k.shape[2])),   # N_KV_HEADS
            ctypes.c_int(int(pool_k.shape[3])),   # HEAD_DIM
        )

    def write_kv_cuda(
        self,
        pool_k      : torch.Tensor,            # (N_PHYS, BS, NKV, HD) fp16
        pool_v      : torch.Tensor,
        k_in        : torch.Tensor,            # (T, NKV, HD) fp16
        v_in        : torch.Tensor,
        phys_blocks : torch.Tensor,            # (T,) int32
        slots       : torch.Tensor,            # (T,) int32
    ) -> None:
        """
        Scatter K/V tokens into paged pool slots.

        Kernel: kv_write
          grid  = (T, N_KV_HEADS)
          block = (HEAD_DIM,)
        """
        T = int(k_in.shape[0])
        if T == 0:
            return
        k_c  = k_in.contiguous().half()
        v_c  = v_in.contiguous().half()
        pb   = phys_blocks.contiguous().int()
        sl   = slots.contiguous().int()
        self._lib.launch_write(
            _ptr(pool_k), _ptr(pool_v),
            _ptr(k_c),    _ptr(v_c),
            _ptr(pb),     _ptr(sl),
            ctypes.c_int(T),
            ctypes.c_int(int(pool_k.shape[2])),   # N_KV_HEADS
            ctypes.c_int(int(pool_k.shape[3])),   # HEAD_DIM
            ctypes.c_int(int(pool_k.shape[1])),   # BLOCK_SIZE
        )

    def read_kv_cuda(
        self,
        pool_k      : torch.Tensor,
        pool_v      : torch.Tensor,
        phys_blocks : torch.Tensor,   # (T,) int32
        slots       : torch.Tensor,   # (T,) int32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K/V from paged pool into fresh contiguous buffers.

        Returns:
            k_out, v_out : (T, N_KV_HEADS, HEAD_DIM) fp16

        Kernel: kv_read
          grid  = (T, N_KV_HEADS)
          block = (HEAD_DIM,)
        """
        T   = int(phys_blocks.shape[0])
        NKV = int(pool_k.shape[2])
        HD  = int(pool_k.shape[3])
        k_out = torch.empty((T, NKV, HD), dtype=torch.float16, device=self._device)
        v_out = torch.empty_like(k_out)
        if T == 0:
            return k_out, v_out
        pb  = phys_blocks.contiguous().int()
        sl  = slots.contiguous().int()
        self._lib.launch_read(
            _ptr(pool_k), _ptr(pool_v),
            _ptr(k_out),  _ptr(v_out),
            _ptr(pb),     _ptr(sl),
            ctypes.c_int(T),
            ctypes.c_int(NKV),
            ctypes.c_int(HD),
            ctypes.c_int(int(pool_k.shape[1])),   # BLOCK_SIZE
        )
        return k_out, v_out

    def clear_blocks_cuda(
        self,
        pool_k    : torch.Tensor,
        pool_v    : torch.Tensor,
        block_ids : List[int],
    ) -> None:
        """Zero-fill freed blocks — prevents stale KV from leaking. Reuses init kernel."""
        self.init_blocks_cuda(pool_k, pool_v, block_ids)


# ── Singleton loader ──────────────────────────────────────────────────────────

_ops: Optional[KVKernelOps] = None


def load_kv_kernels(device: Optional[str] = None, verbose: bool = True) -> KVKernelOps:
    """
    Compile (if needed) and load the KV cache CUDA kernels.

    Compilation is skipped if the .so exists and is newer than the .cu source.
    Returns a KVKernelOps instance (singleton per process).

    Args:
        device  : CUDA device string, e.g. "cuda" or "cuda:0".
                  Defaults to the current default CUDA device.
        verbose : Print compilation progress.

    Raises:
        RuntimeError: if CUDA is not available or nvcc is not found.
    """
    global _ops
    if _ops is not None:
        return _ops

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. KV cache CUDA kernels require a GPU."
        )

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    if _so_is_stale():
        _compile(verbose=verbose)

    lib  = _load_lib()
    _ops = KVKernelOps(lib, device)

    if verbose:
        print(f"[kv_kernels] Loaded on {device}")

    return _ops

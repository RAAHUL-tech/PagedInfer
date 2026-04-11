"""
GPUPagedKVCache — contiguous GPU tensor pool for paged KV cache.

Drop-in replacement for PagedKVCache when running on CUDA.

Storage layout:
    pool_k : (N_PHYSICAL_BLOCKS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM)  fp16
    pool_v : same

Compared to PagedKVCache (dict of tensors, one tensor per block):
  ┌─────────────────┬──────────────────────────────────────────────────┐
  │ PagedKVCache    │ dict per layer, on-demand allocation              │
  │                 │ flexible but fragmented, slow for large pools     │
  ├─────────────────┼──────────────────────────────────────────────────┤
  │ GPUPagedKVCache │ single cudaMalloc for K and V (contiguous)       │
  │                 │ CUDA kernels for read / write / init / clear      │
  │                 │ cache-friendly layout, better GPU utilisation     │
  └─────────────────┴──────────────────────────────────────────────────┘

Key design:
  - pool_k and pool_v are pre-allocated once; no further GPU allocation
    during generation.
  - Block init (zero-fill) and read/write go through CUDA kernels
    (KVKernelOps from kernels/kv_cache_kernels.py).
  - Paged attention can be computed by the CUDA kernel
    (PagedAttnKernelOps from kernels/paged_attention_kernels.py),
    which reads K/V directly from the pool without a separate gather step.
  - The pool is shared across all transformer layers (LayeredBlockTable
    guarantees each physical block belongs to exactly one layer).
  - API is compatible with PagedKVCache:
      allocate_block_for_layer(layer, block_id)
      free_blocks(block_ids)
      write_tokens(layer, block_table, k_seq, v_seq, start_pos)
      read_sequence(layer, block_table) -> (k, v)
      compute_paged_attn(q, layer, block_table, ...) -> out   [GPU-only]

Typical creation:
    cache = GPUPagedKVCache.from_memory_budget(
        free_gpu_bytes = 8 * 1024**3,   # 8 GB KV budget
        n_layers       = cfg.n_layers,
        block_size     = 16,
        n_kv_heads     = cfg.n_kv_heads,
        head_dim       = cfg.head_dim,
        device         = "cuda",
    )
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from kernels.kv_io_ops import KVKernelOps, load_kv_kernels
from kernels.attn_decode_ops import PagedAttnKernelOps, load_paged_attn_kernels

if TYPE_CHECKING:
    from .block_table import LayeredBlockTable


class GPUPagedKVCache:
    """
    Paged KV cache backed by a single contiguous GPU allocation per K and V.

    All I/O routes through CUDA kernels:
      - allocate_block_for_layer → kv_init_blocks kernel (zero-fill)
      - free_blocks              → kv_init_blocks kernel (clear on free)
      - write_tokens             → kv_write kernel (scatter)
      - read_sequence            → kv_read kernel  (gather)
      - compute_paged_attn       → paged_attn kernel (attention in-place)

    Args:
        pool_k         : Pre-allocated (N_PHYS, BS, NKV, HD) fp16 CUDA tensor.
        pool_v         : Same shape as pool_k.
        n_layers       : Number of transformer layers (for bookkeeping only).
        kv_ops         : Compiled KV read/write/init kernel wrappers.
        attn_ops       : Compiled paged attention kernel wrapper.
        device         : CUDA device string.
    """

    def __init__(
        self,
        pool_k  : torch.Tensor,
        pool_v  : torch.Tensor,
        n_layers: int,
        kv_ops  : KVKernelOps,
        attn_ops: PagedAttnKernelOps,
        device  : str,
    ) -> None:
        assert pool_k.is_cuda and pool_v.is_cuda, "pools must be on CUDA"
        assert pool_k.is_contiguous() and pool_v.is_contiguous()
        assert pool_k.dtype == torch.float16 and pool_v.dtype == torch.float16
        assert pool_k.shape == pool_v.shape and pool_k.ndim == 4

        self.pool_k   = pool_k
        self.pool_v   = pool_v
        self.n_layers = n_layers
        self._kv_ops  = kv_ops
        self._attn_ops = attn_ops
        self.device   = device

        self.n_phys     = int(pool_k.shape[0])
        self.block_size = int(pool_k.shape[1])
        self.n_kv_heads = int(pool_k.shape[2])
        self.head_dim   = int(pool_k.shape[3])

        # block_id → owning layer (mirrors PagedKVCache._id_to_layer)
        self._id_to_layer: Dict[int, int] = {}

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_pool_size(
        cls,
        n_physical_blocks : int,
        n_layers          : int,
        block_size        : int,
        n_kv_heads        : int,
        head_dim          : int,
        device            : str = "cuda",
        verbose           : bool = True,
    ) -> "GPUPagedKVCache":
        """
        Allocate a contiguous GPU pool of the given size and return a
        GPUPagedKVCache instance backed by it.
        """
        shape  = (n_physical_blocks, block_size, n_kv_heads, head_dim)
        pool_k = torch.zeros(shape, dtype=torch.float16, device=device)
        pool_v = torch.zeros(shape, dtype=torch.float16, device=device)

        kv_ops   = load_kv_kernels(device=device, verbose=verbose)
        attn_ops = load_paged_attn_kernels(device=device, verbose=verbose)

        cache = cls(pool_k, pool_v, n_layers, kv_ops, attn_ops, device)

        if verbose:
            mb = pool_k.numel() * 2 * 2 / 1e6   # K+V, fp16
            print(f"[GPUPagedKVCache] pool {list(shape)} × K+V = {mb:.1f} MB")
            print(f"[GPUPagedKVCache] {n_physical_blocks} physical blocks "
                  f"({n_physical_blocks // max(n_layers, 1)} logical blocks)")

        return cache

    @classmethod
    def from_memory_budget(
        cls,
        free_gpu_bytes : int,
        n_layers       : int,
        block_size     : int,
        n_kv_heads     : int,
        head_dim       : int,
        device         : str = "cuda",
        verbose        : bool = True,
    ) -> "GPUPagedKVCache":
        """
        Allocate as many physical blocks as fit in free_gpu_bytes.

        One physical block = block_size × n_kv_heads × head_dim × 2 bytes (fp16).
        Both K and V have this shape, so one logical block pair = 2 × block bytes.
        Each logical block needs n_layers physical ids, so:

            bytes_per_phys_block = block_size × n_kv_heads × head_dim × 2
            n_physical_blocks    = free_gpu_bytes // (2 × bytes_per_phys_block)
        """
        bytes_per_phys = block_size * n_kv_heads * head_dim * 2   # fp16
        n_phys = free_gpu_bytes // (2 * bytes_per_phys)           # K + V

        if n_phys < n_layers:
            raise RuntimeError(
                f"GPU memory budget too small: only {n_phys} physical blocks fit, "
                f"need at least {n_layers} (one logical block)."
            )

        return cls.from_pool_size(
            n_physical_blocks=n_phys,
            n_layers=n_layers,
            block_size=block_size,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            device=device,
            verbose=verbose,
        )

    # ── Allocation / deallocation ─────────────────────────────────────────────

    def allocate_block_for_layer(self, layer: int, block_id: int) -> None:
        """
        Called by LayeredBlockTable.append_token() when a new logical block
        is needed.  Zero-fills the physical block in the pool via the CUDA
        init kernel and records the layer ownership for O(1) free.
        """
        if block_id not in self._id_to_layer:
            self._kv_ops.init_blocks_cuda(self.pool_k, self.pool_v, [block_id])
            self._id_to_layer[block_id] = layer

    def free_blocks(self, block_ids: List[int]) -> None:
        """
        Zero-fill freed blocks (prevents stale KV from leaking to future
        sequences) and remove them from the ownership map.
        """
        valid = [bid for bid in block_ids if bid in self._id_to_layer]
        if valid:
            self._kv_ops.clear_blocks_cuda(self.pool_k, self.pool_v, valid)
            for bid in valid:
                self._id_to_layer.pop(bid)

    # ── Write ─────────────────────────────────────────────────────────────────

    def batch_write_kv(
        self,
        k_all   : torch.Tensor,   # (total_tokens, n_kv_heads, head_dim) fp16
        v_all   : torch.Tensor,
        phys_t  : torch.Tensor,   # (total_tokens,) int32 — pre-built physical ids
        slots_t : torch.Tensor,   # (total_tokens,) int32 — pre-built slot offsets
    ) -> None:
        """
        Scatter K/V for multiple sequences in ONE CUDA kernel launch.

        Used by ContinuousBatchingEngine.forward_unified() which pre-builds
        the phys/slots tensors for all active sequences at once, then calls
        this method to write all tokens with a single launch instead of one
        per sequence per layer.

        Args:
            k_all   : Concatenated K across all seqs, (total_tokens, NKV, HD)
            v_all   : Concatenated V across all seqs, (total_tokens, NKV, HD)
            phys_t  : Physical block id for each token
            slots_t : Slot offset within block for each token
        """
        self._kv_ops.write_kv_cuda(
            self.pool_k, self.pool_v,
            k_all.half().contiguous(),
            v_all.half().contiguous(),
            phys_t.contiguous().int(),
            slots_t.contiguous().int(),
        )

    def write_tokens(
        self,
        layer       : int,
        block_table : "LayeredBlockTable",
        k_seq       : torch.Tensor,   # (T, n_kv_heads, head_dim) fp16
        v_seq       : torch.Tensor,
        start_pos   : int = 0,
    ) -> None:
        """
        Scatter T token keys and values into the paged pool via CUDA kv_write.

        phys_blocks and slots are built from block_table.translate() calls
        and passed as GPU int32 tensors to the kernel.
        """
        T = k_seq.shape[0]
        if T == 0:
            return

        phys_list: List[int] = []
        slot_list: List[int] = []
        for i in range(T):
            pb, s = block_table.translate(layer, start_pos + i)
            phys_list.append(pb)
            slot_list.append(s)

        phys_t = torch.tensor(phys_list, dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_list, dtype=torch.int32, device=self.device)

        self._kv_ops.write_kv_cuda(
            self.pool_k, self.pool_v,
            k_seq.half().contiguous(),
            v_seq.half().contiguous(),
            phys_t, slot_t,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def read_sequence(
        self,
        layer       : int,
        block_table : "LayeredBlockTable",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gather K and V for all tokens in the sequence via CUDA kv_read.

        Returns:
            k_gathered : (T, n_kv_heads, head_dim) fp16
            v_gathered : (T, n_kv_heads, head_dim) fp16
        """
        phys_list, slot_list = block_table.get_all_physical_slots(layer)
        phys_t = torch.tensor(phys_list, dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_list, dtype=torch.int32, device=self.device)
        return self._kv_ops.read_kv_cuda(self.pool_k, self.pool_v, phys_t, slot_t)

    # ── Paged attention (CUDA kernel) ─────────────────────────────────────────

    def compute_paged_attn(
        self,
        q           : torch.Tensor,   # (1, n_heads, T_q, head_dim) fp16
        layer       : int,
        block_table : "LayeredBlockTable",
        n_heads     : int,
        n_kv_heads  : int,
        head_dim    : int,
        start_pos   : int,
    ) -> torch.Tensor:
        """
        Compute paged attention entirely on GPU via the paged_attn kernel.

        K and V are accessed directly from the pool inside the CUDA kernel
        using (phys_blocks, slots) — no separate kv_read call needed.

        Args:
            q         : (1, n_heads, T_q, head_dim) fp16 — query tensor
            layer     : transformer layer index (selects which slots to read)
            block_table: LayeredBlockTable for the current sequence
            n_heads   : number of attention heads (Q)
            n_kv_heads: number of KV heads (GQA: <= n_heads)
            head_dim  : dimension per head
            start_pos : absolute position of q[0] in the sequence

        Returns:
            out: (1, T_q, n_heads * head_dim) fp16 — ready for output projection
        """
        # q: (1, n_heads, T_q, head_dim) → (T_q, n_heads, head_dim)
        T_q   = int(q.shape[2])
        q_2d  = q.squeeze(0).permute(1, 0, 2).contiguous().half()
        # q_2d: (T_q, n_heads, head_dim)

        # Build index arrays for all KV positions seen so far
        phys_list, slot_list = block_table.get_all_physical_slots(layer)
        T_total = len(phys_list)

        phys_t = torch.tensor(phys_list, dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_list, dtype=torch.int32, device=self.device)

        scale = 1.0 / math.sqrt(head_dim)

        # out: (T_q, n_heads, head_dim) fp16
        out = self._attn_ops.paged_attention_cuda(
            q          = q_2d,
            pool_k     = self.pool_k,
            pool_v     = self.pool_v,
            phys_blocks= phys_t,
            slots      = slot_t,
            T_q        = T_q,
            T_total    = T_total,
            start_pos  = start_pos,
            n_heads    = n_heads,
            n_kv_heads = n_kv_heads,
            head_dim   = head_dim,
            block_size = self.block_size,
            scale      = scale,
        )   # (T_q, n_heads, head_dim)

        # Reshape to (1, T_q, n_heads * head_dim) for output projection
        return out.unsqueeze(0).reshape(1, T_q, n_heads * head_dim)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def allocated_blocks(self) -> int:
        """Number of physical blocks currently in use."""
        return len(self._id_to_layer)

    def memory_pool_mb(self) -> float:
        """Total GPU memory reserved by pool_k + pool_v (always fully allocated)."""
        return self.pool_k.numel() * 2 * 2 / 1e6   # K+V, fp16=2 bytes

    def memory_used_mb(self) -> float:
        """
        Logical memory in use: only blocks with live sequences.
        Internal fragmentation (last partial block) is included.
        """
        used = self.allocated_blocks()
        return used * self.block_size * self.n_kv_heads * self.head_dim * 2 * 2 / 1e6

    def __repr__(self) -> str:
        return (
            f"GPUPagedKVCache("
            f"phys_blocks={self.n_phys}, "
            f"used={self.allocated_blocks()}, "
            f"pool={self.memory_pool_mb():.1f} MB, "
            f"used={self.memory_used_mb():.1f} MB)"
        )

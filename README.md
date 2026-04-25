# PagedInfer

A ground-up LLM inference engine in Python and CUDA — paged KV cache, continuous batching, prefix caching, and an HTTP inference server.

---

## What it is

PagedInfer rebuilds the core of a production inference system from scratch. No vLLM, no TRT-LLM — every subsystem is implemented directly: GPU memory management, CUDA attention kernels, the scheduler, and the serving layer. The notebooks document each concept as it is built; the Python packages are the production-ready versions.

---

## Architecture

```
Client (HTTP / SSE)
        │
        ▼
server/api_server.py        FastAPI — POST /generate, POST /generate/stream
        │
        ▼
server/engine_server.py     EngineWorker (background thread) +
                            StreamingPrefixEngine (per-token callbacks)
        │
        ▼
engine/prefix_engine.py     PrefixAwareEngine — scheduler loop
  ├── engine/scheduler.py       decides prefill / decode / preempt each step
  ├── engine/forward_pass.py    one batched forward pass for all active seqs
  └── engine/prefix_sequence.py per-request state machine
        │
        ▼
kv_cache/                   GPU memory management
  ├── gpu_paged_kv_cache.py     contiguous fp16 pool, CUDA kernel I/O
  ├── block_allocator.py        free-list with reference counting
  ├── block_table.py            logical → physical block mapping (per layer)
  ├── prefix_cache.py           hash-based block store, LRU eviction
  └── prefix_block_table.py     block table with cached-prefix attachment
        │
        ▼
kernels/                    Custom CUDA kernels (compiled at first use)
  ├── kv_io.cu                  block init, K/V scatter-write, K/V gather-read
  ├── attn_decode.cu            fused paged attention — single sequence
  └── attn_continuous.cu        fused paged attention — all sequences in one launch
        │
        ▼
models/                     LLaMA transformer
  ├── transformer.py            full model with configurable attention type
  ├── attention.py              PagedAttention — dispatches to GPU or CPU backend
  ├── rope.py                   RoPE with pre-computed cos/sin tables
  ├── ffn.py                    SwiGLU feed-forward
  └── norm.py                   RMSNorm
```

---

## Key subsystems

### Paged KV cache

The KV cache is a single pre-allocated `(N_PHYS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM)` fp16 tensor on the GPU. Sequences do not get contiguous slices — they get a list of physical block IDs (one per layer), and two CUDA kernels scatter writes and gather reads from those blocks. This eliminates fragmentation and lets the pool be shared across arbitrarily many sequences.

`BlockAllocator` manages the free-list with reference counting. Reference counts > 1 indicate a block shared between a live sequence and the prefix cache.

### Continuous batching

Every engine iteration the `Scheduler` decides:
1. **Decode** — all currently running sequences get one token slot (highest priority)
2. **Restore** — preempted sequences with CPU-saved KV are written back (no re-prefill)
3. **Admit** — new sequences are admitted for prefill up to `max_prefill_seqs`

If the pool is exhausted, the decode sequence with the fewest generated tokens is preempted: its K/V tensors are copied to pinned CPU RAM and its GPU blocks are freed. On restore, the tensors are written back via the kv_write kernel — no forward pass needed.

`forward_unified` runs **one batched forward pass** per iteration covering all prefill and decode sequences simultaneously. QKV projection, RoPE, and the FFN each execute once regardless of batch size. The `attn_continuous.cu` kernel handles causal masking for prefill tokens and full-context attention for decode tokens in the same launch.

### Prefix caching

When sequences share a common prefix (e.g. a system prompt), `PrefixCache` maps a hash chain over the token blocks to physical block IDs. On admission, `match()` finds the longest cached prefix and pins those blocks by incrementing their ref counts. The sequence's `prefill_offset` jumps past the cached tokens — the forward pass only runs on the suffix.

After each prefill completes, `insert_sequence_blocks()` stores the new blocks so future requests benefit. LRU eviction discards the least-recently-used block when the cache is full, skipping any block still held by a live sequence.

### HTTP inference server

`api_server.py` is a FastAPI server. The engine runs in a background daemon thread. When a streaming request arrives, the HTTP handler registers an `asyncio.Queue` for that sequence; `_on_token()` (called by the engine after each sampled token) pushes a `TokenEvent` to that queue via `call_soon_threadsafe`. The handler drains the queue and sends SSE events to the client. Non-streaming requests drain the same queue and return the assembled text as JSON.

---

## File map

```
models/          LLaMA backbone (transformer, attention, RoPE, FFN, norm)
kv_cache/        Paged KV pool, block allocator, prefix cache
kernels/         CUDA kernels + Python wrappers (compiled on first use)
engine/          Scheduler, sequence state machine, continuous batching engine,
                 prefix-caching engine
server/          FastAPI HTTP server, engine worker, SSE streaming, test client
inference/       Standalone generation scripts (no-cache → flat KV → paged CPU
                 → paged GPU → prefix-cached → continuous-batching server)
notebooks/       Progressive build-up: 01 transformer → 02 KV cache →
                 03 GPU paged attention → 04 continuous batching →
                 05 prefix caching
```

---

## Inference scripts

| Script | What it tests |
|--------|---------------|
| `inference/generate_no_cache.py` | Baseline — full recompute every step |
| `inference/generate_flat_kv.py` | Flat KV cache, O(n) decoding |
| `inference/generate_paged_cpu.py` | CPU-backed paged KV cache |
| `inference/generate_paged_gpu.py` | GPU pool + CUDA kernels |
| `inference/generate_prefix_cached.py` | Prefix cache hit rate, TTFT comparison, LRU eviction |
| `inference/serve_continuous.py` | Multi-request continuous batching, speedup vs static |

---

## References

- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

---

## License

MIT

# PagedInfer

> A high-performance LLM inference engine built from scratch, exploring every layer of the modern inference stack, from GPU memory management to multi-node parallelism.

---

## What is PagedInfer?

PagedInfer is a ground-up implementation of a production-grade LLM inference engine. It is not a wrapper around existing frameworks — every subsystem is designed and built from first principles, with the goal of deeply understanding the performance characteristics, tradeoffs, and engineering decisions that underpin systems like vLLM, TensorRT-LLM, and Megatron-LM.

The project covers the full vertical slice: from how tokens are scheduled and batched, to how GPU memory is managed at the block level, to how custom CUDA kernels reduce memory bandwidth and fuse operations for maximum throughput.

---

## Core Subsystems

### Execution Engine

The heart of PagedInfer is an autoregressive decoding loop built for real-world serving conditions. It supports streaming token generation, pluggable sampling strategies (greedy, top-k, top-p, temperature, repetition penalty), and a model runner abstraction that decouples weight loading from execution logic.

### KV Cache & Memory Management

Efficient inference depends almost entirely on how key-value cache memory is managed. PagedInfer implements:

- **Flat KV cache** — baseline O(n) decoding by reusing past attention states across generation steps
- **Paged KV cache** — block-based memory allocation inspired by virtual memory in operating systems; sequences are broken into fixed-size pages, eliminating fragmentation and enabling fine-grained memory reuse
- **Prefix caching** — shared prompt prefixes (system prompts, few-shot examples) are hashed and cached, so repeated prefixes are computed once and reused across requests
- **Memory pooling** — pre-allocated memory pools avoid costly runtime allocations and prevent GPU memory fragmentation under high concurrency

### Continuous Batching & Scheduling

Unlike static batching (where all sequences in a batch must finish before a new one starts), PagedInfer implements **continuous batching**: requests are merged and split at the token level, so the GPU is never idle waiting for long sequences to finish. The scheduler handles:

- Dynamic request merging and preemption
- Token-level scheduling across heterogeneous sequence lengths
- Request prioritization, cancellation, and backpressure handling

### Speculative Decoding

A draft model generates multiple candidate tokens per step; a larger verifier model checks them in parallel. Accepted tokens are committed; rejected ones fall back to standard sampling. This reduces the number of sequential forward passes required per output token, lowering latency without changing output distribution.

### Custom CUDA Kernels

PyTorch's built-in ops carry significant overhead for inference-critical paths. PagedInfer replaces them with hand-written CUDA kernels:

- **Fused QKV attention** — query, key, value projection and softmax computed in a single kernel pass, reducing memory round-trips
- **RMSNorm / LayerNorm** — fused normalization kernels that avoid intermediate tensor materialization
- **KV cache update** — custom paged memory write kernel that handles block-level cache population efficiently
- **Kernel fusion** — eliminates intermediate DRAM writes between adjacent ops; critical for memory-bandwidth-bound workloads

### Model Parallelism

To support models that exceed single-GPU memory:

- **Tensor parallelism** — weight matrices are sharded column- or row-wise across devices; each GPU holds a slice of every layer and results are reduced via all-reduce (NCCL)
- **Pipeline parallelism** — transformer layers are distributed across GPUs as pipeline stages; micro-batching keeps all stages busy, amortizing inter-GPU communication latency

---

## Architecture

```
┌─────────────────────────────────────┐
│            REST / gRPC API          │
│     Streaming · Async · Multi-req   │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│             Scheduler               │
│   Continuous batching · Priority    │
│   Preemption · Backpressure         │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│          Execution Engine           │
│   Autoregressive loop · Sampling    │
│   Speculative decoding              │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│       KV Cache Manager (Paged)      │
│   Block allocator · Prefix cache    │
│   Memory pool · Eviction policy     │
└──────────────────┬──────────────────┘
                   ↓
┌─────────────────────────────────────┐
│    CUDA Kernels / GPU Runtime       │
│  Fused attention · RMSNorm          │
│  Cache writes · Tensor parallelism  │
└─────────────────────────────────────┘
```

---

## Key Design Decisions

| Concern | Approach |
|---|---|
| Memory efficiency | Paged KV cache + memory pooling |
| Throughput | Continuous batching + token-level scheduling |
| Latency | Speculative decoding + kernel fusion |
| Scalability | Tensor + pipeline parallelism via NCCL |
| Compute efficiency | Custom CUDA kernels, fused ops |
| Prompt reuse | Prefix caching with hash-based deduplication |

---

## Techniques Implemented

- Autoregressive decoding with KV cache (O(n) per step)
- Paged memory management for KV cache (vLLM-style)
- Prefix / prompt caching
- Continuous batching (Orca-style)
- Speculative decoding (draft + verifier)
- Top-k / top-p / greedy / temperature sampling
- Fused QKV attention CUDA kernel
- RMSNorm / LayerNorm CUDA kernels
- Paged KV cache write kernel
- Tensor parallelism (Megatron-LM style)
- Pipeline parallelism with micro-batching
- NCCL-based inter-GPU communication
- Streaming token generation
- Async REST and gRPC serving

---

## Goals

This project is an exercise in rebuilding rather than reusing. The goal is to understand every tradeoff in the inference stack — why paging beats contiguous KV allocation, why continuous batching doubles throughput, why speculative decoding cuts latency — by implementing each idea from scratch and measuring the results.

---

## Getting Started

```bash
pip install -r requirements.txt
python main.py
```

For GPU testing, clone into a Colab or Kaggle notebook, enable GPU runtime, and run the benchmarks and kernel tests in `notebooks/`.

---

## References & Inspiration

- [vLLM: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [Speculative Decoding](https://arxiv.org/abs/2302.01318)

---

## License

MIT
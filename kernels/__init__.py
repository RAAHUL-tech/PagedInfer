"""
kernels/ — CUDA kernels for PagedInfer.

Current kernels:
    kv_cache_kernels.cu  — paged KV cache write, read, and block init
    kv_cache_kernels.py  — compile + ctypes wrappers, exposes KVKernelOps

Usage:
    from kernels.kv_cache_kernels import load_kv_kernels
    ops = load_kv_kernels()   # compiles on first call, cached thereafter
"""

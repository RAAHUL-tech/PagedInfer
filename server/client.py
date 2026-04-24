"""
server/client.py — command-line test client for the PagedInfer inference server.

Supports both streaming (SSE) and non-streaming modes, plus a load test that
fires multiple requests concurrently to demonstrate continuous batching.

Usage
─────
    # Non-streaming single request
    python server/client.py --prompt "Once upon a time"

    # Streaming single request (prints tokens as they arrive)
    python server/client.py --prompt "Tell me about black holes" --stream

    # Send all built-in prompts concurrently (load test)
    python server/client.py --load_test

    # Custom server URL
    python server/client.py --url http://34.123.45.67:8000 --prompt "Hello"

    # Health check + metrics
    python server/client.py --health
    python server/client.py --metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

try:
    import requests
except ImportError:
    print("requests not installed — run: pip install requests")
    sys.exit(1)


# ── Default load-test prompts ──────────────────────────────────────────────────

_LOAD_TEST_PROMPTS: List[Tuple[str, int]] = [
    ("Once upon a time there was a little dragon who",               64),
    ("The key insight of the transformer architecture is",            64),
    ("Explain the difference between supervised and unsupervised",    64),
    ("In the year 2075, humanity had finally",                        64),
    ("The recipe for the perfect sourdough bread requires",           64),
    ("Write a haiku about the autumn moon:",                          32),
    ("To debug a segmentation fault in C, you should",                64),
    ("The attention mechanism works by computing",                     64),
]


# ── Single non-streaming request ───────────────────────────────────────────────

def generate(
    url            : str,
    prompt         : str,
    max_new_tokens : int   = 128,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 1.0,
    timeout        : int   = 120,
) -> dict:
    """Send a POST /generate request and return the JSON response."""
    resp = requests.post(
        f"{url}/generate",
        json    = {
            "prompt"         : prompt,
            "max_new_tokens" : max_new_tokens,
            "temperature"    : temperature,
            "top_k"          : top_k,
            "top_p"          : top_p,
        },
        timeout = timeout,
    )
    resp.raise_for_status()
    return resp.json()


# ── Streaming request ──────────────────────────────────────────────────────────

def generate_stream(
    url            : str,
    prompt         : str,
    max_new_tokens : int   = 128,
    temperature    : float = 0.8,
    top_k          : int   = 50,
    top_p          : float = 1.0,
    timeout        : int   = 120,
) -> None:
    """Send a POST /generate/stream request and print tokens as they arrive."""
    print(f"  Prompt : {prompt!r}")
    print(f"  Output : ", end="", flush=True)

    t0 = time.perf_counter()
    first_token_t = None

    with requests.post(
        f"{url}/generate/stream",
        json    = {
            "prompt"         : prompt,
            "max_new_tokens" : max_new_tokens,
            "temperature"    : temperature,
            "top_k"          : top_k,
            "top_p"          : top_p,
        },
        stream  = True,
        timeout = timeout,
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())

            if payload.get("done"):
                elapsed = (time.perf_counter() - t0) * 1000
                print()   # newline after streamed tokens
                print(f"\n  [done]  gen={payload['tokens_generated']} tokens  "
                      f"TTFT={payload['ttft_ms']:.1f}ms  "
                      f"E2E={payload['e2e_ms']:.1f}ms  "
                      f"cached={payload['cached_tokens']} tokens  "
                      f"wall={elapsed:.0f}ms")
                break
            else:
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                print(payload.get("token", ""), end="", flush=True)


# ── Load test ──────────────────────────────────────────────────────────────────

def load_test(
    url      : str,
    prompts  : List[Tuple[str, int]],
    n_threads: int = 8,
) -> None:
    """
    Fire all prompts concurrently using a thread pool.

    Each thread sends a non-streaming request and waits for the result.
    This exercises the engine's continuous batching — multiple requests
    are in-flight simultaneously and the engine interleaves their prefill
    and decode steps.
    """
    print(f"\n  Sending {len(prompts)} concurrent requests to {url}...\n")
    results = {}
    t_start = time.perf_counter()

    def _worker(idx: int, prompt: str, max_new: int) -> Tuple[int, dict]:
        t0  = time.perf_counter()
        res = generate(url, prompt, max_new_tokens=max_new)
        res["wall_ms"] = (time.perf_counter() - t0) * 1000
        return idx, res

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = {
            pool.submit(_worker, i, p, n): i
            for i, (p, n) in enumerate(prompts)
        }
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res

    elapsed = time.perf_counter() - t_start
    total_toks = sum(r["tokens_generated"] for r in results.values())

    print(f"  {'Req':>4}  {'TTFT':>8}  {'E2E':>8}  {'gen':>5}  {'cached':>7}  Prompt")
    print(f"  {'───':>4}  {'────':>8}  {'───':>8}  {'───':>5}  {'──────':>7}  ──────")
    for i in sorted(results):
        r = results[i]
        print(f"  {i:4d}  {r['ttft_ms']:7.1f}ms  {r['e2e_ms']:7.1f}ms  "
              f"{r['tokens_generated']:5d}  {r['cached_tokens']:7d}  "
              f"{prompts[i][0][:55]!r}")

    print(f"\n  Total: {total_toks} tokens in {elapsed:.2f}s  "
          f"→  {total_toks / elapsed:.1f} tok/s  (wall clock, concurrent)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PagedInfer inference server client")
    p.add_argument("--url",            default="http://localhost:8000")
    p.add_argument("--prompt",         default=None)
    p.add_argument("--max_new_tokens", type=int,   default=128)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--stream",         action="store_true",
                   help="Use SSE streaming mode")
    p.add_argument("--load_test",      action="store_true",
                   help="Fire all built-in prompts concurrently")
    p.add_argument("--health",         action="store_true",
                   help="Print /health response and exit")
    p.add_argument("--metrics",        action="store_true",
                   help="Print /metrics response and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    url  = args.url.rstrip("/")

    if args.health:
        r = requests.get(f"{url}/health", timeout=5)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))
        return

    if args.metrics:
        r = requests.get(f"{url}/metrics", timeout=5)
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))
        return

    if args.load_test:
        load_test(url, _LOAD_TEST_PROMPTS)
        return

    prompt = args.prompt
    if not prompt:
        print("Provide --prompt, --load_test, --health, or --metrics.")
        sys.exit(1)

    if args.stream:
        generate_stream(
            url,
            prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
        )
    else:
        t0  = time.perf_counter()
        res = generate(
            url,
            prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
        )
        wall = (time.perf_counter() - t0) * 1000
        print(f"  Prompt : {prompt!r}")
        print(f"  Output : {res['text']!r}")
        print(f"  gen={res['tokens_generated']} tokens  "
              f"TTFT={res['ttft_ms']:.1f}ms  E2E={res['e2e_ms']:.1f}ms  "
              f"cached={res['cached_tokens']} tokens  wall={wall:.0f}ms")


if __name__ == "__main__":
    main()

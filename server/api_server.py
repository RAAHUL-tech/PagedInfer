"""
server/api_server.py — FastAPI HTTP inference server.

Endpoints
─────────
    POST /generate
        Non-streaming inference.  Waits for the full response and returns it
        as JSON.  Use when the caller needs the complete text in one shot.

    POST /generate/stream
        Streaming inference via Server-Sent Events (SSE).  Each token is sent
        as it is generated.  The final event carries full latency metrics.
        Use for interactive UIs or low-latency first-token display.

    GET  /health
        Returns {"status": "ok"} plus engine liveness info.  Suitable for
        load-balancer health checks.

    GET  /metrics
        Engine throughput, TTFT, and prefix-cache statistics.

Usage
─────
    # Start with default settings
    uv run python server/api_server.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt

    # Custom GPU budget and cache fraction
    uv run python server/api_server.py \\
        --checkpoint model_checkpoint/llama_ckpt.pt \\
        --kv_budget_gb 4.0 \\
        --cache_fraction 0.5 \\
        --port 8000

    # Then hit it from server/client.py or curl:
    curl -X POST http://localhost:8000/generate \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "Once upon a time", "max_new_tokens": 64}'

Wire-up
───────
    Startup:  build PrefixAwareEngine → wrap in StreamingPrefixEngine
              → start EngineWorker thread → store in app.state
    Shutdown: call worker.stop()
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.prefix_engine import PrefixAwareEngine
from engine.request import Request as EngineRequest
from inference.model_loader import load_tokenizer
from server.engine_server import EngineWorker, StreamingPrefixEngine


# ── Request / response schemas ─────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt         : str
    max_new_tokens : int   = Field(default=128,  ge=1,   le=2048)
    temperature    : float = Field(default=0.8,  ge=0.0, le=2.0)
    top_k          : int   = Field(default=50,   ge=0)
    top_p          : float = Field(default=1.0,  ge=0.0, le=1.0)
    priority       : int   = Field(default=0,    ge=0)


class GenerateResponse(BaseModel):
    request_id       : int
    text             : str
    tokens_generated : int
    ttft_ms          : float
    e2e_ms           : float
    cached_tokens    : int


# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    worker    : EngineWorker
    tokenizer : object
    req_counter : int = 0


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the engine, start the worker thread, yield, then stop cleanly."""
    args = app.state.cli_args

    print("[server] Loading tokenizer...")
    tokenizer = load_tokenizer(args.tokenizer)

    print("[server] Building PrefixAwareEngine...")
    base_engine = PrefixAwareEngine.build(
        checkpoint     = args.checkpoint,
        eos_token_id   = tokenizer.eos_token_id,
        device         = args.device,
        block_size     = args.block_size,
        kv_budget_gb   = args.kv_budget_gb,
        cache_fraction = args.cache_fraction,
        max_batch      = args.max_batch,
        max_prefill    = args.max_prefill,
        chunk_size     = args.chunk_size,
        log_every      = args.log_every,
        verbose        = True,
    )

    streaming_engine = StreamingPrefixEngine(base_engine, tokenizer)
    worker           = EngineWorker(streaming_engine)

    state             = AppState()
    state.worker      = worker
    state.tokenizer   = tokenizer
    app.state.server  = state

    loop = asyncio.get_event_loop()
    worker.start(loop)
    print("[server] Engine worker started. Ready to serve requests.")

    yield   # server is running

    print("[server] Shutting down engine worker...")
    worker.stop()


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "PagedInfer Inference Server",
    description = "Continuous batching LLM inference with prefix KV caching",
    version     = "1.0.0",
    lifespan    = lifespan,
)


def _next_request_id(state: AppState) -> int:
    state.req_counter += 1
    return state.req_counter


# ── POST /generate ─────────────────────────────────────────────────────────────

@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest) -> GenerateResponse:
    """
    Non-streaming generation.

    Waits until the full sequence is generated and returns it as JSON.
    Suitable for batch workloads where first-token latency is not critical.
    """
    state     = app.state.server
    worker    = state.worker
    tokenizer = state.tokenizer
    req_id    = _next_request_id(state)

    token_ids = tokenizer.encode(body.prompt, add_special_tokens=True)
    if not token_ids:
        raise HTTPException(status_code=400, detail="Prompt encodes to empty token list.")

    engine_req = EngineRequest(
        request_id     = req_id,
        prompt         = body.prompt,
        max_new_tokens = body.max_new_tokens,
        temperature    = body.temperature,
        top_k          = body.top_k,
        top_p          = body.top_p,
        priority       = body.priority,
        arrival_time   = time.time(),
    )

    seq_id, q = worker.submit(engine_req, token_ids)

    # Drain the token queue until the final event
    collected_ids : list[int] = []
    ttft_ms = e2e_ms = 0.0

    while True:
        event = await q.get()
        collected_ids.append(event.token_id)
        if event.is_final:
            ttft_ms = event.ttft_ms
            e2e_ms  = event.e2e_ms
            break

    text = tokenizer.decode(collected_ids, skip_special_tokens=True)

    # Look up cached_tokens from finished seq
    cached = 0
    for seq in worker.engine._engine.finished:
        if seq.seq_id == seq_id:
            cached = seq.n_cached_tokens
            break

    return GenerateResponse(
        request_id       = req_id,
        text             = text,
        tokens_generated = len(collected_ids),
        ttft_ms          = ttft_ms,
        e2e_ms           = e2e_ms,
        cached_tokens    = cached,
    )


# ── POST /generate/stream ──────────────────────────────────────────────────────

@app.post("/generate/stream")
async def generate_stream(body: GenerateRequest) -> StreamingResponse:
    """
    Streaming generation via Server-Sent Events.

    Each token is sent as an SSE event as soon as it is sampled by the engine.
    The stream ends with a final event that includes latency metrics.

    Event format:
        data: {"token": "hello", "tokens_generated": 1}
        data: {"token": " world", "tokens_generated": 2}
        data: {"done": true, "tokens_generated": 5, "ttft_ms": 42.1, "e2e_ms": 310.5, "cached_tokens": 32}
    """
    state     = app.state.server
    worker    = state.worker
    tokenizer = state.tokenizer
    req_id    = _next_request_id(state)

    token_ids = tokenizer.encode(body.prompt, add_special_tokens=True)
    if not token_ids:
        raise HTTPException(status_code=400, detail="Prompt encodes to empty token list.")

    engine_req = EngineRequest(
        request_id     = req_id,
        prompt         = body.prompt,
        max_new_tokens = body.max_new_tokens,
        temperature    = body.temperature,
        top_k          = body.top_k,
        top_p          = body.top_p,
        priority       = body.priority,
        arrival_time   = time.time(),
    )

    seq_id, q = worker.submit(engine_req, token_ids)

    async def event_generator() -> AsyncIterator[str]:
        cached = 0
        try:
            while True:
                event = await q.get()

                if event.is_final:
                    # Look up cached_tokens
                    for seq in worker.engine._engine.finished:
                        if seq.seq_id == seq_id:
                            cached = seq.n_cached_tokens
                            break
                    payload = json.dumps({
                        "done"             : True,
                        "tokens_generated" : event.tokens_generated,
                        "ttft_ms"          : round(event.ttft_ms, 2),
                        "e2e_ms"           : round(event.e2e_ms, 2),
                        "cached_tokens"    : cached,
                    })
                    yield f"data: {payload}\n\n"
                    break
                else:
                    payload = json.dumps({
                        "token"            : event.token_text,
                        "tokens_generated" : event.tokens_generated,
                    })
                    yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            worker.engine.unregister(seq_id)

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


# ── GET /health ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Liveness check — returns engine status and queue depth."""
    state  = app.state.server
    engine = state.worker.engine
    return {
        "status"        : "ok",
        "requests_done" : engine.metrics.total_requests_done,
        "throughput"    : round(engine.metrics.throughput_tok_s, 1),
        "pool_used"     : engine.allocator.n_used,
        "pool_free"     : engine.allocator.n_free,
    }


# ── GET /metrics ───────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics() -> dict:
    """Full engine and prefix-cache statistics."""
    state  = app.state.server
    engine = state.worker.engine
    m      = engine.metrics
    pc     = engine.prefix_cache
    alloc  = engine.allocator
    return {
        "engine": {
            "iterations"        : m.iteration,
            "total_tokens_gen"  : m.total_tokens_gen,
            "total_requests"    : m.total_requests_done,
            "throughput_tok_s"  : round(m.throughput_tok_s, 2),
            "avg_ttft_ms"       : round(m.avg_ttft_ms, 2),
            "avg_e2e_ms"        : round(m.avg_e2e_ms, 2),
            "preemptions"       : m.n_preemptions,
            "swap_ins"          : m.n_swap_ins,
            "chunked_steps"     : m.n_chunked_steps,
            "fully_cached"      : m.n_fully_cached,
        },
        "prefix_cache": {
            "cached_blocks" : pc.n_cached,
            "hits"          : pc.n_hits,
            "misses"        : pc.n_misses,
            "hit_rate"      : round(pc.hit_rate, 4),
            "evicted"       : pc.n_evicted,
        },
        "memory_pool": {
            "n_total" : alloc.n_total,
            "n_used"  : alloc.n_used,
            "n_free"  : alloc.n_free,
            "util"    : round(alloc.utilization(), 4),
        },
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PagedInfer HTTP inference server")
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--tokenizer",      default="huggyllama/llama-7b")
    p.add_argument("--host",           default="0.0.0.0")
    p.add_argument("--port",           type=int,   default=8000)
    p.add_argument("--device",         default="cuda")
    p.add_argument("--block_size",     type=int,   default=16)
    p.add_argument("--kv_budget_gb",   type=float, default=2.0)
    p.add_argument("--cache_fraction", type=float, default=0.5)
    p.add_argument("--max_batch",      type=int,   default=8)
    p.add_argument("--max_prefill",    type=int,   default=4)
    p.add_argument("--chunk_size",     type=int,   default=32)
    p.add_argument("--log_every",      type=int,   default=20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.state.cli_args = args
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

"""
server/engine_server.py — streaming engine + background worker thread.

Two classes:

StreamingPrefixEngine
─────────────────────
Extends PrefixAwareEngine by overriding _on_token().  Each time the engine
samples a token it calls _on_token(seq, token_id, is_final).  This subclass
maintains a dict of per-request queues and pushes a TokenEvent into the
appropriate queue — thread-safely, via asyncio's call_soon_threadsafe().

The HTTP handler awaits tokens from its asyncio.Queue and streams them to the
client as Server-Sent Events.  No polling: the engine pushes, the handler pulls.

EngineWorker
────────────
Owns the engine and runs its step() loop in a daemon thread.

    start()  — launches the background thread
    stop()   — signals the thread to exit cleanly
    submit() — called from async HTTP handlers; adds a request to the engine
               and returns (seq_id, asyncio.Queue) for token streaming.

Thread model:
    HTTP handler (async, event loop thread)
        → worker.submit(request, token_ids)       [acquires lock, calls engine.add_request]
        → awaits asyncio.Queue for TokenEvents    [blocks handler coroutine, not the thread]

    Engine thread (daemon)
        → calls engine.step() in a tight loop
        → _on_token() fires for each token
        → loop.call_soon_threadsafe(queue.put_nowait, event)  [wakes the handler]
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.prefix_sequence import PrefixSequenceGroup


# ── Token event ────────────────────────────────────────────────────────────────

@dataclass
class TokenEvent:
    """One token pushed from the engine thread to an HTTP handler."""
    token_id         : int
    token_text       : str
    is_final         : bool
    tokens_generated : int   = 0
    ttft_ms          : float = 0.0
    e2e_ms           : float = 0.0


# ── Streaming engine ───────────────────────────────────────────────────────────

class StreamingPrefixEngine:
    """
    PrefixAwareEngine subclass that pushes tokens to per-request asyncio queues.

    The asyncio event loop reference is injected by EngineWorker.start() so
    call_soon_threadsafe() can be called safely from the engine thread.
    """

    def __init__(self, base_engine, tokenizer):
        # Monkey-patch _on_token onto the already-constructed engine instance
        self._engine    = base_engine
        self._tokenizer = tokenizer
        self._queues    : Dict[int, asyncio.Queue] = {}   # seq_id → asyncio.Queue
        self._lock      = threading.Lock()
        self._loop      : Optional[asyncio.AbstractEventLoop] = None

        # Patch the hook onto the engine instance (no subclassing needed)
        engine = base_engine
        server = self

        def _on_token(
            _self,
            seq      : "PrefixSequenceGroup",
            token_id : int,
            is_final : bool,
        ) -> None:
            with server._lock:
                q = server._queues.get(seq.seq_id)
            if q is None or server._loop is None:
                return

            try:
                text = server._tokenizer.decode(
                    [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
            except Exception:
                text = ""

            event = TokenEvent(
                token_id         = token_id,
                token_text       = text,
                is_final         = is_final,
                tokens_generated = seq.n_generated,
                ttft_ms          = seq.ttft_ms or 0.0,
                e2e_ms           = seq.e2e_latency_ms or 0.0,
            )
            server._loop.call_soon_threadsafe(q.put_nowait, event)
            if is_final:
                with server._lock:
                    server._queues.pop(seq.seq_id, None)

        # Bind as instance method
        import types
        engine._on_token = types.MethodType(_on_token, engine)

    def register(self, seq_id: int, q: asyncio.Queue) -> None:
        with self._lock:
            self._queues[seq_id] = q

    def unregister(self, seq_id: int) -> None:
        with self._lock:
            self._queues.pop(seq_id, None)

    # Proxy the engine's public API
    def add_request(self, request, token_ids: List[int]) -> int:
        return self._engine.add_request(request, token_ids)

    def has_work(self) -> bool:
        return self._engine.has_work()

    def step(self):
        return self._engine.step()

    @property
    def metrics(self):
        return self._engine.metrics

    @property
    def prefix_cache(self):
        return self._engine.prefix_cache

    @property
    def allocator(self):
        return self._engine.allocator


# ── Engine worker ──────────────────────────────────────────────────────────────

class EngineWorker:
    """
    Runs the inference engine in a background daemon thread.

    The engine thread calls step() in a tight loop when work is available,
    or sleeps 1 ms when the queue is empty to avoid busy-waiting.

    All calls that touch the engine (add_request) are serialised via a lock
    so HTTP handlers from multiple asyncio tasks can submit safely.
    """

    IDLE_SLEEP_S = 0.001   # 1 ms sleep when engine has no work

    def __init__(self, engine: StreamingPrefixEngine) -> None:
        self.engine   = engine
        self._lock    = threading.Lock()
        self._stop    = threading.Event()
        self._thread  = threading.Thread(
            target   = self._run,
            name     = "engine-worker",
            daemon   = True,
        )
        self._loop    : Optional[asyncio.AbstractEventLoop] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Start the engine thread.

        Args:
            loop : The asyncio event loop used by the HTTP server.
                   Stored so _on_token can call call_soon_threadsafe().
        """
        self._loop                = loop
        self.engine._loop         = loop   # inject into streaming engine
        self._thread.start()

    def stop(self) -> None:
        """Signal the engine thread to stop and wait for it to exit."""
        self._stop.set()
        self._thread.join(timeout=5.0)

    # ── Request submission ─────────────────────────────────────────────────────

    def submit(
        self,
        request,
        token_ids : List[int],
    ) -> Tuple[int, asyncio.Queue]:
        """
        Add a request to the engine and return (seq_id, token_queue).

        The asyncio.Queue will receive TokenEvent objects as the engine
        generates tokens.  The final token has is_final=True.

        Thread-safe: can be called from any asyncio task.
        """
        assert self._loop is not None, "call start() before submit()"
        q      = asyncio.Queue()
        with self._lock:
            seq_id = self.engine.add_request(request, token_ids)
            self.engine.register(seq_id, q)
        return seq_id, q

    # ── Engine loop ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop.is_set():
            # Hold the lock for the full check-and-step so that submit()
            # cannot interleave between has_work() returning False and the
            # thread sleeping — that would lose a just-added request for one
            # extra idle sleep (harmless but wasteful).
            with self._lock:
                if self.engine.has_work():
                    self.engine.step()
                    continue
            time.sleep(self.IDLE_SLEEP_S)

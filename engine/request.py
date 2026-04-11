"""
engine/request.py — Request dataclass and RequestQueue.

A Request is the unit of work entering the engine.  It carries the prompt
text, generation parameters, and metadata for priority scheduling.

RequestQueue wraps a min-heap so the engine can pop requests in
(priority, arrival_time) order — FIFO within the same priority level.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Request:
    """
    One inference request entering the continuous batching engine.

    Args:
        request_id     : Unique integer identifier.
        prompt         : Raw input text.
        max_new_tokens : Maximum number of tokens to generate.
        temperature    : Sampling temperature (lower = more deterministic).
        top_k          : Top-k truncation (0 = disabled).
        top_p          : Nucleus sampling threshold (1.0 = disabled).
        priority       : Scheduling priority — lower number = higher priority.
        arrival_time   : Wall-clock time the request arrived.
    """

    request_id     : int
    prompt         : str
    max_new_tokens : int   = 128
    temperature    : float = 0.8
    top_k          : int   = 50
    top_p          : float = 1.0
    priority       : int   = 0
    arrival_time   : float = field(default_factory=time.time)

    def __lt__(self, other: "Request") -> bool:
        # Secondary sort key for heapq tie-breaking (never actually compared
        # as Request objects — the heap stores (priority, counter, request))
        return self.arrival_time < other.arrival_time


class RequestQueue:
    """
    Priority queue of incoming requests.

    Requests are ordered by (priority, arrival_time) — FIFO within the
    same priority level.  Backed by a min-heap.

    Usage:
        q = RequestQueue()
        q.add(Request(0, "hello"))
        req = q.pop()     # highest priority request
    """

    def __init__(self) -> None:
        self._heap    : List[Tuple] = []
        self._counter : int         = 0   # tie-break FIFO within same priority

    def add(self, request: Request) -> None:
        """Push a request into the queue."""
        heapq.heappush(
            self._heap,
            (request.priority, self._counter, request)
        )
        self._counter += 1

    def pop(self) -> Optional[Request]:
        """Pop and return the highest-priority request, or None if empty."""
        if self._heap:
            return heapq.heappop(self._heap)[2]
        return None

    def peek_priority(self) -> Optional[int]:
        """Return the priority of the next request without removing it."""
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __repr__(self) -> str:
        return f"RequestQueue(size={len(self)})"

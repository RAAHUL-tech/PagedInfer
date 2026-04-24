"""
server/ — HTTP inference server for PagedInfer.

    engine_server.py   StreamingPrefixEngine  — extends PrefixAwareEngine with
                           per-token callbacks so HTTP clients receive tokens as
                           they are generated rather than waiting for completion.
                       EngineWorker — runs the engine step() loop in a background
                           thread and bridges it to asyncio-based HTTP handlers.

    api_server.py      FastAPI application with four endpoints:
                           POST /generate          non-streaming, returns full text
                           POST /generate/stream   streaming via Server-Sent Events
                           GET  /health            liveness check
                           GET  /metrics           engine throughput + cache stats

    client.py          Command-line test client for both streaming and
                       non-streaming modes.
"""

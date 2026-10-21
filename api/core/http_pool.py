"""Shared process-lifespan HTTP/2 connection pool & asyncio event-loop stall monitor."""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger("nimagents.core.http_pool")

_shared_client: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.AsyncClient:
    """Return a single process-wide HTTP/2 `httpx.AsyncClient` (`max_connections=1000`)."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0, connect=8.0),
            limits=httpx.Limits(
                max_connections=1000,
                max_keepalive_connections=200,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_client


async def close_shared_http_client() -> None:
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None


class EventLoopWatchdog:
    """Continuously measures asyncio event-loop scheduling jitter to detect blocking I/O stalls."""

    def __init__(self, interval_s: float = 0.05, stall_threshold_ms: float = 100.0) -> None:
        self.interval_s = interval_s
        self.stall_threshold_ms = stall_threshold_ms
        self._samples_ms: deque[float] = deque(maxlen=500)
        self.max_lag_ms: float = 0.0
        self.stall_events: int = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._task.get_loop() is not loop:
            self._task = asyncio.create_task(self._loop(), name="nimagents-loop-watchdog")

    async def stop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is not None and not self._task.done() and self._task.get_loop() is loop:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(self.interval_s)
            elapsed_ms = (time.perf_counter() - t0 - self.interval_s) * 1000.0
            lag_ms = max(0.0, elapsed_ms)
            self._samples_ms.append(lag_ms)
            if lag_ms > self.max_lag_ms:
                self.max_lag_ms = lag_ms
            if lag_ms >= self.stall_threshold_ms:
                self.stall_events += 1
                logger.warning("Detected asyncio event-loop stall: %.1f ms", lag_ms)

    def snapshot(self) -> dict[str, Any]:
        samples = sorted(self._samples_ms)
        if not samples:
            return {
                "samples": 0,
                "p50_lag_ms": 0.0,
                "p95_lag_ms": 0.0,
                "max_lag_ms": round(self.max_lag_ms, 2),
                "stall_events": self.stall_events,
            }
        p50 = samples[int(len(samples) * 0.50)]
        p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
        return {
            "samples": len(samples),
            "p50_lag_ms": round(p50, 2),
            "p95_lag_ms": round(p95, 2),
            "max_lag_ms": round(self.max_lag_ms, 2),
            "stall_events": self.stall_events,
        }


loop_watchdog = EventLoopWatchdog()

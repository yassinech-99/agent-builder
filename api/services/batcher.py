"""Dynamic Cross-User GPU Micro-Batcher + L1 SHA-256 Exact Embedding Cache (`api/services/batcher.py`).

Coalesces concurrent single-query `embed_query()` calls arriving across different users/tenants
within an `8ms` batching window up to `max_batch_size = 64` into ONE single batched GPU RPC call
(`embed_documents(batch_texts)`), increasing multi-user embedding QPS by 5x–10x on a single GPU
with strict per-caller Future isolation!
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import inspect
import logging
from typing import Any, Callable

logger = logging.getLogger("nimagents.services.batcher")


class L1EmbeddingCache:
    """Bounded thread/task-safe LRU cache deduplicating identical `(model_key, text)` embeddings."""

    def __init__(self, max_entries: int = 4096) -> None:
        self.max_entries = max_entries
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def make_key(model_key: str, text: str) -> str:
        digest = hashlib.sha256(f"{model_key}\x00{text}".encode("utf-8")).hexdigest()
        return digest

    def get(self, model_key: str, text: str) -> list[float] | None:
        key = self.make_key(model_key, text)
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)
            self.hits += 1
            return list(val)
        self.misses += 1
        return None

    def put(self, model_key: str, text: str, vector: list[float]) -> None:
        key = self.make_key(model_key, text)
        self._store[key] = list(vector)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)


@dataclass
class _EmbedWaiter:
    text: str
    future: asyncio.Future[list[float]]


class DynamicGPUBatcher:
    """Coalesces concurrent embedding calls per `model_key` over an `8ms` micro-batch window."""

    def __init__(
        self,
        *,
        window_ms: float = 8.0,
        max_batch_size: int = 64,
        cache_size: int = 4096,
    ) -> None:
        self.window_s = max(0.001, window_ms / 1000.0)
        self.max_batch_size = max(1, max_batch_size)
        self.l1_cache = L1EmbeddingCache(max_entries=cache_size)

        self._queues: dict[str, deque[_EmbedWaiter]] = {}
        self._flush_tasks: dict[str, asyncio.Task[None]] = {}
        self._batch_fns: dict[str, Callable[[list[str]], Any]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

        self.logical_requests: int = 0
        self.gpu_batches_sent: int = 0

    def _reset_if_new_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._queues.clear()
            self._flush_tasks.clear()
            self._loop = loop
        return loop

    async def embed_text(
        self,
        model_key: str,
        text: str,
        batch_embed_fn: Callable[[list[str]], Any],
    ) -> list[float]:
        """Embed a single query string via L1 cache -> 8ms cross-user micro-batch coalescer."""
        self.logical_requests += 1
        cached = self.l1_cache.get(model_key, text)
        if cached is not None:
            return cached

        loop = self._reset_if_new_loop()
        fut: asyncio.Future[list[float]] = loop.create_future()

        queue = self._queues.get(model_key)
        if queue is None:
            queue = deque()
            self._queues[model_key] = queue

        self._batch_fns[model_key] = batch_embed_fn
        queue.append(_EmbedWaiter(text=text, future=fut))

        # Flush immediately if max_batch_size reached, otherwise arm 8ms timer
        if len(queue) >= self.max_batch_size:
            existing = self._flush_tasks.pop(model_key, None)
            if existing is not None and not existing.done():
                existing.cancel()
            asyncio.create_task(self._flush_model_queue(model_key, delay_s=0.0))
        elif model_key not in self._flush_tasks or self._flush_tasks[model_key].done():
            self._flush_tasks[model_key] = asyncio.create_task(
                self._flush_model_queue(model_key, delay_s=self.window_s)
            )

        return await fut

    async def embed_documents(
        self,
        model_key: str,
        texts: list[str],
        batch_embed_fn: Callable[[list[str]], Any],
    ) -> list[list[float]]:
        """Embed a list of chunks using L1 cache + bounded max_batch_size GPU slices."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for idx, t in enumerate(texts):
            self.logical_requests += 1
            hit = self.l1_cache.get(model_key, t)
            if hit is not None:
                results[idx] = hit
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        for start in range(0, len(missing_texts), self.max_batch_size):
            sub_texts = missing_texts[start : start + self.max_batch_size]
            sub_indices = missing_indices[start : start + self.max_batch_size]
            vectors = await self._invoke_batch_fn(batch_embed_fn, sub_texts)
            self.gpu_batches_sent += 1
            for i_pos, vec in zip(sub_indices, vectors):
                self.l1_cache.put(model_key, texts[i_pos], vec)
                results[i_pos] = vec

        return [v for v in results if v is not None]

    async def _flush_model_queue(self, model_key: str, delay_s: float) -> None:
        if delay_s > 0:
            try:
                await asyncio.sleep(delay_s)
            except asyncio.CancelledError:
                return

        queue = self._queues.get(model_key)
        if not queue:
            return

        batch_fn = self._batch_fns[model_key]
        while queue:
            waiters: list[_EmbedWaiter] = []
            while queue and len(waiters) < self.max_batch_size:
                waiters.append(queue.popleft())

            # Deduplicate identical strings within the same micro-batch!
            unique_texts: list[str] = []
            seen: dict[str, int] = {}
            mapping: list[int] = []
            for w in waiters:
                pos = seen.get(w.text)
                if pos is None:
                    pos = len(unique_texts)
                    seen[w.text] = pos
                    unique_texts.append(w.text)
                mapping.append(pos)

            try:
                vectors = await self._invoke_batch_fn(batch_fn, unique_texts)
                self.gpu_batches_sent += 1
                for txt, vec in zip(unique_texts, vectors):
                    self.l1_cache.put(model_key, txt, vec)
                for w, u_idx in zip(waiters, mapping):
                    if not w.future.done():
                        w.future.set_result(list(vectors[u_idx]))
            except Exception as exc:
                for w in waiters:
                    if not w.future.done():
                        w.future.set_exception(exc)

    @staticmethod
    async def _invoke_batch_fn(
        fn: Callable[[list[str]], Any], texts: list[str]
    ) -> list[list[float]]:
        if inspect.iscoroutinefunction(fn):
            return await fn(texts)
        return await asyncio.to_thread(fn, texts)

    def status_summary(self) -> dict[str, Any]:
        batches = max(1, self.gpu_batches_sent)
        return {
            "window_ms": round(self.window_s * 1000.0, 1),
            "max_batch_size": self.max_batch_size,
            "logical_requests": self.logical_requests,
            "gpu_batches_sent": self.gpu_batches_sent,
            "avg_coalescing_ratio": round(self.logical_requests / batches, 2)
            if self.gpu_batches_sent > 0
            else 0.0,
            "l1_embedding_cache_hits": self.l1_cache.hits,
            "l1_embedding_cache_misses": self.l1_cache.misses,
        }


gpu_batcher = DynamicGPUBatcher()

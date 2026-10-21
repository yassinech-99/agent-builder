"""Two-Lane Weighted Fair Queuing Pool (`Lane P0: interactive` vs `Lane P1: background`).

Solves head-of-line multi-tenant starvation:
  - Lane `P0` (`interactive`): Dedicated high-concurrency lane for live chat turns, HITL approvals,
    and voice requests (`<5ms` queue wait target — completely insulated from bulk ingestion!).
  - Lane `P1` (`background`): Dedicated worker pool using **Per-Tenant Round-Robin Fair Scheduling**:
    if Tenant A enqueues 50 heavy PDF indexing jobs and Tenant B enqueues 1 job right behind them,
    Tenant B is scheduled immediately on the next available worker slot instead of waiting behind
    49 of Tenant A's jobs!
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import time
from typing import Any, Awaitable, Callable
import uuid

from api.core.tenancy import (
    TenantContext,
    get_tenant_context,
    reset_tenant_context,
    set_tenant_context,
)
from api.db import get_db

logger = logging.getLogger("nimagents.workers.priority_queue")


@dataclass
class _QueuedWorkItem:
    job_id: str
    tenant_ctx: TenantContext
    lane: str  # "interactive" | "background"
    job_type: str
    coro_factory: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    enqueued_at: float
    persist: bool = False


class PriorityWorkPool:
    """Async two-lane fair scheduler with automatic `TenantContext` propagation."""

    def __init__(
        self,
        *,
        p0_concurrency: int = 16,
        p1_concurrency: int = 4,
    ) -> None:
        self.p0_concurrency = max(1, p0_concurrency)
        self.p1_concurrency = max(1, p1_concurrency)

        self._p0_queue: asyncio.Queue[_QueuedWorkItem] | None = None
        # Per-tenant round-robin queues for Lane P1 (`background`)
        self._p1_lock: asyncio.Lock | None = None
        self._p1_not_empty: asyncio.Condition | None = None
        self._p1_tenant_queues: dict[str, deque[_QueuedWorkItem]] = {}
        self._p1_tenant_ring: deque[str] = deque()

        self._workers: list[asyncio.Task[None]] = []
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None

        self.active_p0: int = 0
        self.active_p1: int = 0
        self.completed_p0: int = 0
        self.completed_p1: int = 0
        self.failed_jobs: int = 0
        self._wait_p0_ms: deque[float] = deque(maxlen=500)
        self._wait_p1_ms: deque[float] = deque(maxlen=500)

    def _ensure_started_on_current_loop(self) -> None:
        current_loop = asyncio.get_running_loop()
        if self._running and self._loop is current_loop:
            return

        # Re-bind primitives safely if called from a fresh test event loop
        self._loop = current_loop
        self._p0_queue = asyncio.Queue()
        self._p1_lock = asyncio.Lock()
        self._p1_not_empty = asyncio.Condition(self._p1_lock)
        self._p1_tenant_queues.clear()
        self._p1_tenant_ring.clear()
        self._workers.clear()
        self._running = True

        for idx in range(self.p0_concurrency):
            self._workers.append(
                asyncio.create_task(self._run_p0_worker(idx), name=f"nimagents-p0-{idx}")
            )
        for idx in range(self.p1_concurrency):
            self._workers.append(
                asyncio.create_task(self._run_p1_worker(idx), name=f"nimagents-p1-{idx}")
            )

    async def stop(self) -> None:
        self._running = False
        current_loop = asyncio.get_running_loop()
        same_loop_tasks: list[asyncio.Task[None]] = []
        for task in self._workers:
            if not task.done() and task.get_loop() is current_loop:
                task.cancel()
                same_loop_tasks.append(task)
        if same_loop_tasks:
            await asyncio.gather(*same_loop_tasks, return_exceptions=True)
        self._workers.clear()

    async def submit_interactive(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        job_type: str = "chat_turn",
        ctx: TenantContext | None = None,
    ) -> Any:
        """Execute `coro_factory` on Lane `P0` (`interactive`) and await its result."""
        fut = await self.enqueue(
            coro_factory,
            lane="interactive",
            job_type=job_type,
            ctx=ctx,
            persist=False,
        )
        return await fut

    async def submit_background(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        job_type: str = "background_task",
        ctx: TenantContext | None = None,
        persist: bool = False,
        wait: bool = True,
    ) -> Any:
        """Enqueue `coro_factory` on fair per-tenant Lane `P1` (`background`)."""
        job_id = str(uuid.uuid4())
        fut = await self.enqueue(
            coro_factory,
            lane="background",
            job_type=job_type,
            ctx=ctx,
            persist=persist,
            job_id=job_id,
        )
        if wait:
            return await fut
        return job_id

    async def enqueue(
        self,
        coro_factory: Callable[[], Awaitable[Any]],
        *,
        lane: str = "background",
        job_type: str = "task",
        ctx: TenantContext | None = None,
        persist: bool = False,
        job_id: str | None = None,
    ) -> asyncio.Future[Any]:
        self._ensure_started_on_current_loop()
        assert self._loop is not None
        active = ctx or get_tenant_context()
        jid = job_id or str(uuid.uuid4())
        fut = self._loop.create_future()

        if persist:
            await asyncio.to_thread(
                self._record_job_db,
                jid,
                active.tenant_id,
                active.workspace_id,
                lane,
                job_type,
                "queued",
                0,
                "",
            )

        item = _QueuedWorkItem(
            job_id=jid,
            tenant_ctx=active,
            lane=lane,
            job_type=job_type,
            coro_factory=coro_factory,
            future=fut,
            enqueued_at=time.perf_counter(),
            persist=persist,
        )

        if lane == "interactive":
            assert self._p0_queue is not None
            await self._p0_queue.put(item)
        else:
            assert self._p1_not_empty is not None
            async with self._p1_not_empty:
                tid = active.tenant_id
                q = self._p1_tenant_queues.get(tid)
                if q is None:
                    q = deque()
                    self._p1_tenant_queues[tid] = q
                if not q and tid not in self._p1_tenant_ring:
                    self._p1_tenant_ring.append(tid)
                q.append(item)
                self._p1_not_empty.notify(1)

        return fut

    async def _run_p0_worker(self, _worker_idx: int) -> None:
        assert self._p0_queue is not None
        while self._running:
            item = await self._p0_queue.get()
            wait_ms = (time.perf_counter() - item.enqueued_at) * 1000.0
            self._wait_p0_ms.append(wait_ms)
            self.active_p0 += 1
            try:
                await self._execute_item(item)
                self.completed_p0 += 1
            finally:
                self.active_p0 -= 1
                self._p0_queue.task_done()

    async def _pop_next_fair_p1_item(self) -> _QueuedWorkItem:
        assert self._p1_not_empty is not None
        async with self._p1_not_empty:
            while not self._p1_tenant_ring:
                await self._p1_not_empty.wait()
            # Round-robin pop: take 1 item from front tenant, rotate tenant to tail if non-empty
            tid = self._p1_tenant_ring.popleft()
            q = self._p1_tenant_queues[tid]
            item = q.popleft()
            if q:
                self._p1_tenant_ring.append(tid)
            else:
                self._p1_tenant_queues.pop(tid, None)
            return item

    async def _run_p1_worker(self, _worker_idx: int) -> None:
        while self._running:
            item = await self._pop_next_fair_p1_item()
            wait_ms = (time.perf_counter() - item.enqueued_at) * 1000.0
            self._wait_p1_ms.append(wait_ms)
            self.active_p1 += 1
            try:
                await self._execute_item(item)
                self.completed_p1 += 1
            finally:
                self.active_p1 -= 1

    async def _execute_item(self, item: _QueuedWorkItem) -> None:
        if item.persist:
            await asyncio.to_thread(
                self._update_job_db, item.job_id, "running", 25, ""
            )

        tok = set_tenant_context(item.tenant_ctx)
        try:
            res = await item.coro_factory()
            if not item.future.done():
                item.future.set_result(res)
            if item.persist:
                await asyncio.to_thread(
                    self._update_job_db, item.job_id, "completed", 100, ""
                )
        except Exception as exc:
            self.failed_jobs += 1
            if not item.future.done():
                item.future.set_exception(exc)
            if item.persist:
                await asyncio.to_thread(
                    self._update_job_db, item.job_id, "error", 100, str(exc)[:400]
                )
        finally:
            reset_tenant_context(tok)

    @staticmethod
    def _record_job_db(
        job_id: str,
        tenant_id: str,
        workspace_id: str,
        lane: str,
        job_type: str,
        status: str,
        progress: int,
        detail: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO background_jobs
                   (id, tenant_id, workspace_id, lane, job_type, status, progress, detail, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, tenant_id, workspace_id, lane, job_type, status, progress, detail, now, now),
            )
            conn.commit()

    @staticmethod
    def _update_job_db(job_id: str, status: str, progress: int, detail: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with get_db() as conn:
            conn.execute(
                "UPDATE background_jobs SET status=?, progress=?, detail=?, updated_at=? WHERE id=?",
                (status, progress, detail, now, job_id),
            )
            conn.commit()

    def status_summary(self) -> dict[str, Any]:
        p0_depth = self._p0_queue.qsize() if self._p0_queue is not None else 0
        p1_depth = sum(len(q) for q in self._p1_tenant_queues.values())
        s0 = sorted(self._wait_p0_ms)
        s1 = sorted(self._wait_p1_ms)
        return {
            "p0_interactive": {
                "workers": self.p0_concurrency,
                "active": self.active_p0,
                "queued": p0_depth,
                "completed": self.completed_p0,
                "p50_wait_ms": round(s0[len(s0) // 2], 2) if s0 else 0.0,
                "p95_wait_ms": round(s0[min(len(s0) - 1, int(len(s0) * 0.95))], 2) if s0 else 0.0,
            },
            "p1_background": {
                "workers": self.p1_concurrency,
                "active": self.active_p1,
                "queued": p1_depth,
                "queued_tenants": len(self._p1_tenant_queues),
                "completed": self.completed_p1,
                "p50_wait_ms": round(s1[len(s1) // 2], 2) if s1 else 0.0,
                "p95_wait_ms": round(s1[min(len(s1) - 1, int(len(s1) * 0.95))], 2) if s1 else 0.0,
            },
            "failed_jobs": self.failed_jobs,
        }


work_pool = PriorityWorkPool()

"""nimAgents AI Inference Gateway (`agent/inference_gateway.py`).

Production multi-user GPU optimization layer implementing Section 2.4 of `plan.md`:
  1. Load-Bounded Prefix Consistent Hashing (`PrefixAwareEndpointRouter`) across self-hosted
     NVIDIA NIM / vLLM / SGLang replicas so multi-turn agent loops hit the exact same GPU pod's
     Radix KV cache (~90% prefix cache hit rate instead of `1/N`), with automatic ring spillover
     under hot-slot contention or replica failure.
  2. Multi-Tier Tenant-Safe Caching:
     - L0: Byte-stable prefix hash computation (`compute_prefix_hash`).
     - L1: Sub-millisecond tenant-scoped exact response cache for `temperature == 0` utility calls.
     - L2: Tenant-partitioned Qdrant Semantic Cache (`TenantSemanticCache`) keyed strictly on
       `(tenant_id, workspace_id, agent_id, system_prompt_hash, kb_version_hash)` with `0.96`
       cosine threshold — zero cross-tenant bleed and instant automatic KB invalidation!
  3. Async Write-Behind Usage & FinOps Telemetry (`UsageTelemetryCollector`) flushing token/USD
     accounting in bulk batches off the critical streaming path.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import time
from typing import Any, Generator
import uuid

from qdrant_client import models as qmodels

from api.core.tenancy import TenantContext, get_tenant_context, sanitize_tenant_identifier
from api.core.vector_store import MultiTenantVectorStore, vector_store
from api.db import get_db

logger = logging.getLogger("nimagents.inference.gateway")


def compute_prefix_hash(*parts: str) -> str:
    """Compute stable 16-hex-char SHA-256 fingerprint for KV-cache routing & semantic partitions."""
    joined = "\x1f".join(p or "" for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ─── 1. Load-Bounded Consistent Prefix Hash Router ──────────────────────


class PrefixAwareEndpointRouter:
    """Consistent hash ring with active slot tracking & health circuit breakers for NIM clusters."""

    def __init__(
        self,
        *,
        virtual_nodes: int = 64,
        max_inflight_per_replica: int = 32,
        unhealthy_cooldown_s: float = 30.0,
    ) -> None:
        self.virtual_nodes = max(8, virtual_nodes)
        self.max_inflight_per_replica = max(1, max_inflight_per_replica)
        self.unhealthy_cooldown_s = unhealthy_cooldown_s
        self._in_flight: dict[str, int] = {}
        self._unhealthy_until: dict[str, float] = {}
        self.affinity_hits: int = 0
        self.spillovers: int = 0

    @staticmethod
    def parse_endpoints(raw_base_url: str) -> list[str]:
        return [u.strip().rstrip("/") for u in (raw_base_url or "").split(",") if u.strip()]

    @staticmethod
    def _ring_point(key: str) -> int:
        return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")

    def _is_healthy(self, endpoint: str, now: float) -> bool:
        until = self._unhealthy_until.get(endpoint, 0.0)
        if until <= now:
            self._unhealthy_until.pop(endpoint, None)
            return True
        return False

    def mark_unhealthy(self, endpoint: str) -> None:
        ep = endpoint.strip().rstrip("/")
        self._unhealthy_until[ep] = time.monotonic() + self.unhealthy_cooldown_s

    def mark_healthy(self, endpoint: str) -> None:
        self._unhealthy_until.pop(endpoint.strip().rstrip("/"), None)

    def select_endpoint(
        self,
        raw_base_url: str,
        *,
        model_id: str = "",
        prefix_hash: str = "",
        thread_id: str = "",
    ) -> str:
        """Select optimal NIM replica maximizing GPU KV-cache reuse while bounding queue depth."""
        endpoints = self.parse_endpoints(raw_base_url)
        if not endpoints:
            return ""
        if len(endpoints) == 1:
            self.affinity_hits += 1
            return endpoints[0]

        ring: list[tuple[int, str]] = []
        for ep in sorted(endpoints):
            for v in range(self.virtual_nodes):
                ring.append((self._ring_point(f"{ep}#vn{v}"), ep))
        ring.sort(key=lambda item: item[0])
        points = [pt for pt, _ in ring]

        routing_key = f"{model_id}:{prefix_hash}:{thread_id or 'stateless'}"
        target = self._ring_point(routing_key)
        start_idx = bisect_left(points, target) % len(ring)

        # Ordered unique replicas walking clockwise from prefix hash position
        ordered_candidates: list[str] = []
        seen: set[str] = set()
        for offset in range(len(ring)):
            ep = ring[(start_idx + offset) % len(ring)][1]
            if ep not in seen:
                seen.add(ep)
                ordered_candidates.append(ep)
                if len(ordered_candidates) == len(endpoints):
                    break

        now = time.monotonic()
        for rank, candidate in enumerate(ordered_candidates):
            if not self._is_healthy(candidate, now):
                continue
            active = self._in_flight.get(candidate, 0)
            if active < self.max_inflight_per_replica:
                if rank == 0:
                    self.affinity_hits += 1
                else:
                    self.spillovers += 1
                return candidate

        # All healthy replicas are saturated: pick healthy (or any) replica with lowest active slots
        healthy = [c for c in ordered_candidates if self._is_healthy(c, now)] or ordered_candidates
        chosen = min(healthy, key=lambda ep: self._in_flight.get(ep, 0))
        self.spillovers += 1
        return chosen

    @contextmanager
    def lease_slot(
        self,
        raw_base_url: str,
        *,
        model_id: str = "",
        prefix_hash: str = "",
        thread_id: str = "",
    ) -> Generator[str, None, None]:
        ep = self.select_endpoint(
            raw_base_url,
            model_id=model_id,
            prefix_hash=prefix_hash,
            thread_id=thread_id,
        )
        if ep:
            self._in_flight[ep] = self._in_flight.get(ep, 0) + 1
        try:
            yield ep
        finally:
            if ep:
                self._in_flight[ep] = max(0, self._in_flight.get(ep, 1) - 1)

    def status_summary(self) -> dict[str, Any]:
        total = self.affinity_hits + self.spillovers
        return {
            "in_flight_by_replica": dict(self._in_flight),
            "unhealthy_replicas": list(self._unhealthy_until.keys()),
            "affinity_hits": self.affinity_hits,
            "spillovers": self.spillovers,
            "kv_affinity_rate": round(self.affinity_hits / total, 4) if total else 1.0,
        }


# ─── 2. L1 Exact + L2 Tenant-Scoped Qdrant Semantic Cache ───────────────


class ExactL1ResponseCache:
    """Sub-millisecond tenant-scoped LRU+TTL cache for deterministic `temperature == 0` prompts."""

    def __init__(self, max_entries: int = 2048, ttl_s: float = 1800.0) -> None:
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0

    def get(self, partition_key: str, prompt_text: str) -> str | None:
        full_key = compute_prefix_hash(partition_key, prompt_text)
        row = self._entries.get(full_key)
        now = time.monotonic()
        if row is not None:
            expires_at, response_text = row
            if expires_at > now:
                self._entries.move_to_end(full_key)
                self.hits += 1
                return response_text
            self._entries.pop(full_key, None)
        self.misses += 1
        return None

    def put(self, partition_key: str, prompt_text: str, response_text: str) -> None:
        full_key = compute_prefix_hash(partition_key, prompt_text)
        self._entries[full_key] = (time.monotonic() + self.ttl_s, response_text)
        self._entries.move_to_end(full_key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)


class TenantSemanticCache:
    """Qdrant-backed L2 Semantic Cache strictly partitioned by `(tenant_id, workspace_id, agent_id, prompt_hash, kb_hash)`."""

    def __init__(
        self,
        store: MultiTenantVectorStore | None = None,
        *,
        similarity_threshold: float = 0.96,
        ttl_s: float = 3600.0,
    ) -> None:
        self.store = store or vector_store
        self.similarity_threshold = similarity_threshold
        self.ttl_s = ttl_s
        self.l1 = ExactL1ResponseCache(ttl_s=ttl_s)
        self.l2_hits: int = 0
        self.l2_misses: int = 0

    @staticmethod
    def build_partition_key(
        *,
        ctx: TenantContext,
        agent_id: str = "agent",
        system_prompt_hash: str = "default",
        kb_version_hash: str = "none",
    ) -> str:
        tid = sanitize_tenant_identifier(ctx.tenant_id)
        wid = sanitize_tenant_identifier(ctx.workspace_id)
        return compute_prefix_hash(tid, wid, agent_id, system_prompt_hash, kb_version_hash)

    async def lookup(
        self,
        *,
        prompt_text: str,
        query_vector: list[float],
        ctx: TenantContext | None = None,
        agent_id: str = "agent",
        system_prompt_hash: str = "default",
        kb_version_hash: str = "none",
    ) -> tuple[str | None, str]:
        """Return `(cached_answer, tier_name)` where `tier_name` is `'l1_exact'`, `'l2_semantic'`, or `'miss'`."""
        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        part = self.build_partition_key(
            ctx=active,
            agent_id=agent_id,
            system_prompt_hash=system_prompt_hash,
            kb_version_hash=kb_version_hash,
        )

        # Tier L1 check (< 0.1ms)
        l1_hit = self.l1.get(part, prompt_text)
        if l1_hit is not None:
            return l1_hit, "l1_exact"

        if not query_vector:
            self.l2_misses += 1
            return None, "miss"

        collection = self.store.collection_name_for_dim(len(query_vector), namespace="semcache")
        client = await self.store.get_client()
        if not await client.collection_exists(collection):
            self.l2_misses += 1
            return None, "miss"

        now_epoch = time.time()
        res = await client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(key="tenant_id", match=qmodels.MatchValue(value=tid)),
                    qmodels.FieldCondition(key="workspace_id", match=qmodels.MatchValue(value=wid)),
                    qmodels.FieldCondition(key="cache_partition", match=qmodels.MatchValue(value=part)),
                    qmodels.FieldCondition(key="expires_at", range=qmodels.Range(gt=now_epoch)),
                ]
            ),
            limit=1,
            score_threshold=self.similarity_threshold,
            with_payload=True,
        )
        if res.points:
            answer = (res.points[0].payload or {}).get("response_text")
            if isinstance(answer, str) and answer:
                self.l2_hits += 1
                self.l1.put(part, prompt_text, answer)
                return answer, "l2_semantic"

        self.l2_misses += 1
        return None, "miss"

    async def store_response(
        self,
        *,
        prompt_text: str,
        query_vector: list[float],
        response_text: str,
        ctx: TenantContext | None = None,
        agent_id: str = "agent",
        system_prompt_hash: str = "default",
        kb_version_hash: str = "none",
    ) -> None:
        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        part = self.build_partition_key(
            ctx=active,
            agent_id=agent_id,
            system_prompt_hash=system_prompt_hash,
            kb_version_hash=kb_version_hash,
        )
        self.l1.put(part, prompt_text, response_text)
        if not query_vector:
            return

        collection = await self.store.ensure_collection(len(query_vector), namespace="semcache")
        client = await self.store.get_client()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"semcache:{part}:{prompt_text}"))
        await client.upsert(
            collection_name=collection,
            points=[
                qmodels.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "tenant_id": tid,
                        "workspace_id": wid,
                        "cache_partition": part,
                        "response_text": response_text,
                        "expires_at": time.time() + self.ttl_s,
                    },
                )
            ],
        )

    def status_summary(self) -> dict[str, Any]:
        return {
            "similarity_threshold": self.similarity_threshold,
            "l1_hits": self.l1.hits,
            "l1_misses": self.l1.misses,
            "l2_semantic_hits": self.l2_hits,
            "l2_semantic_misses": self.l2_misses,
        }


# ─── 3. Async Write-Behind Usage & FinOps Telemetry Collector ───────────


@dataclass(slots=True)
class UsageRecord:
    id: str
    tenant_id: str
    workspace_id: str
    user_id: str
    provider: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    estimated_usd: float
    cache_tier: str
    latency_ms: float
    created_at: str


# Approximate blended per-1M-token USD reference rates for FinOps tracking
MODEL_RATES_PER_1M_TOKENS: dict[str, tuple[float, float]] = {
    "nvidia/nemotron-3-ultra-550b-a55b": (1.20, 3.60),
    "nvidia/nemotron-3-super-120b-a12b": (0.45, 1.35),
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (0.35, 0.90),
    "nvidia/llama-3.1-nemotron-nano-8b-v1": (0.08, 0.20),
    "deepseek-ai/deepseek-r1": (0.55, 2.19),
    "qwen/qwen3-235b-a22b": (0.40, 1.20),
    "meta/llama-3.3-70b-instruct": (0.30, 0.80),
}


def estimate_call_usd(model_name: str, prompt_tokens: int, completion_tokens: int, cache_tier: str = "none") -> float:
    if cache_tier in ("l1_exact", "l2_semantic"):
        return 0.0
    in_rate, out_rate = MODEL_RATES_PER_1M_TOKENS.get(model_name, (0.40, 1.20))
    return round((prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0, 8)


class UsageTelemetryCollector:
    """Lock-free in-memory buffer bulk-flushing usage telemetry every `1.0s` off the hot path."""

    def __init__(self, flush_interval_s: float = 1.0, max_batch: int = 250) -> None:
        self.flush_interval_s = flush_interval_s
        self.max_batch = max_batch
        self._buffer: deque[UsageRecord] = deque(maxlen=20_000)
        self._task: asyncio.Task[None] | None = None
        self.flushed_records: int = 0

    def record(
        self,
        *,
        provider: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_tier: str = "none",
        latency_ms: float = 0.0,
        ctx: TenantContext | None = None,
    ) -> UsageRecord:
        active = ctx or get_tenant_context()
        usd = estimate_call_usd(model_name, prompt_tokens, completion_tokens, cache_tier=cache_tier)
        rec = UsageRecord(
            id=str(uuid.uuid4()),
            tenant_id=active.tenant_id,
            workspace_id=active.workspace_id,
            user_id=active.user_id,
            provider=provider,
            model_name=model_name,
            prompt_tokens=max(0, int(prompt_tokens)),
            completion_tokens=max(0, int(completion_tokens)),
            estimated_usd=usd,
            cache_tier=cache_tier,
            latency_ms=round(float(latency_ms), 2),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        self._buffer.append(rec)
        return rec

    async def flush_now(self) -> int:
        if not self._buffer:
            return 0
        batch: list[tuple[Any, ...]] = []
        while self._buffer and len(batch) < self.max_batch:
            r = self._buffer.popleft()
            batch.append(
                (
                    r.id,
                    r.tenant_id,
                    r.workspace_id,
                    r.user_id,
                    r.provider,
                    r.model_name,
                    r.prompt_tokens,
                    r.completion_tokens,
                    r.estimated_usd,
                    r.cache_tier,
                    r.latency_ms,
                    r.created_at,
                )
            )
        if not batch:
            return 0
        await asyncio.to_thread(self._bulk_insert_sync, batch)
        self.flushed_records += len(batch)
        return len(batch)

    @staticmethod
    def _bulk_insert_sync(rows: list[tuple[Any, ...]]) -> None:
        with get_db() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO usage_ledger
                   (id, tenant_id, workspace_id, user_id, provider, model_name,
                    prompt_tokens, completion_tokens, estimated_usd, cache_tier,
                    latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()

    def start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._task.get_loop() is not loop:
            self._task = asyncio.create_task(self._background_flush_loop(), name="nimagents-telemetry-flush")

    async def stop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is not None and not self._task.done() and self._task.get_loop() is loop:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        await self.flush_now()

    async def _background_flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval_s)
            try:
                while self._buffer:
                    await self.flush_now()
            except Exception as exc:
                logger.warning("Telemetry flush warning: %s", exc)

    def status_summary(self) -> dict[str, Any]:
        return {
            "buffered_records": len(self._buffer),
            "flushed_records": self.flushed_records,
            "flush_interval_s": self.flush_interval_s,
        }


prefix_router = PrefixAwareEndpointRouter()
semantic_cache = TenantSemanticCache()
telemetry_collector = UsageTelemetryCollector()

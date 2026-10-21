"""Multi-Tenant FinOps Cost, Token Throughput, Cache & Latency Observatory API (`/api/analytics`)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import math
from typing import Any
import uuid

from fastapi import APIRouter
from pydantic import BaseModel, Field

from agent.inference_gateway import (
    estimate_call_usd,
    prefix_router,
    semantic_cache,
    telemetry_collector,
)
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db
from api.middleware.admission import admission_controller
from api.workers.priority_queue import work_pool

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class RecordSpanRequest(BaseModel):
    provider: str = Field(default="nvidia_nim", max_length=64)
    model_name: str = Field(default="nvidia/nemotron-3-super-120b-a12b", max_length=200)
    agent_id: str = Field(default="nimAgents Assistant", max_length=128)
    thread_id: str = Field(default="", max_length=128)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    kv_cached_tokens: int = Field(default=0, ge=0)
    cache_tier: str = Field(default="none", max_length=32)
    latency_ms: float = Field(default=0.0, ge=0.0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    queue_wait_ms: float = Field(default=0.0, ge=0.0)


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    idx = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = min(len(sorted_vals) - 1, int(math.ceil(idx)))
    frac = idx - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 2)


def _query_analytics_snapshot_sync(
    ctx: TenantContext,
    hours: int,
    admission_summary: dict[str, Any],
) -> dict[str, Any]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, hours))).isoformat()
    with get_db() as conn:
        rows = [
            dict(r)
            for r in conn.execute(
                """SELECT * FROM usage_ledger
                   WHERE tenant_id = ? AND workspace_id = ? AND created_at >= ?
                   ORDER BY created_at ASC""",
                (ctx.tenant_id, ctx.workspace_id, cutoff),
            ).fetchall()
        ]
        compaction_avg = conn.execute(
            """SELECT COALESCE(AVG(kv_cache_hit_pct), 0.0) AS avg_kv
               FROM session_compactions
               WHERE tenant_id = ? AND workspace_id = ?""",
            (ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
        safety_counts = [
            dict(r)
            for r in conn.execute(
                """SELECT action, COUNT(*) AS cnt
                   FROM guardrail_events
                   WHERE tenant_id = ? AND workspace_id = ?
                   GROUP BY action""",
                (ctx.tenant_id, ctx.workspace_id),
            ).fetchall()
        ]
        latest_eval = conn.execute(
            """SELECT pass_rate, gate_passed, dataset_name, created_at
               FROM eval_runs
               WHERE tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (ctx.tenant_id, ctx.workspace_id),
        ).fetchone()

    total_calls = len(rows)
    total_prompt = sum(int(r.get("prompt_tokens") or 0) for r in rows)
    total_completion = sum(int(r.get("completion_tokens") or 0) for r in rows)
    total_tokens = total_prompt + total_completion
    total_kv_cached = sum(int(r.get("kv_cached_tokens") or 0) for r in rows)
    total_usd = round(sum(float(r.get("estimated_usd") or 0.0) for r in rows), 6)

    saved_usd = 0.0
    cache_hit_calls = 0
    for r in rows:
        tier = str(r.get("cache_tier") or "none")
        if tier in ("l1_exact", "l2_semantic"):
            cache_hit_calls += 1
            saved_usd += estimate_call_usd(
                str(r.get("model_name") or ""),
                int(r.get("prompt_tokens") or 0),
                int(r.get("completion_tokens") or 0),
                cache_tier="none",
            )

    latencies = sorted(float(r.get("latency_ms") or 0.0) for r in rows if float(r.get("latency_ms") or 0.0) > 0)
    ttfts = sorted(
        (
            float(r.get("ttft_ms") or 0.0)
            if float(r.get("ttft_ms") or 0.0) > 0
            else round(float(r.get("latency_ms") or 0.0) * 0.28, 2)
        )
        for r in rows
        if (float(r.get("ttft_ms") or 0.0) > 0 or float(r.get("latency_ms") or 0.0) > 0)
    )
    queue_waits = sorted(float(r.get("queue_wait_ms") or 0.0) for r in rows)

    router_summary = prefix_router.status_summary()
    sem_summary = semantic_cache.status_summary()

    token_kv_ratio = (total_kv_cached / total_prompt * 100.0) if total_prompt > 0 else 0.0
    compaction_kv = float(compaction_avg["avg_kv"] or 0.0) if compaction_avg else 0.0
    effective_kv_hit_pct = round(
        token_kv_ratio
        if token_kv_ratio > 0
        else (compaction_kv if compaction_kv > 0 else router_summary["kv_affinity_rate"] * 100.0),
        2,
    )

    sem_total_lookups = (
        sem_summary["l1_hits"]
        + sem_summary["l2_semantic_hits"]
        + sem_summary["l2_semantic_misses"]
    )
    live_sem_pct = (
        (sem_summary["l1_hits"] + sem_summary["l2_semantic_hits"]) / sem_total_lookups * 100.0
        if sem_total_lookups > 0
        else 0.0
    )
    ledger_sem_pct = (cache_hit_calls / total_calls * 100.0) if total_calls > 0 else 0.0
    semantic_hit_pct = round(max(ledger_sem_pct, live_sem_pct), 2)

    policy = admission_summary.get("policy") or {}
    daily_budget_usd = float(policy.get("max_daily_usd") or 100.0)
    budget_pct = round(min(100.0, (total_usd / max(daily_budget_usd, 0.01)) * 100.0), 2)
    window_minutes = max(1.0, float(hours) * 60.0)
    avg_tpm = round(total_tokens / window_minutes, 2)

    # Time-series buckets (`YYYY-MM-DD HH:00`) for `recharts`
    buckets: dict[str, dict[str, Any]] = {}
    by_model_map: dict[str, dict[str, Any]] = {}
    by_agent_map: dict[str, dict[str, Any]] = {}

    for r in rows:
        raw_ts = str(r.get("created_at") or "")
        bucket_label = raw_ts[:13].replace("T", " ") + ":00" if len(raw_ts) >= 13 else "recent"
        pt = int(r.get("prompt_tokens") or 0)
        ct = int(r.get("completion_tokens") or 0)
        kv_t = int(r.get("kv_cached_tokens") or 0)
        usd = float(r.get("estimated_usd") or 0.0)
        lat = float(r.get("latency_ms") or 0.0)
        ttft = float(r.get("ttft_ms") or 0.0) or round(lat * 0.28, 2)
        qwait = float(r.get("queue_wait_ms") or 0.0)
        tier = str(r.get("cache_tier") or "none")

        b = buckets.setdefault(
            bucket_label,
            {
                "bucket": bucket_label[-5:] if len(bucket_label) >= 5 else bucket_label,
                "full_bucket": bucket_label,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens": 0,
                "kv_cached_tokens": 0,
                "usd": 0.0,
                "saved_usd": 0.0,
                "calls": 0,
                "_ttfts": [],
                "_qwaits": [],
            },
        )
        b["prompt_tokens"] += pt
        b["completion_tokens"] += ct
        b["tokens"] += pt + ct
        b["kv_cached_tokens"] += kv_t
        b["usd"] += usd
        if tier in ("l1_exact", "l2_semantic"):
            b["saved_usd"] += estimate_call_usd(str(r.get("model_name") or ""), pt, ct, cache_tier="none")
        b["calls"] += 1
        if ttft > 0:
            b["_ttfts"].append(ttft)
        b["_qwaits"].append(qwait)

        mname = str(r.get("model_name") or "unknown")
        m_entry = by_model_map.setdefault(
            mname,
            {
                "model_name": mname,
                "provider": str(r.get("provider") or "nvidia_nim"),
                "calls": 0,
                "tokens": 0,
                "usd": 0.0,
                "_latencies": [],
                "cache_hits": 0,
            },
        )
        m_entry["calls"] += 1
        m_entry["tokens"] += pt + ct
        m_entry["usd"] += usd
        if lat > 0:
            m_entry["_latencies"].append(lat)
        if tier in ("l1_exact", "l2_semantic"):
            m_entry["cache_hits"] += 1

        aid = str(r.get("agent_id") or "agent")
        a_entry = by_agent_map.setdefault(
            aid,
            {
                "agent_id": aid,
                "calls": 0,
                "tokens": 0,
                "usd": 0.0,
                "_ttfts": [],
            },
        )
        a_entry["calls"] += 1
        a_entry["tokens"] += pt + ct
        a_entry["usd"] += usd
        if ttft > 0:
            a_entry["_ttfts"].append(ttft)

    timeseries: list[dict[str, Any]] = []
    peak_tpm = 0.0
    for key in sorted(buckets.keys()):
        b = buckets[key]
        t_sorted = sorted(b.pop("_ttfts"))
        q_sorted = sorted(b.pop("_qwaits"))
        bucket_tpm = round(b["tokens"] / 60.0, 2)
        peak_tpm = max(peak_tpm, bucket_tpm)
        kv_pct = (
            round((b["kv_cached_tokens"] / b["prompt_tokens"]) * 100.0, 1)
            if b["prompt_tokens"] > 0
            else effective_kv_hit_pct
        )
        timeseries.append(
            {
                **b,
                "tpm": bucket_tpm,
                "usd": round(b["usd"], 6),
                "saved_usd": round(b["saved_usd"], 6),
                "kv_hit_pct": kv_pct,
                "p50_ttft_ms": _percentile(t_sorted, 50),
                "p95_ttft_ms": _percentile(t_sorted, 95),
                "p99_ttft_ms": _percentile(t_sorted, 99),
                "queue_wait_ms": _percentile(q_sorted, 95),
            }
        )

    by_model = [
        {
            "model_name": m["model_name"],
            "short_name": m["model_name"].split("/")[-1],
            "provider": m["provider"],
            "calls": m["calls"],
            "tokens": m["tokens"],
            "usd": round(m["usd"], 6),
            "avg_latency_ms": round(sum(m["_latencies"]) / len(m["_latencies"]), 2)
            if m["_latencies"]
            else 0.0,
            "cache_hits": m["cache_hits"],
        }
        for m in sorted(by_model_map.values(), key=lambda x: x["tokens"], reverse=True)
    ]

    by_agent = [
        {
            "agent_id": a["agent_id"],
            "calls": a["calls"],
            "tokens": a["tokens"],
            "tpm": round(a["tokens"] / window_minutes, 2),
            "usd": round(a["usd"], 6),
            "p95_ttft_ms": _percentile(sorted(a["_ttfts"]), 95),
        }
        for a in sorted(by_agent_map.values(), key=lambda x: x["tokens"], reverse=True)
    ]

    safety_map = {str(s["action"]): int(s["cnt"]) for s in safety_counts}

    return {
        "tenant": {
            "tenant_id": ctx.tenant_id,
            "workspace_id": ctx.workspace_id,
            "plan_tier": ctx.plan_tier,
        },
        "window_hours": hours,
        "kpis": {
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "avg_tpm": avg_tpm,
            "peak_tpm": peak_tpm,
            "total_usd": total_usd,
            "daily_budget_usd": daily_budget_usd,
            "budget_utilization_pct": budget_pct,
            "projected_monthly_usd": round(total_usd * (720.0 / max(1.0, float(hours))), 4),
            "kv_cache_hit_pct": effective_kv_hit_pct,
            "semantic_cache_hit_pct": semantic_hit_pct,
            "semantic_cache_saved_usd": round(saved_usd, 6),
            "p50_ttft_ms": _percentile(ttfts, 50),
            "p95_ttft_ms": _percentile(ttfts, 95),
            "p99_ttft_ms": _percentile(ttfts, 99),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p99_latency_ms": _percentile(latencies, 99),
            "p50_queue_wait_ms": _percentile(queue_waits, 50),
            "p95_queue_wait_ms": _percentile(queue_waits, 95),
        },
        "safety": {
            "blocked": safety_map.get("block", 0),
            "redacted": safety_map.get("redact", 0),
            "warned": safety_map.get("warn", 0),
            "total": sum(safety_map.values()),
        },
        "latest_eval": dict(latest_eval) if latest_eval else None,
        "workers": work_pool.status_summary(),
        "timeseries": timeseries,
        "by_model": by_model,
        "by_agent": by_agent,
    }


@router.get("/overview")
async def get_analytics_overview(hours: int = 24):
    """Return real-time multi-tenant FinOps cost, TPM, cache savings & latency percentiles."""
    ctx = get_tenant_context()
    await telemetry_collector.flush_now()
    admission_summary = await admission_controller.status_summary(ctx)
    return await asyncio.to_thread(
        _query_analytics_snapshot_sync,
        ctx,
        max(1, min(hours, 720)),
        admission_summary,
    )


@router.post("/record", status_code=201)
async def record_analytics_span(body: RecordSpanRequest):
    """Record a multi-tenant inference telemetry span and flush it to `usage_ledger`."""
    ctx = get_tenant_context()
    rec = telemetry_collector.record(
        provider=body.provider,
        model_name=body.model_name,
        prompt_tokens=body.prompt_tokens,
        completion_tokens=body.completion_tokens,
        kv_cached_tokens=body.kv_cached_tokens,
        cache_tier=body.cache_tier,
        latency_ms=body.latency_ms,
        ttft_ms=body.ttft_ms,
        queue_wait_ms=body.queue_wait_ms,
        agent_id=body.agent_id,
        thread_id=body.thread_id,
        ctx=ctx,
    )
    await telemetry_collector.flush_now()
    return {
        "id": rec.id,
        "tenant_id": rec.tenant_id,
        "estimated_usd": rec.estimated_usd,
        "ttft_ms": rec.ttft_ms,
    }


@router.post("/seed-demo", status_code=201)
async def seed_demo_telemetry():
    """Seed realistic multi-agent, multi-NIM telemetry spans across the past 12 hours for live visualization."""
    ctx = get_tenant_context()
    now = datetime.now(timezone.utc)
    profiles = [
        ("nvidia/nemotron-3-super-120b-a12b", "Architecture Copilot", "none", 14200, 620, 12100, 410.0, 96.0, 4.2),
        ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "Support RAG Agent", "none", 9800, 410, 8600, 265.0, 64.0, 2.8),
        ("nvidia/llama-3.1-nemotron-nano-8b-v1", "Fast Router & Context Shaper", "l1_exact", 4200, 190, 3900, 38.0, 11.0, 0.7),
        ("nvidia/nemotron-3-super-120b-a12b", "Support RAG Agent", "l2_semantic", 11400, 510, 11400, 16.0, 8.5, 0.5),
        ("deepseek-ai/deepseek-r1", "Deep Research Sub-Agent", "none", 18600, 1150, 15400, 690.0, 148.0, 7.4),
    ]
    rows: list[tuple[Any, ...]] = []
    for hour_offset in range(11, -1, -1):
        for idx, (model_name, agent_id, tier, pt_base, ct_base, kv_base, lat_base, ttft_base, qw_base) in enumerate(profiles):
            scale = 1.0 + ((11 - hour_offset) % 4) * 0.18 + idx * 0.05
            pt = int(pt_base * scale)
            ct = int(ct_base * scale)
            kv = min(pt, int(kv_base * scale))
            usd = estimate_call_usd(model_name, pt, ct, cache_tier=tier)
            ts = (now - timedelta(hours=hour_offset, minutes=idx * 9)).isoformat()
            rows.append(
                (
                    str(uuid.uuid4()),
                    ctx.tenant_id,
                    ctx.workspace_id,
                    ctx.user_id,
                    agent_id,
                    f"demo-thread-{idx + 1}",
                    "nvidia_nim",
                    model_name,
                    pt,
                    ct,
                    kv,
                    usd,
                    tier,
                    round(lat_base * (0.9 + (idx % 3) * 0.12), 2),
                    round(ttft_base * (0.92 + (hour_offset % 3) * 0.09), 2),
                    round(qw_base * (0.85 + (hour_offset % 4) * 0.15), 2),
                    ts,
                )
            )

    def _insert() -> int:
        with get_db() as conn:
            conn.executemany(
                """INSERT INTO usage_ledger
                   (id, tenant_id, workspace_id, user_id, agent_id, thread_id,
                    provider, model_name, prompt_tokens, completion_tokens,
                    kv_cached_tokens, estimated_usd, cache_tier,
                    latency_ms, ttft_ms, queue_wait_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        return len(rows)

    inserted = await asyncio.to_thread(_insert)
    return {"seeded_spans": inserted, "tenant_id": ctx.tenant_id}

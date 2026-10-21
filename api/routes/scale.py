"""Multi-User Scale, Tenant Quotas, Worker Queues & Inference Gateway Observatory API (`/api/scale`)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.inference_gateway import prefix_router, semantic_cache, telemetry_collector
from api.core.http_pool import loop_watchdog
from api.core.tenancy import (
    generate_meili_tenant_token,
    get_tenant_context,
)
from api.core.vector_store import vector_store
from api.db import get_db
from api.db_async import check_pgbouncer_pool_budget, resolve_database_url
from api.middleware.admission import TenantQuotaPolicy, admission_controller
from api.workers.priority_queue import work_pool

router = APIRouter(prefix="/api/scale", tags=["scale"])


class QuotaUpdateRequest(BaseModel):
    max_concurrent_runs: int = Field(default=12, ge=1, le=500)
    max_rpm: int = Field(default=240, ge=1, le=50_000)
    max_tpm: int = Field(default=400_000, ge=100, le=100_000_000)
    max_daily_usd: float = Field(default=100.0, ge=0.01, le=1_000_000.0)


class MeiliTenantTokenRequest(BaseModel):
    api_key_uid: str = Field(..., min_length=1, max_length=128)
    signing_api_key: str = Field(..., min_length=1, max_length=512)
    indexes: list[str] = Field(default_factory=lambda: ["*"])
    expires_in_seconds: int = Field(default=3600, ge=60, le=86400)


@router.get("/status")
async def get_scale_status():
    """Return real-time health, tenant isolation, queue lag, and GPU gateway metrics."""
    ctx = get_tenant_context()
    await telemetry_collector.flush_now()
    admission = await admission_controller.status_summary(ctx)
    qdrant_status = await vector_store.status_summary()

    with get_db() as conn:
        ledger_row = conn.execute(
            """SELECT
                 COUNT(*) AS calls,
                 COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                 COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                 COALESCE(SUM(estimated_usd), 0.0) AS total_usd
               FROM usage_ledger
               WHERE tenant_id = ? AND workspace_id = ?""",
            (ctx.tenant_id, ctx.workspace_id),
        ).fetchone()

    db_url = resolve_database_url()
    db_engine = "postgresql_asyncpg_rls" if db_url.startswith("postgresql") else "sqlite_wal_dual"

    return {
        "tenant": {
            "tenant_id": ctx.tenant_id,
            "workspace_id": ctx.workspace_id,
            "user_id": ctx.user_id,
            "plan_tier": ctx.plan_tier,
            "storage_prefix": ctx.storage_prefix,
        },
        "database": {
            "engine": db_engine,
            "pool_budget": check_pgbouncer_pool_budget(
                app_replicas=2,
                pool_size=30,
                postgres_max_connections=200,
            ),
        },
        "event_loop": loop_watchdog.snapshot(),
        "admission": admission,
        "workers": work_pool.status_summary(),
        "vector_store": qdrant_status,
        "inference_gateway": {
            "prefix_router": prefix_router.status_summary(),
            "semantic_cache": semantic_cache.status_summary(),
            "telemetry": telemetry_collector.status_summary(),
            "gpu_micro_batcher": __import__("api.services.batcher", fromlist=["gpu_batcher"]).gpu_batcher.status_summary(),
        },
        "tenant_usage": {
            "calls": int(ledger_row["calls"]),
            "prompt_tokens": int(ledger_row["prompt_tokens"]),
            "completion_tokens": int(ledger_row["completion_tokens"]),
            "estimated_usd": round(float(ledger_row["total_usd"]), 6),
        },
    }


@router.put("/quotas")
async def update_tenant_quotas(body: QuotaUpdateRequest):
    """Dynamically update admission quotas for the active tenant."""
    ctx = get_tenant_context()
    policy = TenantQuotaPolicy(
        max_concurrent_runs=body.max_concurrent_runs,
        max_rpm=body.max_rpm,
        max_tpm=body.max_tpm,
        max_daily_usd=body.max_daily_usd,
    )
    admission_controller.set_tenant_policy(ctx.tenant_id, policy)
    return await admission_controller.status_summary(ctx)


@router.get("/jobs")
def list_background_jobs(limit: int = 50):
    """List background jobs strictly scoped to the active `(tenant_id, workspace_id)`."""
    ctx = get_tenant_context()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM background_jobs
               WHERE tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (ctx.tenant_id, ctx.workspace_id, min(limit, 200)),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/meili-token")
def create_meili_tenant_token(body: MeiliTenantTokenRequest):
    """Mint a signed HS256 Meilisearch Tenant Token locked to the caller's tenant & workspace."""
    ctx = get_tenant_context()
    try:
        token = generate_meili_tenant_token(
            api_key_uid=body.api_key_uid,
            api_key=body.signing_api_key,
            ctx=ctx,
            indexes=body.indexes,
            expires_in_seconds=body.expires_in_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "tenant_id": ctx.tenant_id,
        "workspace_id": ctx.workspace_id,
        "token": token,
    }

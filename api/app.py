"""nimAgents Control Plane API — Multi-Tenant FastAPI Service for Models, RAG, Storage, MCP, Voice, Scale & Guardrails."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.inference_gateway import telemetry_collector
from agent.tools import get_catalog_info
from api.core.http_pool import (
    close_shared_http_client,
    get_shared_http_client,
    loop_watchdog,
)
from api.core.tenancy import TenancyMiddleware
from api.core.vector_store import vector_store
from api.db import init_db
from api.db_async import close_async_engines
from api.middleware.admission import AdmissionMiddleware
from api.routes import (
    analytics as analytics_router,
    context as context_router,
    evals as evals_router,
    guardrails,
    lexical_rag,
    local_rag,
    mcp,
    memories as memories_router,
    models,
    rag,
    rag_pipeline as rag_pipeline_router,
    scale,
    sessions as sessions_router,
    skills as skills_router,
    storage,
    tools as tools_router,
    voice,
)
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.api")

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled HTTP/2 AsyncClient (`max_connections=1000`)."""
    return get_shared_http_client()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle for pooled I/O, telemetry flusher, and loop watchdog."""
    logger.info("Initialising nimAgents multi-tenant database…")
    init_db()
    loop_watchdog.start()
    telemetry_collector.start()
    client = get_http_client()

    headers: dict[str, str] = {}
    if MEILI_MASTER_KEY:
        headers["Authorization"] = f"Bearer {MEILI_MASTER_KEY}"

    try:
        resp = await client.get(f"{MEILI_URL}/health", headers=headers, timeout=3.0)
        if resp.status_code == 200:
            logger.info("Meilisearch is healthy at %s", MEILI_URL)
        else:
            logger.warning("Meilisearch responded with %s", resp.status_code)
    except Exception:
        logger.warning(
            "Meilisearch unreachable at %s — lexical RAG will activate once started",
            MEILI_URL,
        )

    logger.info("nimAgents Control Plane API ready")
    yield
    await telemetry_collector.stop()
    await loop_watchdog.stop()
    await work_pool.stop()
    await vector_store.close()
    await close_async_engines()
    await close_shared_http_client()
    logger.info("nimAgents Control Plane API shutting down")


app = FastAPI(
    title="nimAgents Control Plane API",
    version="1.0.0",
    lifespan=lifespan,
)

# Note: Starlette executes `add_middleware` LIFO (last added wraps outermost).
# Execution order on incoming request:
#   CORS -> TenancyMiddleware (sets ContextVar) -> AdmissionMiddleware (enforces tenant quotas) -> Router
app.add_middleware(AdmissionMiddleware)
app.add_middleware(TenancyMiddleware)

allowed_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
allowed_origins = (
    ["*"]
    if allowed_origins_env == "*"
    else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=allowed_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


app.include_router(models.router)
app.include_router(mcp.router)
app.include_router(rag.router)
app.include_router(local_rag.router)
app.include_router(lexical_rag.router)
app.include_router(voice.router)
app.include_router(storage.router)
app.include_router(guardrails.router)
app.include_router(scale.router)
app.include_router(tools_router.router)
app.include_router(skills_router.router)
app.include_router(context_router.router)
app.include_router(sessions_router.router)
app.include_router(memories_router.router)
app.include_router(rag_pipeline_router.router)
app.include_router(analytics_router.router)
app.include_router(evals_router.router)


@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "nimagents-api",
        "version": "1.0.0",
        "tenancy": True,
        "vector_engine": "qdrant",
        "admission_control": True,
        "docs": "/docs",
    }


# Freeze immortal module/schema objects after router registration so Generation-2 GC sweeps
# never pause the main `asyncio` event loop under high-concurrency multi-user bursts!
import gc as _gc

_gc.collect()
_gc.freeze()
_gc.set_threshold(50_000, 20, 20)

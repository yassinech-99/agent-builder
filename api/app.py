"""nimAgents Control Plane API — FastAPI service for models, RAG, storage, MCP, voice, and guardrails."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.db import init_db
from api.routes import guardrails, lexical_rag, local_rag, mcp, models, rag, storage, voice
from agent.tools import get_catalog_info

logger = logging.getLogger("nimagents.api")

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "")

_shared_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled HTTP/2 AsyncClient (lazy-creating in tests if needed)."""
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.AsyncClient(
            http2=True,
            timeout=httpx.Timeout(30.0, connect=8.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _shared_http_client


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    global _shared_http_client
    logger.info("Initialising nimAgents database…")
    init_db()
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
    if _shared_http_client and not _shared_http_client.is_closed:
        await _shared_http_client.aclose()
        _shared_http_client = None
    logger.info("nimAgents Control Plane API shutting down")


app = FastAPI(
    title="nimAgents Control Plane API",
    version="1.0.0",
    lifespan=lifespan,
)

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


@app.get("/api/tools/catalog")
async def list_prebuilt_tools():
    """Return the full prebuilt-tool catalog with required config metadata."""
    return get_catalog_info()


@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "nimagents-api",
        "version": "1.0.0",
        "docs": "/docs",
    }

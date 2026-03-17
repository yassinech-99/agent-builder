"""NovaGrid Agents API — production-ready FastAPI microservice.

Serves model management, RAG, MCP discovery, and tool catalog endpoints.
"""

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
from api.routes import models, mcp, rag, local_rag, lexical_rag, voice, storage, guardrails
from agent.tools import get_catalog_info

logger = logging.getLogger("novagrid.api")

MEILI_URL = os.getenv("MEILI_URL", "http://172.30.6.3:7700")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    logger.info("Initialising database…")
    init_db()

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{MEILI_URL}/health")
            if resp.status_code == 200:
                logger.info("Meilisearch is healthy at %s", MEILI_URL)
            else:
                logger.warning("Meilisearch responded with %s", resp.status_code)
    except Exception:
        logger.warning("Meilisearch unreachable at %s — lexical RAG will be unavailable", MEILI_URL)

    logger.info("NovaGrid Agents API ready")
    yield
    logger.info("NovaGrid Agents API shutting down")


app = FastAPI(
    title="NovaGrid Agents API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/health")
async def health():
    return {"status": "ok", "service": "novagrid-agents-api"}

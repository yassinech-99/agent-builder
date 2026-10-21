"""Control Plane API for Phase 7 — Contextual, Parent-Child & Hybrid Agentic RAG (`/api/rag-pipeline`)."""

from __future__ import annotations

import asyncio
from pathlib import Path
import shutil
import tempfile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.core.tenancy import get_tenant_context
from api.services.rag_pipeline import rag_pipeline, stream_upload_with_sha256
from api.utils.doc_loader import ALLOWED_EXTENSIONS

router = APIRouter(prefix="/api/rag-pipeline", tags=["rag-pipeline"])


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=160)
    embedding_model: str = Field(default="nvidia/llama-nemotron-embed-1b-v2", max_length=160)
    storage_bucket: str = Field(default="", max_length=120)
    storage_prefix: str = Field(default="", max_length=240)


class HybridQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    source_ids: list[str] | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True
    enable_query_rewrite: bool = True


@router.get("/collections")
async def list_contextual_rag_collections():
    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: rag_pipeline.list_collections(ctx=ctx))


@router.post("/collections", status_code=201)
async def create_contextual_rag_collection(body: CreateCollectionRequest):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: rag_pipeline.create_collection(
            name=body.name,
            embedding_model=body.embedding_model,
            storage_bucket=body.storage_bucket,
            storage_prefix=body.storage_prefix,
            ctx=ctx,
        )
    )


@router.delete("/collections/{source_id}")
async def delete_contextual_rag_collection(source_id: str):
    ctx = get_tenant_context()
    ok = await rag_pipeline.delete_collection(source_id, ctx=ctx)
    if not ok:
        raise HTTPException(status_code=404, detail="RAG collection not found")
    return {"ok": True}


@router.post("/collections/{source_id}/ingest")
async def ingest_document_into_collection(
    source_id: str,
    file: UploadFile = File(...),
    contextual_enrich: bool = Form(default=True),
    async_mode: bool = Form(default=True),
):
    """Stream upload (`O(1)` RAM + SHA-256 dedup) and index asynchronously (`202 Accepted`) or synchronously (`200 OK`)."""
    ctx = get_tenant_context()
    filename = Path(file.filename or "document.md").name
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS and ext not in (".markdown",):
        raise HTTPException(status_code=400, detail=f"Unsupported file extension '{ext}'")

    staged_dir = Path(tempfile.mkdtemp(prefix="nimagents_rag_stage_"))
    staged_path = staged_dir / filename
    try:
        byte_size, sha256_hash = await stream_upload_with_sha256(file, staged_path)
    except ValueError as exc:
        await asyncio.to_thread(shutil.rmtree, staged_dir, True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    if async_mode:
        queued = rag_pipeline.enqueue_staged_file_ingestion(
            source_id=source_id,
            staged_dir=staged_dir,
            file_path=staged_path,
            filename=filename,
            sha256_hash=sha256_hash,
            byte_size=byte_size,
            contextual_enrich=contextual_enrich,
            ctx=ctx,
        )
        return JSONResponse(status_code=202, content=queued)

    try:
        result = await rag_pipeline.ingest_staged_file(
            source_id=source_id,
            file_path=staged_path,
            filename=filename,
            sha256_hash=sha256_hash,
            byte_size=byte_size,
            contextual_enrich=contextual_enrich,
            ctx=ctx,
        )
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        await asyncio.to_thread(shutil.rmtree, staged_dir, True)


@router.get("/jobs/{job_id}")
async def get_rag_ingestion_job_status(job_id: str):
    ctx = get_tenant_context()
    job = await asyncio.to_thread(lambda: rag_pipeline.get_job(job_id, ctx=ctx))
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")
    return job


@router.get("/jobs/{job_id}/stream")
async def stream_rag_ingestion_job_progress(job_id: str):
    ctx = get_tenant_context()
    job = await asyncio.to_thread(lambda: rag_pipeline.get_job(job_id, ctx=ctx))
    if job is None:
        raise HTTPException(status_code=404, detail="Ingestion job not found")

    return StreamingResponse(
        rag_pipeline.subscribe_job_sse(job_id, ctx=ctx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/query")
async def execute_hybrid_contextual_rag_query(body: HybridQueryRequest):
    ctx = get_tenant_context()
    return await rag_pipeline.hybrid_retrieve(
        body.query,
        source_ids=body.source_ids,
        top_k=body.top_k,
        rerank=body.rerank,
        enable_query_rewrite=body.enable_query_rewrite,
        ctx=ctx,
    )

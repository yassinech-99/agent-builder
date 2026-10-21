"""Multi-Tenant Lexical RAG endpoints — non-blocking Meilisearch BM25 indexing with mandatory tenant filters."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.core.tenancy import (
    TenantContext,
    build_meili_tenant_filter,
    get_tenant_context,
    stream_upload_to_path,
)
from api.db import get_db
from api.utils.doc_loader import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, load_documents
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.api.lexical_rag")
router = APIRouter(prefix="/api/lexical-rag", tags=["lexical-rag"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lexical_rag"
MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY") or None


def _get_meili_client():
    import meilisearch

    return meilisearch.Client(MEILI_URL, MEILI_MASTER_KEY)


class LexicalQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)


@router.get("")
def list_sources():
    ctx = get_tenant_context()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM lexical_rag_sources
               WHERE tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC""",
            (ctx.tenant_id, ctx.workspace_id),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    name: str = Form(..., min_length=1, max_length=128),
    chunk_size: int = Form(default=1000, ge=100, le=10000),
    chunk_overlap: int = Form(default=200, ge=0, le=5000),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    ctx = get_tenant_context()
    source_id = str(uuid.uuid4())
    short_id = source_id.replace("-", "")[:12]
    meili_index = f"nimrag_{ctx.tenant_id}_{short_id}"[:64]

    source_dir = DATA_DIR / ctx.tenant_id / ctx.workspace_id / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    file_path = source_dir / Path(file.filename or "upload.bin").name
    await stream_upload_to_path(file, file_path, max_bytes=MAX_FILE_SIZE)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO lexical_rag_sources
               (id, tenant_id, workspace_id, name, source_type, file_path, meili_index,
                chunk_size, chunk_overlap, status, doc_count)
               VALUES (?, ?, ?, ?, 'file', ?, ?, ?, ?, 'indexing', 0)""",
            (
                source_id,
                ctx.tenant_id,
                ctx.workspace_id,
                name,
                str(file_path),
                meili_index,
                chunk_size,
                chunk_overlap,
            ),
        )
        conn.commit()

    try:
        doc_count = await work_pool.submit_background(
            lambda: _index_into_meilisearch_async(
                str(file_path),
                meili_index,
                source_id,
                chunk_size,
                chunk_overlap,
                ctx=ctx,
            ),
            job_type="lexical_rag_index",
            ctx=ctx,
            persist=True,
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE lexical_rag_sources SET status='ready', doc_count=? WHERE id=? AND tenant_id=?",
                (doc_count, source_id, ctx.tenant_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Lexical indexing failed for source %s", source_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE lexical_rag_sources SET status='error' WHERE id=? AND tenant_id=?",
                (source_id, ctx.tenant_id),
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lexical_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
    return dict(row)


@router.delete("/{source_id}")
async def delete_source(source_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, meili_index FROM lexical_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        meili_index = row["meili_index"]
        conn.execute(
            "DELETE FROM lexical_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        )
        conn.commit()

    try:
        client = _get_meili_client()
        await asyncio.to_thread(client.delete_index, meili_index)
    except Exception:
        logger.warning("Failed to delete Meilisearch index %s", meili_index)

    source_dir = DATA_DIR / ctx.tenant_id / ctx.workspace_id / source_id
    if source_dir.exists():
        await asyncio.to_thread(shutil.rmtree, source_dir, True)

    return {"ok": True}


@router.post("/{source_id}/query")
async def query_source(source_id: str, body: LexicalQueryRequest):
    ctx = get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lexical_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Source not ready")

    client = _get_meili_client()
    index = client.index(row["meili_index"])
    filter_expr = build_meili_tenant_filter(ctx, source_id=source_id)
    results = await asyncio.to_thread(
        index.search,
        body.query,
        {"limit": body.k, "filter": filter_expr},
    )

    return {
        "results": [
            {
                "content": hit.get("content", ""),
                "metadata": hit.get("metadata", {}),
            }
            for hit in results.get("hits", [])
        ]
    }


def _prepare_chunks(
    file_path: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
    ctx: TenantContext,
) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    documents = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        documents.append(
            {
                "id": f"{ctx.tenant_id}_{ctx.workspace_id}_{source_id}_{i}",
                "tenant_id": ctx.tenant_id,
                "workspace_id": ctx.workspace_id,
                "source_id": source_id,
                "chunk_id": i,
                "content": chunk.page_content,
                "metadata": json.dumps(meta) if isinstance(meta, dict) else str(meta),
            }
        )
    return documents


async def _index_into_meilisearch_async(
    file_path: str,
    meili_index: str,
    source_id: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    ctx: TenantContext,
) -> int:
    documents = await asyncio.to_thread(
        _prepare_chunks, file_path, source_id, chunk_size, chunk_overlap, ctx
    )
    if not documents:
        return 0

    client = _get_meili_client()
    index = client.index(meili_index)

    filter_task = await asyncio.to_thread(
        index.update_filterable_attributes,
        ["tenant_id", "workspace_id", "source_id"],
    )
    await _wait_for_task_async(client, filter_task.task_uid)
    await asyncio.to_thread(index.update_searchable_attributes, ["content"])
    await asyncio.to_thread(
        index.update_displayed_attributes,
        ["content", "metadata", "chunk_id", "tenant_id", "workspace_id", "source_id"],
    )

    batch_size = 500
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        task = await asyncio.to_thread(index.add_documents, batch, "id")
        await _wait_for_task_async(client, task.task_uid)

    return len(documents)


async def _wait_for_task_async(client, task_uid: int, timeout_ms: int = 30000):
    """Non-blocking async poll for Meilisearch task completion (never blocks asyncio loop)."""
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        task = await asyncio.to_thread(client.get_task, task_uid)
        if task.status in ("succeeded", "failed"):
            if task.status == "failed":
                raise RuntimeError(f"Meilisearch task {task_uid} failed: {task.error}")
            return task
        await asyncio.sleep(0.25)
    raise TimeoutError(f"Meilisearch task {task_uid} timed out after {timeout_ms}ms")

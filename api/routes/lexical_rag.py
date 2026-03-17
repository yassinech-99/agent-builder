"""Lexical RAG endpoints — Meilisearch-backed keyword/BM25 search."""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from api.db import get_db
from api.utils.doc_loader import load_documents, ALLOWED_EXTENSIONS, MAX_FILE_SIZE

logger = logging.getLogger("novagrid.api.lexical_rag")
router = APIRouter(prefix="/api/lexical-rag", tags=["lexical-rag"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "lexical_rag"
MEILI_URL = os.getenv("MEILI_URL", "http://172.30.6.3:7700")


def _get_meili_client():
    import meilisearch
    return meilisearch.Client(MEILI_URL)


class LexicalQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)


@router.get("")
def list_sources():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM lexical_rag_sources ORDER BY created_at DESC"
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
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    source_id = str(uuid.uuid4())
    short_id = source_id.replace("-", "")[:12]
    meili_index = f"lexrag_{short_id}"

    source_dir = DATA_DIR / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    file_path = source_dir / (file.filename or "upload.bin")
    file_path.write_bytes(content)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO lexical_rag_sources
               (id, name, source_type, file_path, meili_index,
                chunk_size, chunk_overlap, status, doc_count)
               VALUES (?, ?, 'file', ?, ?, ?, ?, 'indexing', 0)""",
            (source_id, name, str(file_path), meili_index, chunk_size, chunk_overlap),
        )
        conn.commit()

    try:
        doc_count = _index_into_meilisearch(
            str(file_path), meili_index, chunk_size, chunk_overlap
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE lexical_rag_sources SET status='ready', doc_count=? WHERE id=?",
                (doc_count, source_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Lexical indexing failed for source %s", source_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE lexical_rag_sources SET status='error' WHERE id=?",
                (source_id,),
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lexical_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    return dict(row)


@router.delete("/{source_id}")
def delete_source(source_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, meili_index FROM lexical_rag_sources WHERE id=?",
            (source_id,),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        meili_index = row["meili_index"]
        conn.execute("DELETE FROM lexical_rag_sources WHERE id=?", (source_id,))
        conn.commit()

    try:
        client = _get_meili_client()
        client.delete_index(meili_index)
    except Exception:
        logger.warning("Failed to delete Meilisearch index %s", meili_index)

    source_dir = DATA_DIR / source_id
    if source_dir.exists():
        shutil.rmtree(source_dir, ignore_errors=True)

    return {"ok": True}


@router.post("/{source_id}/query")
def query_source(source_id: str, body: LexicalQueryRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM lexical_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Source not ready")

    client = _get_meili_client()
    index = client.index(row["meili_index"])
    results = index.search(body.query, {"limit": body.k})

    return {
        "results": [
            {
                "content": hit.get("content", ""),
                "metadata": hit.get("metadata", {}),
            }
            for hit in results.get("hits", [])
        ]
    }


def _index_into_meilisearch(
    file_path: str,
    meili_index: str,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs = load_documents(file_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    client = _get_meili_client()
    index = client.index(meili_index)

    index.update_searchable_attributes(["content"])
    index.update_displayed_attributes(["content", "metadata", "chunk_id"])

    documents = []
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        documents.append({
            "id": str(i),
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": json.dumps(meta) if isinstance(meta, dict) else str(meta),
        })

    batch_size = 500
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        task = index.add_documents(batch, primary_key="id")
        _wait_for_task(client, task.task_uid)

    return len(chunks)


def _wait_for_task(client, task_uid: int, timeout_ms: int = 30000):
    """Poll Meilisearch until task completes or fails."""
    import time
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        task = client.get_task(task_uid)
        if task.status in ("succeeded", "failed"):
            if task.status == "failed":
                raise RuntimeError(f"Meilisearch task {task_uid} failed: {task.error}")
            return task
        time.sleep(0.3)
    raise TimeoutError(f"Meilisearch task {task_uid} timed out after {timeout_ms}ms")

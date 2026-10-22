"""Multi-Tenant Vector RAG endpoints — Qdrant (`is_tenant=True` + `INT8`) + Dynamic GPU Micro-Batcher + NVIDIARerank."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from api.core.tenancy import (
    TenantContext,
    get_tenant_context,
    scoped_object_prefix,
    stream_upload_to_path,
)
from api.core.object_store import build_object_store_client_for_bucket
from api.core.vector_store import vector_store
from api.db import get_db
from api.services.batcher import gpu_batcher
from api.utils.doc_loader import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, load_documents
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.api.local_rag")
router = APIRouter(prefix="/api/local-rag", tags=["local-rag"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "local_rag"

EMBEDDING_MODELS: dict[str, dict[str, str]] = {
    "nvidia/llama-nemotron-embed-1b-v2": {
        "provider": "nvidia_nim",
        "label": "NVIDIA Llama Nemotron Embed 1B v2",
    },
    "nvidia/nv-embedqa-e5-v5": {
        "provider": "nvidia_nim",
        "label": "NVIDIA NV-EmbedQA E5 v5",
    },
    "local/all-MiniLM-L6-v2": {
        "provider": "local_hf",
        "label": "Local SentenceTransformers (all-MiniLM-L6-v2, Offline)",
    },
    "text-embedding-3-small": {
        "provider": "openai",
        "label": "OpenAI text-embedding-3-small",
    },
}

DEFAULT_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-1b-v2"

# Cache heavy embedding model wrappers per process so SentenceTransformers / HTTP pools initialize once
_EMBEDDER_CACHE: dict[str, Any] = {}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)
    rerank: bool = True


class BucketImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    bucket_name: str = Field(..., min_length=1, max_length=256)
    prefix: str = Field(default="", max_length=1024)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=5000)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, max_length=128)


def _get_embeddings(embedding_model: str = "") -> Any:
    model_key = (embedding_model or DEFAULT_EMBEDDING_MODEL).strip()
    cached = _EMBEDDER_CACHE.get(model_key)
    if cached is not None:
        return cached

    if model_key.startswith("local/"):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        hf_name = model_key.split("local/", 1)[1] or "all-MiniLM-L6-v2"
        emb = HuggingFaceEmbeddings(model_name=hf_name)
        _EMBEDDER_CACHE[model_key] = emb
        return emb

    if model_key.startswith("text-embedding-"):
        from langchain_openai import OpenAIEmbeddings

        emb = OpenAIEmbeddings(model=model_key)
        _EMBEDDER_CACHE[model_key] = emb
        return emb

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    kwargs: dict[str, Any] = {
        "model": model_key,
        "truncate": "END",
    }
    base_url = os.getenv("NVIDIA_EMBED_BASE_URL") or os.getenv("NIM_BASE_URL")
    api_key = os.getenv("NVIDIA_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    emb = NVIDIAEmbeddings(**kwargs)
    _EMBEDDER_CACHE[model_key] = emb
    return emb


def _maybe_rerank_documents(docs: list[Document], query: str, top_n: int) -> list[Document]:
    """Apply NVIDIA cross-encoder reranking when credentials/endpoints are available."""
    if len(docs) <= 1:
        return docs
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    rerank_url = os.getenv("NVIDIA_RERANK_BASE_URL", "").strip()
    if not api_key and not rerank_url:
        return docs[:top_n]

    try:
        from langchain_nvidia_ai_endpoints import NVIDIARerank

        kwargs: dict[str, Any] = {
            "model": os.getenv("NVIDIA_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2"),
            "top_n": top_n,
            "truncate": "END",
        }
        if rerank_url:
            kwargs["base_url"] = rerank_url
        if api_key:
            kwargs["api_key"] = api_key
        reranker = NVIDIARerank(**kwargs)
        return list(reranker.compress_documents(documents=docs, query=query))[:top_n]
    except Exception as exc:
        logger.warning("NVIDIARerank fallback to dense similarity ranking: %s", exc)
        return docs[:top_n]


async def search_local_source_async(
    row: dict[str, Any],
    query: str,
    k: int = 4,
    rerank: bool = True,
    ctx: TenantContext | None = None,
) -> list[dict[str, Any]]:
    """Query tenant-partitioned Qdrant vectors via dynamic GPU micro-batcher + optional NVIDIA Rerank."""
    active = ctx or get_tenant_context()
    model_key = (row.get("embedding_model") or DEFAULT_EMBEDDING_MODEL).strip()
    embeddings = _get_embeddings(model_key)

    query_vec = await gpu_batcher.embed_text(
        model_key,
        query,
        embeddings.embed_documents,
    )
    fetch_k = max(k * 3, 10) if rerank else k
    hits = await vector_store.search(
        source_id=row["id"],
        query_vector=query_vec,
        k=fetch_k,
        ctx=active,
    )
    if not hits:
        return []

    if rerank:
        candidate_docs = [
            Document(page_content=h["page_content"], metadata=h.get("metadata") or {})
            for h in hits
        ]
        reranked = await asyncio.to_thread(
            _maybe_rerank_documents, candidate_docs, query, k
        )
        return [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in reranked[:k]
        ]

    return [
        {"page_content": h["page_content"], "metadata": h.get("metadata") or {}}
        for h in hits[:k]
    ]


def _split_docs_sync(docs: list[Any], chunk_size: int, chunk_overlap: int) -> list[Any]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


async def _index_documents_qdrant(
    *,
    source_id: str,
    docs: list[Any],
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
    ctx: TenantContext | None = None,
) -> tuple[int, int]:
    """Split documents on worker thread, batch-embed on GPU, and upsert into Qdrant `is_tenant=True` collection."""
    active = ctx or get_tenant_context()
    chunks = await asyncio.to_thread(_split_docs_sync, docs, chunk_size, chunk_overlap)
    if not chunks:
        return 0, 0

    model_key = (embedding_model or DEFAULT_EMBEDDING_MODEL).strip()
    embeddings = _get_embeddings(model_key)
    texts = [c.page_content for c in chunks]
    metas = [c.metadata or {} for c in chunks]

    vectors = await gpu_batcher.embed_documents(
        model_key,
        texts,
        embeddings.embed_documents,
    )
    dim = len(vectors[0]) if vectors else 0
    count = await vector_store.upsert_chunks(
        source_id=source_id,
        vectors=vectors,
        texts=texts,
        metadatas=metas,
        ctx=active,
    )
    return count, dim


async def _index_file_qdrant(
    *,
    source_id: str,
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
    ctx: TenantContext | None = None,
) -> tuple[int, int]:
    docs = await asyncio.to_thread(load_documents, file_path)
    return await _index_documents_qdrant(
        source_id=source_id,
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        ctx=ctx,
    )


async def _import_and_index_bucket_qdrant(
    *,
    source_id: str,
    bucket_name: str,
    prefix: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    ctx: TenantContext | None = None,
) -> tuple[int, int]:
    active = ctx or get_tenant_context()
    enforced_prefix = scoped_object_prefix(prefix, active)
    client = build_object_store_client_for_bucket(bucket_name, ctx=active)

    def _download_bucket_docs() -> list[Any]:
        all_docs: list[Any] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            objects = list(client.list_objects(bucket_name, prefix=enforced_prefix, recursive=True))
            for obj in objects:
                if obj.is_dir:
                    continue
                ext = Path(obj.object_name).suffix.lower()
                if ext not in ALLOWED_EXTENSIONS:
                    continue
                local_target = Path(tmpdir) / Path(obj.object_name).name
                client.fget_object(bucket_name, obj.object_name, str(local_target))
                loaded = load_documents(str(local_target))
                for d in loaded:
                    d.metadata["storage_uri"] = f"s3://{bucket_name}/{obj.object_name}"
                all_docs.extend(loaded)
        if not all_docs:
            raise ValueError(
                f"No supported documents found under tenant prefix s3://{bucket_name}/{enforced_prefix}"
            )
        return all_docs

    all_docs = await asyncio.to_thread(_download_bucket_docs)
    return await _index_documents_qdrant(
        source_id=source_id,
        docs=all_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        ctx=active,
    )


@router.get("/embedding-models")
def list_embedding_models():
    return [
        {"value": model_id, **meta}
        for model_id, meta in EMBEDDING_MODELS.items()
    ]


@router.get("")
def list_sources():
    ctx = get_tenant_context()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM local_rag_sources
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
    embedding_model: str = Form(default=DEFAULT_EMBEDDING_MODEL),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    ctx = get_tenant_context()
    source_id = str(uuid.uuid4())
    source_dir = DATA_DIR / ctx.tenant_id / ctx.workspace_id / source_id
    await asyncio.to_thread(lambda: source_dir.mkdir(parents=True, exist_ok=True))

    file_path = source_dir / Path(file.filename or "upload.bin").name
    await stream_upload_to_path(file, file_path, max_bytes=MAX_FILE_SIZE)

    def _insert_initial() -> None:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO local_rag_sources
                   (id, tenant_id, workspace_id, name, source_type, file_path, chunk_size, chunk_overlap,
                    index_path, embedding_model, status, doc_count)
                   VALUES (?, ?, ?, ?, 'file', ?, ?, ?, ?, ?, 'indexing', 0)""",
                (
                    source_id,
                    ctx.tenant_id,
                    ctx.workspace_id,
                    name,
                    str(file_path),
                    chunk_size,
                    chunk_overlap,
                    f"qdrant://{source_id}",
                    embedding_model,
                ),
            )
            conn.commit()

    await asyncio.to_thread(_insert_initial)

    try:
        doc_count, vec_dim = await work_pool.submit_background(
            lambda: _index_file_qdrant(
                source_id=source_id,
                file_path=str(file_path),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                ctx=ctx,
            ),
            job_type="local_rag_index",
            ctx=ctx,
            persist=True,
        )

        def _mark_ready() -> dict[str, Any]:
            with get_db() as conn:
                conn.execute(
                    "UPDATE local_rag_sources SET status='ready', doc_count=?, vector_dim=? WHERE id=? AND tenant_id=?",
                    (doc_count, vec_dim, source_id, ctx.tenant_id),
                )
                conn.commit()
                saved = conn.execute(
                    "SELECT * FROM local_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
                    (source_id, ctx.tenant_id, ctx.workspace_id),
                ).fetchone()
            return dict(saved)

        return await asyncio.to_thread(_mark_ready)
    except Exception as exc:
        logger.exception("Qdrant vector indexing failed for source %s", source_id)

        def _mark_error() -> None:
            with get_db() as conn:
                conn.execute(
                    "UPDATE local_rag_sources SET status='error' WHERE id=? AND tenant_id=?",
                    (source_id, ctx.tenant_id),
                )
                conn.commit()

        await asyncio.to_thread(_mark_error)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/bucket-import")
async def bucket_import(body: BucketImportRequest):
    """Import documents from tenant-scoped RustFS / S3 object prefix into Qdrant on Lane P1."""
    ctx = get_tenant_context()
    source_id = str(uuid.uuid4())
    enforced_prefix = scoped_object_prefix(body.prefix, ctx)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO local_rag_sources
               (id, tenant_id, workspace_id, name, source_type, file_path, chunk_size, chunk_overlap,
                index_path, embedding_model, status, doc_count, storage_bucket, storage_prefix)
               VALUES (?, ?, ?, ?, 'bucket', '', ?, ?, ?, ?, 'indexing', 0, ?, ?)""",
            (
                source_id,
                ctx.tenant_id,
                ctx.workspace_id,
                body.name,
                body.chunk_size,
                body.chunk_overlap,
                f"qdrant://{source_id}",
                body.embedding_model,
                body.bucket_name,
                enforced_prefix,
            ),
        )
        conn.commit()

    try:
        doc_count, vec_dim = await work_pool.submit_background(
            lambda: _import_and_index_bucket_qdrant(
                source_id=source_id,
                bucket_name=body.bucket_name,
                prefix=body.prefix,
                chunk_size=body.chunk_size,
                chunk_overlap=body.chunk_overlap,
                embedding_model=body.embedding_model,
                ctx=ctx,
            ),
            job_type="bucket_rag_import",
            ctx=ctx,
            persist=True,
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='ready', doc_count=?, vector_dim=? WHERE id=? AND tenant_id=?",
                (doc_count, vec_dim, source_id, ctx.tenant_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Object bucket import failed for source %s", source_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='error' WHERE id=? AND tenant_id=?",
                (source_id, ctx.tenant_id),
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM local_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
    return dict(row)


@router.delete("/{source_id}")
async def delete_source(source_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, vector_dim FROM local_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        vec_dim = int(row["vector_dim"] or 0) or None
        conn.execute(
            "DELETE FROM local_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
            (source_id, ctx.tenant_id, ctx.workspace_id),
        )
        conn.commit()

    await vector_store.delete_source(source_id=source_id, dim=vec_dim, ctx=ctx)

    source_dir = DATA_DIR / ctx.tenant_id / ctx.workspace_id / source_id
    if source_dir.exists():
        await asyncio.to_thread(shutil.rmtree, source_dir, True)

    return {"ok": True}


@router.post("/{source_id}/query")
async def query_source(source_id: str, body: QueryRequest):
    ctx = get_tenant_context()

    def _fetch_source() -> dict[str, Any] | None:
        with get_db() as conn:
            found = conn.execute(
                "SELECT * FROM local_rag_sources WHERE id=? AND tenant_id=? AND workspace_id=?",
                (source_id, ctx.tenant_id, ctx.workspace_id),
            ).fetchone()
        return dict(found) if found else None

    row = await asyncio.to_thread(_fetch_source)
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Source not ready")

    results = await search_local_source_async(
        row,
        body.query,
        body.k,
        body.rerank,
        ctx=ctx,
    )
    return {"results": results}

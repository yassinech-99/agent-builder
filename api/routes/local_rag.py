"""Vector RAG endpoints — NVIDIA NIM Embeddings, Local SentenceTransformers, OpenAI Embeddings + NVIDIARerank."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api.db import get_db
from api.routes.storage import build_minio_client_for_bucket
from api.utils.doc_loader import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, load_documents

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

    if model_key.startswith("local/"):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        hf_name = model_key.split("local/", 1)[1] or "all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=hf_name)

    if model_key.startswith("text-embedding-"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model_key)

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
    return NVIDIAEmbeddings(**kwargs)


def _maybe_rerank_documents(docs: list[Any], query: str, top_n: int) -> list[Any]:
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


def search_local_source_row(
    row: dict[str, Any],
    query: str,
    k: int = 4,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Query a local vector index with candidate over-fetching + optional NVIDIA Rerank."""
    from langchain_community.vectorstores import FAISS

    embeddings = _get_embeddings(row.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
    store = FAISS.load_local(
        row["index_path"],
        embeddings,
        allow_dangerous_deserialization=True,
    )
    fetch_k = max(k * 3, 10) if rerank else k
    docs = store.similarity_search(query, k=fetch_k)
    if rerank:
        docs = _maybe_rerank_documents(docs, query=query, top_n=k)
    return [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs[:k]
    ]


def _index_documents(
    docs: list[Any],
    index_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
) -> int:
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return 0

    embeddings = _get_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(index_path)
    return len(chunks)


def _index_file(
    file_path: str,
    index_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str = "",
) -> int:
    docs = load_documents(file_path)
    return _index_documents(docs, index_path, chunk_size, chunk_overlap, embedding_model)


def _import_and_index_bucket(
    bucket_name: str,
    prefix: str,
    index_path: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
) -> int:
    client = build_minio_client_for_bucket(bucket_name)
    all_docs: list[Any] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        objects = client.list_objects(bucket_name, prefix=prefix or None, recursive=True)
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
        raise ValueError(f"No supported documents found in s3://{bucket_name}/{prefix}")

    return _index_documents(
        all_docs,
        index_path=index_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
    )


@router.get("/embedding-models")
def list_embedding_models():
    return [
        {"value": model_id, **meta}
        for model_id, meta in EMBEDDING_MODELS.items()
    ]


@router.get("")
def list_sources():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM local_rag_sources ORDER BY created_at DESC"
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

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File exceeds 50 MB limit")

    source_id = str(uuid.uuid4())
    source_dir = DATA_DIR / source_id
    source_dir.mkdir(parents=True, exist_ok=True)

    file_path = source_dir / (file.filename or "upload.bin")
    file_path.write_bytes(content)

    with get_db() as conn:
        conn.execute(
            """INSERT INTO local_rag_sources
               (id, name, source_type, file_path, chunk_size, chunk_overlap,
                index_path, embedding_model, status, doc_count)
               VALUES (?, ?, 'file', ?, ?, ?, ?, ?, 'indexing', 0)""",
            (
                source_id,
                name,
                str(file_path),
                chunk_size,
                chunk_overlap,
                str(source_dir / "index"),
                embedding_model,
            ),
        )
        conn.commit()

    try:
        doc_count = await asyncio.to_thread(
            _index_file,
            str(file_path),
            str(source_dir / "index"),
            chunk_size,
            chunk_overlap,
            embedding_model,
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='ready', doc_count=? WHERE id=?",
                (doc_count, source_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Indexing failed for local source %s", source_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='error' WHERE id=?", (source_id,)
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM local_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    return dict(row)


@router.post("/bucket-import")
async def bucket_import(body: BucketImportRequest):
    """Import documents from an S3-compatible object bucket (MinIO / R2 / Ceph / S3) and index vectors."""
    source_id = str(uuid.uuid4())
    source_dir = DATA_DIR / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(source_dir / "index")

    with get_db() as conn:
        conn.execute(
            """INSERT INTO local_rag_sources
               (id, name, source_type, file_path, chunk_size, chunk_overlap,
                index_path, embedding_model, status, doc_count, storage_bucket, storage_prefix)
               VALUES (?, ?, 'bucket', '', ?, ?, ?, ?, 'indexing', 0, ?, ?)""",
            (
                source_id,
                body.name,
                body.chunk_size,
                body.chunk_overlap,
                index_path,
                body.embedding_model,
                body.bucket_name,
                body.prefix,
            ),
        )
        conn.commit()

    try:
        doc_count = await asyncio.to_thread(
            _import_and_index_bucket,
            body.bucket_name,
            body.prefix,
            index_path,
            body.chunk_size,
            body.chunk_overlap,
            body.embedding_model,
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='ready', doc_count=? WHERE id=?",
                (doc_count, source_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("Object bucket import failed for source %s", source_id)
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='error' WHERE id=?", (source_id,)
            )
            conn.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM local_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    return dict(row)


@router.delete("/{source_id}")
def delete_source(source_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM local_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        conn.execute("DELETE FROM local_rag_sources WHERE id=?", (source_id,))
        conn.commit()

    source_dir = DATA_DIR / source_id
    if source_dir.exists():
        shutil.rmtree(source_dir, ignore_errors=True)

    return {"ok": True}


@router.post("/{source_id}/query")
async def query_source(source_id: str, body: QueryRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM local_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Source not ready")

    results = await asyncio.to_thread(
        search_local_source_row,
        dict(row),
        body.query,
        body.k,
        body.rerank,
    )
    return {"results": results}

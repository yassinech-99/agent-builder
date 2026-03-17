"""Local RAG endpoints — FAISS + Bedrock Embeddings."""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from api.db import get_db
from api.utils.doc_loader import load_documents, ALLOWED_EXTENSIONS, MAX_FILE_SIZE

logger = logging.getLogger("novagrid.api.local_rag")
router = APIRouter(prefix="/api/local-rag", tags=["local-rag"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "local_rag"

# For the hackathon demo, we normalize on a single known-good embedding model
# to avoid Bedrock model-id issues. Titan text embeddings v2 are broadly
# available and reliable, so we use that as the default.
EMBEDDING_MODELS = {
    "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
}


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=20)


class S3ImportRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    bucket_name: str = Field(..., min_length=1, max_length=256)
    prefix: str = Field(default="", max_length=1024)
    aws_region: str = Field(default="", max_length=64)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=5000)
    embedding_model: str = Field(default="titan-embed-text-v2", max_length=64)


def _get_embeddings(embedding_model: str = ""):
    model_key = embedding_model or "titan-embed-text-v2"
    model_id = EMBEDDING_MODELS.get(model_key, EMBEDDING_MODELS["titan-embed-text-v2"])

    from langchain_aws import BedrockEmbeddings

    return BedrockEmbeddings(
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        model_id=model_id,
    )


def _index_documents(
    docs: list,
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
    embedding_model: str = Form(default="titan-embed-text-v2"),
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
        doc_count = _index_file(
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


@router.post("/s3-import")
def s3_import(body: S3ImportRequest):
    """Import documents from an S3 bucket and index them locally with FAISS."""
    region = body.aws_region or os.getenv("AWS_REGION", "us-east-1")

    source_id = str(uuid.uuid4())
    source_dir = DATA_DIR / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    index_path = str(source_dir / "index")

    with get_db() as conn:
        conn.execute(
            """INSERT INTO local_rag_sources
               (id, name, source_type, file_path, chunk_size, chunk_overlap,
                index_path, embedding_model, status, doc_count, s3_bucket, s3_prefix)
               VALUES (?, ?, 's3', '', ?, ?, ?, ?, 'indexing', 0, ?, ?)""",
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
        from langchain_community.document_loaders import S3DirectoryLoader

        loader = S3DirectoryLoader(
            body.bucket_name,
            prefix=body.prefix,
            region_name=region,
        )
        docs = loader.load()
        if not docs:
            raise ValueError("No documents found in the specified S3 location")

        doc_count = _index_documents(
            docs, index_path, body.chunk_size, body.chunk_overlap, body.embedding_model
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE local_rag_sources SET status='ready', doc_count=? WHERE id=?",
                (doc_count, source_id),
            )
            conn.commit()
    except Exception as exc:
        logger.exception("S3 import failed for local source %s", source_id)
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
def query_source(source_id: str, body: QueryRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM local_rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")
    if row["status"] != "ready":
        raise HTTPException(status_code=400, detail="Source not ready")

    embeddings = _get_embeddings(row["embedding_model"])
    from langchain_community.vectorstores import FAISS

    store = FAISS.load_local(
        row["index_path"], embeddings, allow_dangerous_deserialization=True
    )
    docs = store.similarity_search(body.query, k=body.k)
    return {
        "results": [
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ]
    }

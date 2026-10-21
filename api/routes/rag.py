"""External / NVIDIA NIM RAG knowledge collection management endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import uuid
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from agent.model import get_model
from api.db import get_db, mask_secret
from api.routes.storage import build_minio_client_for_bucket

logger = logging.getLogger("nimagents.api.rag")
router = APIRouter(prefix="/api/rag", tags=["rag"])

DEFAULT_NIM_RAG_URL = os.getenv("NIM_RAG_BASE_URL", "http://localhost:8081/v1")
DEFAULT_RERANK_MODEL = os.getenv(
    "NVIDIA_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2"
)


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)
    endpoint_url: str = Field(default=DEFAULT_NIM_RAG_URL, min_length=1, max_length=512)
    collection_name: str = Field(..., min_length=1, max_length=128)
    api_key: str = Field(default="", max_length=512)
    ranking_model: str = Field(default=DEFAULT_RERANK_MODEL, max_length=128)
    storage_bucket: str = Field(default="", max_length=256)
    storage_prefix: str = Field(default="", max_length=1024)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class GenerateMetadataRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=4000)
    filename: str = Field(default="", max_length=256)


def _serialize_rag_source(row: Any) -> dict[str, Any]:
    data = dict(row)
    raw_key = data.pop("api_key", "") or ""
    return {
        **data,
        "api_key": "",
        "api_key_masked": mask_secret(raw_key),
        "has_api_key": bool(raw_key.strip()),
    }


def query_nim_rag_row(row: dict[str, Any], query: str, k: int = 5) -> list[dict[str, Any]]:
    """Query an NVIDIA NIM RAG Retriever endpoint using `NVIDIARAGRetriever`."""
    from langchain_nvidia_ai_endpoints import NVIDIARAGRetriever

    api_key = (row.get("api_key") or os.getenv("NVIDIA_API_KEY") or "").strip()
    kwargs: dict[str, Any] = {
        "base_url": row.get("endpoint_url") or DEFAULT_NIM_RAG_URL,
        "collection_names": [row.get("collection_name") or "default"],
        "k": k,
        "enable_reranker": bool(row.get("ranking_model")),
        "enable_citations": True,
    }
    if row.get("ranking_model"):
        kwargs["reranker_model"] = row["ranking_model"]
    if api_key:
        kwargs["api_key"] = api_key

    retriever = NVIDIARAGRetriever(**kwargs)
    docs = retriever.invoke(query)
    return [
        {
            "page_content": d.page_content,
            "metadata": d.metadata,
            "score": d.metadata.get("relevance_score"),
        }
        for d in docs
    ]


@router.get("")
def list_sources():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM rag_sources ORDER BY created_at DESC"
        ).fetchall()
    return [_serialize_rag_source(r) for r in rows]


@router.post("/register", status_code=201)
def register_knowledge_base(body: RegisterRequest):
    """Register an NVIDIA NIM RAG endpoint & collection."""
    source_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO rag_sources
               (id, name, description, endpoint_url, collection_name, api_key,
                ranking_model, storage_bucket, storage_prefix, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready')""",
            (
                source_id,
                body.name.strip(),
                body.description.strip(),
                body.endpoint_url.strip(),
                body.collection_name.strip(),
                body.api_key.strip(),
                body.ranking_model.strip(),
                body.storage_bucket.strip(),
                body.storage_prefix.strip(),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    return _serialize_rag_source(row)


@router.delete("/{source_id}")
def delete_source(source_id: str):
    with get_db() as conn:
        row = conn.execute(
            "SELECT id FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Source not found")
        conn.execute("DELETE FROM rag_sources WHERE id=?", (source_id,))
        conn.commit()
    return {"ok": True}


@router.post("/{source_id}/query")
async def query_source(source_id: str, body: QueryRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    try:
        results = await asyncio.to_thread(
            query_nim_rag_row, dict(row), body.query, body.k
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"NIM RAG retrieve failed: {exc}",
        ) from exc

    return {"results": results}


@router.post("/{source_id}/upload-data")
async def upload_data_to_kb(
    source_id: str,
    file: UploadFile = File(...),
    metadata_json: str = Form(default=""),
    description: str = Form(default=""),
):
    """Upload a document (+ optional metadata sidecar) to the configured S3-compatible bucket."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    bucket = row["storage_bucket"]
    prefix = row["storage_prefix"] or ""
    if not bucket:
        raise HTTPException(
            status_code=400,
            detail="No object storage bucket configured for this knowledge collection.",
        )

    filename = file.filename or "upload.bin"
    content = await file.read()

    meta = metadata_json.strip()
    if not meta and description.strip():
        meta = await _generate_metadata_json(description, filename)

    object_key = f"{prefix}/{filename}".lstrip("/") if prefix else filename
    minio_client = build_minio_client_for_bucket(bucket)

    def _upload_objects() -> bool:
        minio_client.put_object(
            bucket_name=bucket,
            object_name=object_key,
            data=io.BytesIO(content),
            length=len(content),
        )
        uploaded_meta = False
        if meta:
            try:
                parsed = json.loads(meta)
                payload = json.dumps(parsed, indent=2).encode("utf-8")
                minio_client.put_object(
                    bucket_name=bucket,
                    object_name=f"{object_key}.metadata.json",
                    data=io.BytesIO(payload),
                    length=len(payload),
                    content_type="application/json",
                )
                uploaded_meta = True
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON for %s, skipping metadata sidecar", filename)
        return uploaded_meta

    try:
        meta_uploaded = await asyncio.to_thread(_upload_objects)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Object storage upload failed: {exc}") from exc

    return {
        "ok": True,
        "storage_uri": f"s3://{bucket}/{object_key}",
        "metadata_uploaded": meta_uploaded,
    }


@router.post("/generate-metadata")
async def generate_metadata(body: GenerateMetadataRequest):
    """Generate structured metadata JSON using the configured fast_router / default model."""
    metadata = await _generate_metadata_json(body.description, body.filename)
    return {"metadata_json": metadata}


async def _generate_metadata_json(description: str, filename: str) -> str:
    try:
        llm = get_model(role="fast_router")
        prompt = (
            "Generate a clean JSON metadata object for document indexing in nimAgents. "
            'Format: {"metadataAttributes": {"category": "...", "topic": "...", "document_type": "...", "language": "..."}}. '
            f"Filename: {filename}\n"
            f"Content description: {description}\n\n"
            "Return ONLY valid JSON without markdown fences."
        )
        response = await llm.ainvoke(prompt)
        raw = str(getattr(response, "content", response)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        json.loads(raw)
        return raw
    except Exception as exc:
        logger.warning("Metadata generation fell back to structured defaults: %s", exc)
        return json.dumps(
            {
                "metadataAttributes": {
                    "description": description,
                    "filename": filename,
                }
            },
            indent=2,
        )

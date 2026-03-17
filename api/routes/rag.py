"""RAG source management endpoints — Bedrock Knowledge Bases."""

from __future__ import annotations

import json
import logging
import os
import uuid

import boto3
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from api.db import get_db

logger = logging.getLogger("novagrid.api.rag")
router = APIRouter(prefix="/api/rag", tags=["rag"])

DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-1")


class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="", max_length=512)
    knowledge_base_id: str = Field(..., min_length=1, max_length=64)
    region: str = Field(default="", max_length=64)
    s3_bucket: str = Field(default="", max_length=256)
    s3_prefix: str = Field(default="", max_length=1024)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)


class GenerateMetadataRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=4000)
    filename: str = Field(default="", max_length=256)


def _get_bedrock_client(region: str = ""):
    return boto3.client(
        "bedrock-agent-runtime",
        region_name=region or DEFAULT_REGION,
    )


def _get_s3_client(region: str = ""):
    return boto3.client("s3", region_name=region or DEFAULT_REGION)


@router.get("")
def list_sources():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM rag_sources ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/register")
def register_knowledge_base(body: RegisterRequest):
    """Register an existing Bedrock Knowledge Base by its ID."""
    region = body.region or DEFAULT_REGION

    try:
        client = _get_bedrock_client(region)
        client.retrieve(
            knowledgeBaseId=body.knowledge_base_id,
            retrievalQuery={"text": "test"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 1}
            },
        )
    except Exception as exc:
        logger.warning("KB validation call failed (non-fatal): %s", exc)

    source_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO rag_sources
               (id, name, description, knowledge_base_id, region, s3_bucket, s3_prefix, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'ready')""",
            (
                source_id,
                body.name,
                body.description,
                body.knowledge_base_id,
                region,
                body.s3_bucket,
                body.s3_prefix,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    return dict(row)


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
def query_source(source_id: str, body: QueryRequest):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    kb_id = row["knowledge_base_id"]
    region = row["region"] or DEFAULT_REGION

    try:
        client = _get_bedrock_client(region)
        response = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": body.query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": body.k}
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bedrock retrieve failed: {exc}")

    results = []
    for r in response.get("retrievalResults", []):
        content = r.get("content", {})
        results.append({
            "page_content": content.get("text", ""),
            "metadata": r.get("metadata", {}),
            "score": r.get("score"),
        })

    return {"results": results}


@router.post("/{source_id}/upload-data")
async def upload_data_to_kb(
    source_id: str,
    file: UploadFile = File(...),
    metadata_json: str = Form(default=""),
    description: str = Form(default=""),
):
    """Upload a file (+ optional metadata) to the KB's S3 data source bucket."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM rag_sources WHERE id=?", (source_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Source not found")

    bucket = row["s3_bucket"]
    prefix = row["s3_prefix"] or ""
    region = row["region"] or DEFAULT_REGION

    if not bucket:
        raise HTTPException(
            status_code=400,
            detail="No S3 bucket configured for this knowledge base. Edit the registration to add one.",
        )

    filename = file.filename or "upload.bin"
    content = await file.read()

    meta = metadata_json.strip()
    if not meta and description.strip():
        meta = _generate_metadata_json(description, filename, region)

    s3_key = f"{prefix}/{filename}".lstrip("/") if prefix else filename
    s3 = _get_s3_client(region)

    try:
        s3.put_object(Bucket=bucket, Key=s3_key, Body=content)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"S3 upload failed: {exc}")

    if meta:
        try:
            parsed = json.loads(meta)
            if "metadataAttributes" not in parsed:
                parsed = {"metadataAttributes": parsed}
            meta_key = f"{s3_key}.metadata.json"
            s3.put_object(
                Bucket=bucket,
                Key=meta_key,
                Body=json.dumps(parsed, indent=2).encode(),
                ContentType="application/json",
            )
        except json.JSONDecodeError:
            logger.warning("Invalid metadata JSON for %s, skipping metadata upload", filename)

    return {
        "ok": True,
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "metadata_uploaded": bool(meta),
    }


@router.post("/generate-metadata")
def generate_metadata(body: GenerateMetadataRequest):
    """Use the default Bedrock model to generate metadata JSON from a description."""
    metadata = _generate_metadata_json(body.description, body.filename, DEFAULT_REGION)
    return {"metadata_json": metadata}


def _generate_metadata_json(description: str, filename: str, region: str) -> str:
    """Call Bedrock LLM to produce a metadataAttributes JSON blob."""
    try:
        model_id = os.getenv("BEDROCK_MODEL", "amazon.nova-2-lite-v1:0")
        client = boto3.client("bedrock-runtime", region_name=region)

        prompt = (
            "Generate a JSON object for Amazon Bedrock Knowledge Base metadata. "
            "The format must be: {\"metadataAttributes\": {\"key\": \"value\", ...}}. "
            "Include relevant attributes like category, topic, document_type, language, etc. "
            f"Filename: {filename}\n"
            f"Content description: {description}\n\n"
            "Return ONLY valid JSON, no explanation."
        )

        response = client.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 1024, "temperature": 0.3},
        )

        raw = response["output"]["message"]["content"][0]["text"].strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        json.loads(raw)
        return raw
    except Exception as exc:
        logger.warning("Metadata generation failed: %s", exc)
        return json.dumps({
            "metadataAttributes": {
                "description": description,
                "filename": filename,
            }
        })

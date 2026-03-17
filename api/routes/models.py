"""Model CRUD endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.db import get_db

logger = logging.getLogger("novagrid.api.models")
router = APIRouter(prefix="/api/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    provider: str = Field(default="bedrock", max_length=64)
    base_url: str = ""
    api_key: str = ""
    model_name: str = Field(..., min_length=1, max_length=128)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=1_000_000)
    is_default: bool = False
    aws_region: str = ""
    aws_secret_key: str = ""


class ModelOut(ModelCreate):
    id: str
    created_at: str


@router.get("", response_model=list[ModelOut])
def list_models():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("", response_model=ModelOut, status_code=201)
def create_model(body: ModelCreate):
    model_id = str(uuid.uuid4())
    with get_db() as conn:
        if body.is_default:
            conn.execute("UPDATE models SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            """INSERT INTO models (id, name, provider, base_url, api_key,
               model_name, temperature, max_tokens, is_default,
               aws_region, aws_secret_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_id,
                body.name,
                body.provider,
                body.base_url,
                body.api_key,
                body.model_name,
                body.temperature,
                body.max_tokens,
                body.is_default,
                body.aws_region,
                body.aws_secret_key,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return dict(row)


@router.get("/{model_id}", response_model=ModelOut)
def get_model(model_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return dict(row)


@router.put("/{model_id}", response_model=ModelOut)
def update_model(model_id: str, body: ModelCreate):
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Model not found")
        if body.is_default:
            conn.execute("UPDATE models SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            """UPDATE models SET name=?, provider=?, base_url=?, api_key=?,
               model_name=?, temperature=?, max_tokens=?, is_default=?,
               aws_region=?, aws_secret_key=?
               WHERE id=?""",
            (
                body.name,
                body.provider,
                body.base_url,
                body.api_key,
                body.model_name,
                body.temperature,
                body.max_tokens,
                body.is_default,
                body.aws_region,
                body.aws_secret_key,
                model_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return dict(row)


@router.delete("/{model_id}")
def delete_model(model_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
    return {"ok": True}

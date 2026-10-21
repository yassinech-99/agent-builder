"""Multi-provider NVIDIA NIM model registry, live endpoint discovery, and health-check endpoints."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.model import (
    DEFAULT_NIM_MODEL,
    MODEL_ROLES,
    NIM_MODEL_PRESETS,
    SUPPORTED_PROVIDERS,
    create_model_from_config,
    resolve_provider_credentials,
)
from api.db import get_db, mask_secret

logger = logging.getLogger("nimagents.api.models")
router = APIRouter(prefix="/api/models", tags=["models"])


class ModelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    provider: str = Field(default="nvidia_nim", max_length=64)
    base_url: str = Field(default="", max_length=512)
    api_key: str = Field(default="", max_length=512)
    model_name: str = Field(default=DEFAULT_NIM_MODEL, min_length=1, max_length=256)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=2_000_000)
    is_default: bool = False
    role: str = Field(default="primary_chat", max_length=64)
    fallback_model_id: str = Field(default="", max_length=128)
    context_window: int = Field(default=131072, ge=1024, le=10_000_000)
    supports_thinking: bool = False
    clear_api_key: bool = False


class ModelOut(BaseModel):
    id: str
    name: str
    provider: str
    base_url: str
    api_key: str = ""
    api_key_masked: str = ""
    has_api_key: bool = False
    model_name: str
    temperature: float
    max_tokens: int | None
    is_default: bool
    role: str
    fallback_model_id: str
    context_window: int
    supports_thinking: bool
    created_at: str


class DiscoverModelsRequest(BaseModel):
    provider: str = Field(default="nvidia_nim", max_length=64)
    base_url: str = Field(default="", max_length=512)
    api_key: str = Field(default="", max_length=512)


class DiscoveredModelItem(BaseModel):
    id: str
    owned_by: str = ""


class ModelTestRequest(BaseModel):
    provider: str = Field(default="nvidia_nim", max_length=64)
    base_url: str = Field(default="", max_length=512)
    api_key: str = Field(default="", max_length=512)
    model_name: str = Field(default=DEFAULT_NIM_MODEL, min_length=1, max_length=256)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


def _serialize_model_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    raw_key = data.pop("api_key", "") or ""
    return {
        **data,
        "api_key": "",
        "api_key_masked": mask_secret(raw_key),
        "has_api_key": bool(raw_key.strip()),
        "is_default": bool(data.get("is_default", 0)),
        "supports_thinking": bool(data.get("supports_thinking", 0)),
        "role": data.get("role") or "primary_chat",
        "fallback_model_id": data.get("fallback_model_id") or "",
        "context_window": int(data.get("context_window") or 131072),
    }


@router.get("/presets")
def list_model_presets():
    """Return supported provider definitions, curated NIM presets, and model roles."""
    return {
        "providers": SUPPORTED_PROVIDERS,
        "presets": NIM_MODEL_PRESETS,
        "roles": list(MODEL_ROLES),
    }


@router.post("/discover", response_model=list[DiscoveredModelItem])
async def discover_models(body: DiscoverModelsRequest):
    """Query live `/models` catalog from NVIDIA NIM, Dynamo, OpenAI, OpenRouter, vLLM, SGLang, or Ollama."""
    from api.app import get_http_client

    provider = body.provider.strip().lower()
    api_key, base_url = resolve_provider_credentials(
        provider, api_key=body.api_key, base_url=body.base_url
    )
    if not base_url:
        raise HTTPException(status_code=400, detail="Base URL is required for model discovery")

    target_url = base_url.rstrip("/")
    if provider == "anthropic":
        target_url = f"{target_url}/v1/models" if not target_url.endswith("/v1") else f"{target_url}/models"
        headers = {
            "Accept": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key
    else:
        target_url = f"{target_url}/models"
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    client = get_http_client()
    try:
        resp = await client.get(target_url, headers=headers, timeout=12.0)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Endpoint returned HTTP {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to endpoint '{target_url}': {exc}",
        ) from exc

    payload = resp.json()
    raw_items = payload.get("data") or payload.get("models") or []
    discovered: list[DiscoveredModelItem] = []
    for item in raw_items:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name") or ""
            if mid:
                discovered.append(
                    DiscoveredModelItem(
                        id=str(mid),
                        owned_by=str(item.get("owned_by") or item.get("ownedBy") or provider),
                    )
                )
        elif isinstance(item, str):
            discovered.append(DiscoveredModelItem(id=item, owned_by=provider))

    discovered.sort(key=lambda m: m.id.lower())
    return discovered


@router.post("/test")
async def test_unsaved_model(body: ModelTestRequest):
    """Run a fast connectivity and latency ping against an unsaved model config."""
    return await _execute_model_ping(body.model_dump())


@router.post("/{model_id}/test")
async def test_saved_model(model_id: str):
    """Run a fast connectivity and latency ping against a saved model record."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return await _execute_model_ping(dict(row))


async def _execute_model_ping(cfg: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        ping_cfg = {**cfg, "max_tokens": 24, "temperature": 0.1}
        llm = create_model_from_config(ping_cfg)
        reply = await llm.ainvoke("Reply with the single word READY.")
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        content = getattr(reply, "content", str(reply))
        text = content if isinstance(content, str) else str(content)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "provider": cfg.get("provider", "nvidia_nim"),
            "model_name": cfg.get("model_name", ""),
            "reply": text[:160],
        }
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return {
            "ok": False,
            "latency_ms": latency_ms,
            "provider": cfg.get("provider", "nvidia_nim"),
            "model_name": cfg.get("model_name", ""),
            "error": str(exc),
        }


@router.get("", response_model=list[ModelOut])
def list_models():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY is_default DESC, created_at DESC"
        ).fetchall()
    return [_serialize_model_row(r) for r in rows]


@router.post("", response_model=ModelOut, status_code=201)
def create_model(body: ModelCreate):
    if body.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{body.provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}",
        )
    if body.role not in MODEL_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model role '{body.role}'. Valid roles: {', '.join(MODEL_ROLES)}",
        )

    model_id = str(uuid.uuid4())
    with get_db() as conn:
        if body.is_default:
            conn.execute("UPDATE models SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            """INSERT INTO models (
                   id, name, provider, base_url, api_key, model_name,
                   temperature, max_tokens, is_default, role,
                   fallback_model_id, context_window, supports_thinking
               )
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                model_id,
                body.name,
                body.provider,
                body.base_url.strip(),
                body.api_key.strip(),
                body.model_name.strip(),
                body.temperature,
                body.max_tokens,
                int(body.is_default),
                body.role,
                body.fallback_model_id.strip(),
                body.context_window,
                int(body.supports_thinking),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return _serialize_model_row(row)


@router.get("/{model_id}", response_model=ModelOut)
def get_model_endpoint(model_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Model not found")
    return _serialize_model_row(row)


@router.put("/{model_id}", response_model=ModelOut)
def update_model(model_id: str, body: ModelCreate):
    if body.provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider '{body.provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}",
        )
    if body.role not in MODEL_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model role '{body.role}'. Valid roles: {', '.join(MODEL_ROLES)}",
        )

    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM models WHERE id = ?", (model_id,)
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Model not found")

        submitted_key = body.api_key.strip()
        if body.clear_api_key:
            final_api_key = ""
        elif submitted_key and "••••" not in submitted_key:
            final_api_key = submitted_key
        else:
            final_api_key = existing["api_key"] or ""

        if body.is_default:
            conn.execute("UPDATE models SET is_default = 0 WHERE is_default = 1")
        conn.execute(
            """UPDATE models
               SET name=?, provider=?, base_url=?, api_key=?, model_name=?,
                   temperature=?, max_tokens=?, is_default=?, role=?,
                   fallback_model_id=?, context_window=?, supports_thinking=?
               WHERE id=?""",
            (
                body.name,
                body.provider,
                body.base_url.strip(),
                final_api_key,
                body.model_name.strip(),
                body.temperature,
                body.max_tokens,
                int(body.is_default),
                body.role,
                body.fallback_model_id.strip(),
                body.context_window,
                int(body.supports_thinking),
                model_id,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    return _serialize_model_row(row)


@router.delete("/{model_id}")
def delete_model(model_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
        conn.commit()
    return {"ok": True}

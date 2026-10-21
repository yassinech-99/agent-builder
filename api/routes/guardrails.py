"""Multi-tenant Guardrail rules CRUD, content checking, and event logging."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
import re
import threading
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.core.tenancy import get_tenant_context
from api.db import ensure_tenant_guardrails_seeded, get_db, get_db_path, mask_secret

logger = logging.getLogger("nimagents.api.guardrails")
router = APIRouter(prefix="/api/guardrails", tags=["guardrails"])
_GUARDRAIL_DB_LOCK = threading.Lock()
_COMPILED_RULES_CACHE: dict[tuple[str, str], list[tuple[dict, re.Pattern[str]]]] = {}


def _invalidate_rules_cache(tenant_id: str) -> None:
    _COMPILED_RULES_CACHE.pop((str(get_db_path()), tenant_id), None)


class RuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    category: str = Field(..., max_length=64)
    pattern: str = Field(default="", max_length=2000)
    action: str = Field(default="warn", max_length=16)
    enabled: bool = True


class RuleOut(RuleCreate):
    id: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    created_at: str


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    thread_id: str = ""
    assistant_id: str = ""


class CheckHit(BaseModel):
    rule_id: str
    rule_name: str
    category: str
    action: str
    matched_text: str


class CheckResponse(BaseModel):
    passed: bool
    redacted_text: str
    hits: list[CheckHit]


class EventOut(BaseModel):
    id: str
    rule_id: str | None
    rule_name: str | None
    category: str | None
    action: str | None
    matched_text: str | None
    thread_id: str
    assistant_id: str
    created_at: str


@router.get("/rules", response_model=list[RuleOut])
def list_rules():
    ctx = get_tenant_context()
    with get_db() as conn:
        ensure_tenant_guardrails_seeded(conn, ctx.tenant_id)
        rows = conn.execute(
            "SELECT * FROM guardrail_rules WHERE tenant_id = ? ORDER BY created_at DESC",
            (ctx.tenant_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/rules", response_model=RuleOut, status_code=201)
def create_rule(body: RuleCreate):
    if body.pattern:
        try:
            re.compile(body.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")

    ctx = get_tenant_context()
    rule_id = str(uuid.uuid4())
    with get_db() as conn:
        ensure_tenant_guardrails_seeded(conn, ctx.tenant_id)
        conn.execute(
            """INSERT INTO guardrail_rules
               (id, tenant_id, workspace_id, name, category, pattern, action, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rule_id,
                ctx.tenant_id,
                ctx.workspace_id,
                body.name,
                body.category,
                body.pattern,
                body.action,
                body.enabled,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM guardrail_rules WHERE id = ? AND tenant_id = ?",
            (rule_id, ctx.tenant_id),
        ).fetchone()
    _invalidate_rules_cache(ctx.tenant_id)
    return dict(row)


@router.put("/rules/{rule_id}", response_model=RuleOut)
def update_rule(rule_id: str, body: RuleCreate):
    if body.pattern:
        try:
            re.compile(body.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")

    ctx = get_tenant_context()
    with get_db() as conn:
        ensure_tenant_guardrails_seeded(conn, ctx.tenant_id)
        existing = conn.execute(
            "SELECT id FROM guardrail_rules WHERE id = ? AND tenant_id = ?",
            (rule_id, ctx.tenant_id),
        ).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Rule not found")
        conn.execute(
            """UPDATE guardrail_rules SET name=?, category=?, pattern=?, action=?, enabled=?
               WHERE id=? AND tenant_id=?""",
            (body.name, body.category, body.pattern, body.action, body.enabled, rule_id, ctx.tenant_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM guardrail_rules WHERE id = ? AND tenant_id = ?",
            (rule_id, ctx.tenant_id),
        ).fetchone()
    _invalidate_rules_cache(ctx.tenant_id)
    return dict(row)


@router.delete("/{rule_id}")
def delete_rule(rule_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM guardrail_rules WHERE id = ? AND tenant_id = ?",
            (rule_id, ctx.tenant_id),
        )
        conn.commit()
    _invalidate_rules_cache(ctx.tenant_id)
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"ok": True}


def _get_compiled_tenant_rules(tenant_id: str) -> list[tuple[dict, re.Pattern[str]]]:
    cache_key = (str(get_db_path()), tenant_id)
    cached = _COMPILED_RULES_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _GUARDRAIL_DB_LOCK, get_db() as conn:
        ensure_tenant_guardrails_seeded(conn, tenant_id)
        rows = conn.execute(
            "SELECT * FROM guardrail_rules WHERE tenant_id = ? AND enabled = 1",
            (tenant_id,),
        ).fetchall()
    compiled: list[tuple[dict, re.Pattern[str]]] = []
    for r in rows:
        pat = r["pattern"]
        if not pat:
            continue
        try:
            compiled.append((dict(r), re.compile(pat)))
        except re.error:
            continue
    _COMPILED_RULES_CACHE[cache_key] = compiled
    return compiled


@router.post("/check", response_model=CheckResponse)
async def check_text(body: CheckRequest):
    ctx = get_tenant_context()
    compiled_rules = await asyncio.to_thread(_get_compiled_tenant_rules, ctx.tenant_id)

    hits: list[CheckHit] = []
    redacted_text = body.text
    for rule, regex in compiled_rules:
        match = regex.search(body.text)
        if match:
            raw_match = match.group()[:200]
            display_match = (
                mask_secret(raw_match) if rule["category"] == "secrets" else raw_match
            )
            hit = CheckHit(
                rule_id=rule["id"],
                rule_name=rule["name"],
                category=rule["category"],
                action=rule["action"],
                matched_text=display_match,
            )
            hits.append(hit)
            if rule["action"] in ("redact", "block"):
                redacted_text = regex.sub("[REDACTED]", redacted_text)

    if hits:
        await asyncio.to_thread(
            _log_events,
            hits,
            body.thread_id,
            body.assistant_id,
            ctx.tenant_id,
            ctx.workspace_id,
        )

    has_block = any(h.action == "block" for h in hits)
    return CheckResponse(
        passed=not has_block,
        redacted_text=redacted_text,
        hits=hits,
    )


@router.get("/events", response_model=list[EventOut])
def list_events(limit: int = 50):
    ctx = get_tenant_context()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM guardrail_events
               WHERE tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (ctx.tenant_id, ctx.workspace_id, min(limit, 200)),
        ).fetchall()
    return [dict(r) for r in rows]


def _log_events(
    hits: list[CheckHit],
    thread_id: str,
    assistant_id: str,
    tenant_id: str,
    workspace_id: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _GUARDRAIL_DB_LOCK, get_db() as conn:
        for hit in hits:
            conn.execute(
                """INSERT INTO guardrail_events
                   (id, tenant_id, workspace_id, rule_id, rule_name, category, action,
                    matched_text, thread_id, assistant_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    tenant_id,
                    workspace_id,
                    hit.rule_id,
                    hit.rule_name,
                    hit.category,
                    hit.action,
                    hit.matched_text,
                    thread_id,
                    assistant_id,
                    now,
                ),
            )
        conn.commit()

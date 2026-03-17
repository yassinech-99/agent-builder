"""Guardrail rules CRUD, content checking, and event logging."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.db import get_db

logger = logging.getLogger("novagrid.api.guardrails")
router = APIRouter(prefix="/api/guardrails", tags=["guardrails"])


class RuleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    category: str = Field(..., max_length=64)
    pattern: str = Field(default="", max_length=2000)
    action: str = Field(default="warn", max_length=16)
    enabled: bool = True


class RuleOut(RuleCreate):
    id: str
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
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM guardrail_rules ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


@router.post("/rules", response_model=RuleOut, status_code=201)
def create_rule(body: RuleCreate):
    if body.pattern:
        try:
            re.compile(body.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")

    rule_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO guardrail_rules (id, name, category, pattern, action, enabled)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (rule_id, body.name, body.category, body.pattern, body.action, body.enabled),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM guardrail_rules WHERE id = ?", (rule_id,)).fetchone()
    return dict(row)


@router.put("/rules/{rule_id}", response_model=RuleOut)
def update_rule(rule_id: str, body: RuleCreate):
    if body.pattern:
        try:
            re.compile(body.pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid regex: {e}")

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM guardrail_rules WHERE id = ?", (rule_id,)).fetchone()
        if not existing:
            raise HTTPException(status_code=404, detail="Rule not found")
        conn.execute(
            """UPDATE guardrail_rules SET name=?, category=?, pattern=?, action=?, enabled=?
               WHERE id=?""",
            (body.name, body.category, body.pattern, body.action, body.enabled, rule_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM guardrail_rules WHERE id = ?", (rule_id,)).fetchone()
    return dict(row)


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM guardrail_rules WHERE id = ?", (rule_id,))
        conn.commit()
    return {"ok": True}


@router.post("/check", response_model=CheckResponse)
def check_text(body: CheckRequest):
    with get_db() as conn:
        rules = conn.execute(
            "SELECT * FROM guardrail_rules WHERE enabled = 1"
        ).fetchall()

    hits: list[CheckHit] = []
    for rule in rules:
        pattern = rule["pattern"]
        if not pattern:
            continue
        try:
            match = re.search(pattern, body.text)
        except re.error:
            continue
        if match:
            hit = CheckHit(
                rule_id=rule["id"],
                rule_name=rule["name"],
                category=rule["category"],
                action=rule["action"],
                matched_text=match.group()[:200],
            )
            hits.append(hit)

    if hits:
        _log_events(hits, body.thread_id, body.assistant_id)

    has_block = any(h.action == "block" for h in hits)
    return CheckResponse(passed=not has_block, hits=hits)


@router.get("/events", response_model=list[EventOut])
def list_events(limit: int = 50):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM guardrail_events ORDER BY created_at DESC LIMIT ?",
            (min(limit, 200),),
        ).fetchall()
    return [dict(r) for r in rows]


def _log_events(hits: list[CheckHit], thread_id: str, assistant_id: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        for hit in hits:
            conn.execute(
                """INSERT INTO guardrail_events
                   (id, rule_id, rule_name, category, action, matched_text, thread_id, assistant_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
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

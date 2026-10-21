"""Multi-tenant Guardrail rules CRUD, NVIDIA NemoGuard policies, multi-stage checking, and audit telemetry."""

from __future__ import annotations

import re
from typing import Literal
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.guardrails_engine import (
    DEFAULT_BLOCKED_TOPICS,
    DEFAULT_JAILBREAK_MODEL,
    DEFAULT_SAFETY_MODEL,
    DEFAULT_TOPIC_MODEL,
    safety_engine,
)
from api.core.tenancy import get_tenant_context
from api.db import ensure_tenant_guardrails_seeded, get_db

router = APIRouter(prefix="/api/guardrails", tags=["guardrails"])


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


class NemoGuardPolicyUpdate(BaseModel):
    nemoguard_enabled: bool = True
    safety_model: str = Field(default=DEFAULT_SAFETY_MODEL, max_length=200)
    topic_model: str = Field(default=DEFAULT_TOPIC_MODEL, max_length=200)
    jailbreak_model: str = Field(default=DEFAULT_JAILBREAK_MODEL, max_length=200)
    endpoint_url: str = Field(default="", max_length=512)
    enforce_input: bool = True
    enforce_tool_output: bool = True
    enforce_model_output: bool = True
    allowed_topics: list[str] = Field(default_factory=list)
    blocked_topics: list[str] = Field(default_factory=lambda: list(DEFAULT_BLOCKED_TOPICS))
    jailbreak_threshold: float = Field(default=0.65, ge=0.0, le=1.0)


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    thread_id: str = ""
    assistant_id: str = ""


class EvaluateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    stage: Literal["input", "tool_output", "output"] = "input"
    thread_id: str = ""
    assistant_id: str = ""
    record_audit: bool = True


class CheckHit(BaseModel):
    rule_id: str
    rule_name: str
    category: str
    action: str
    matched_text: str
    detector: str = "regex"
    stage: str = "input"
    risk_score: float = 0.0


class CheckResponse(BaseModel):
    passed: bool
    redacted_text: str
    hits: list[CheckHit]
    stage: str = "input"
    blocked_reason: str = ""
    risk_score: float = 0.0
    latency_ms: float = 0.0
    detectors_run: list[str] = Field(default_factory=list)


class EventOut(BaseModel):
    id: str
    rule_id: str | None
    rule_name: str | None
    category: str | None
    action: str | None
    matched_text: str | None
    stage: str = "input"
    detector: str = "regex"
    latency_ms: float = 0.0
    risk_score: float = 0.0
    thread_id: str
    assistant_id: str
    created_at: str


@router.get("/policy")
def get_nemoguard_policy():
    ctx = get_tenant_context()
    return safety_engine.get_policy_sync(ctx).to_dict()


@router.put("/policy")
def update_nemoguard_policy(body: NemoGuardPolicyUpdate):
    ctx = get_tenant_context()
    return safety_engine.upsert_policy_sync(body.model_dump(), ctx).to_dict()


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
    safety_engine.invalidate_tenant_cache(ctx.tenant_id)
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
    safety_engine.invalidate_tenant_cache(ctx.tenant_id)
    return dict(row)


@router.delete("/rules/{rule_id}")
@router.delete("/{rule_id}")
def delete_rule(rule_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM guardrail_rules WHERE id = ? AND tenant_id = ?",
            (rule_id, ctx.tenant_id),
        )
        conn.commit()
    safety_engine.invalidate_tenant_cache(ctx.tenant_id)
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Rule not found")
    return {"ok": True}


@router.post("/check", response_model=CheckResponse)
async def check_text(body: CheckRequest):
    ctx = get_tenant_context()
    res = await safety_engine.evaluate(
        body.text,
        stage="input",
        thread_id=body.thread_id,
        assistant_id=body.assistant_id,
        ctx=ctx,
    )
    return CheckResponse(
        passed=res.passed,
        redacted_text=res.sanitized_text,
        hits=[CheckHit(**h.__dict__) for h in res.hits],
        stage=res.stage,
        blocked_reason=res.blocked_reason,
        risk_score=round(res.risk_score, 4),
        latency_ms=round(res.latency_ms, 3),
        detectors_run=res.detectors_run,
    )


@router.post("/evaluate")
async def evaluate_stage_text(body: EvaluateRequest):
    ctx = get_tenant_context()
    res = await safety_engine.evaluate(
        body.text,
        stage=body.stage,
        thread_id=body.thread_id,
        assistant_id=body.assistant_id,
        ctx=ctx,
        record_audit=body.record_audit,
    )
    return res.to_dict()


@router.get("/events", response_model=list[EventOut])
def list_events(limit: int = 50, stage: str = ""):
    ctx = get_tenant_context()
    with get_db() as conn:
        if stage:
            rows = conn.execute(
                """SELECT * FROM guardrail_events
                   WHERE tenant_id = ? AND workspace_id = ? AND stage = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (ctx.tenant_id, ctx.workspace_id, stage, min(limit, 200)),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM guardrail_events
                   WHERE tenant_id = ? AND workspace_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (ctx.tenant_id, ctx.workspace_id, min(limit, 200)),
            ).fetchall()
    return [dict(r) for r in rows]

"""Control Plane API for the 5-Stage Context Engine, KV-Cache Telemetry & Offloaded Artifacts (`/api/context`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from agent.context_engine import (
    ContextPolicy,
    context_engine,
    read_offloaded_artifact,
)
from agent.registry import tool_registry
from agent.skills import build_tier1_skill_directory_prompt, load_tenant_skills
from api.core.tenancy import get_tenant_context
from api.db import get_db

router = APIRouter(prefix="/api/context", tags=["context"])


class ManualCompactRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(default_factory=list)
    context_window: int = Field(default=131072, ge=2048, le=2_000_000)


class SimulateShaperRequest(BaseModel):
    messages: list[dict[str, Any]]
    context_window: int = Field(default=16384, ge=1024, le=2_000_000)
    offload_threshold_tokens: int = Field(default=3000, ge=100, le=100_000)
    clear_at_least_tokens: int = Field(default=10000, ge=100, le=200_000)
    force_compact: bool = False


def _deserialize_wire_messages(wire: list[dict[str, Any]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for item in wire:
        role = str(item.get("type") or item.get("role") or "human").lower()
        content = item.get("content", "")
        if role in ("human", "user"):
            out.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            out.append(
                AIMessage(
                    content=content,
                    tool_calls=item.get("tool_calls") or [],
                    additional_kwargs=item.get("additional_kwargs") or {},
                )
            )
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=str(content),
                    tool_call_id=str(item.get("tool_call_id") or "call_sim"),
                    name=str(item.get("name") or "tool"),
                )
            )
    return out


@router.get("/threads/{thread_id}")
async def get_thread_context_telemetry(thread_id: str):
    """Return real-time 3-Block Context & KV-Cache breakdown for the Chat Top-Bar `ContextMeter`."""
    ctx = get_tenant_context()
    row_id = f"{ctx.tenant_id}:{ctx.workspace_id}:{thread_id}"

    def _query() -> dict[str, Any]:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM session_compactions WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (row_id, ctx.tenant_id, ctx.workspace_id),
            ).fetchone()
        if row:
            d = dict(row)
            return {
                "thread_id": thread_id,
                "agent_name": d["agent_name"],
                "prefix_hash": d["prefix_hash"],
                "static_prefix_tokens": d["static_prefix_tokens"],
                "session_prefix_tokens": d["session_prefix_tokens"],
                "dynamic_tail_tokens": d["dynamic_tail_tokens"],
                "total_tokens": d["total_tokens"],
                "context_window": d["context_window"],
                "utilization_pct": round((d["total_tokens"] / max(1, d["context_window"])) * 100.0, 2),
                "kv_cache_hit_pct": d["kv_cache_hit_pct"],
                "deferred_tools_count": d["deferred_tools_count"],
                "active_skills": json.loads(d.get("active_skills_json") or "[]"),
                "offloaded_artifacts_count": d["offloaded_artifacts_count"],
                "cleared_thinking_tokens": d["cleared_thinking_tokens"],
                "cleared_tool_tokens": d["cleared_tool_tokens"],
                "compaction_count": d["compaction_count"],
                "checkpoint_summary": d["checkpoint_summary"],
                "pre_compact_facts": json.loads(d.get("pre_compact_facts_json") or "[]"),
                "updated_at": d["updated_at"],
            }

        # Baseline startup estimate before Turn 1 executes on a new thread
        budget = tool_registry.compute_token_budget_report(ctx=ctx)
        skills = load_tenant_skills(ctx, only_enabled=True)
        skill_dir_toks = sum(s["tier1_index_tokens"] for s in skills)
        static_toks = budget["deferred_startup_tokens"] + skill_dir_toks + 45
        return {
            "thread_id": thread_id,
            "agent_name": "nimAgents Assistant",
            "prefix_hash": "",
            "static_prefix_tokens": static_toks,
            "session_prefix_tokens": 0,
            "dynamic_tail_tokens": 0,
            "total_tokens": static_toks,
            "context_window": 131072,
            "utilization_pct": round((static_toks / 131072) * 100.0, 2),
            "kv_cache_hit_pct": 92.0,
            "deferred_tools_count": budget["deferred_tools"],
            "active_skills": [],
            "offloaded_artifacts_count": 0,
            "cleared_thinking_tokens": 0,
            "cleared_tool_tokens": 0,
            "compaction_count": 0,
            "checkpoint_summary": "",
            "pre_compact_facts": [],
            "updated_at": "",
        }

    return await asyncio.to_thread(_query)


@router.post("/threads/{thread_id}/compact")
async def trigger_manual_thread_compaction(thread_id: str, body: ManualCompactRequest):
    """Trigger immediate rolling compaction & pre-compact fact flush from the Chat `ContextMeter`."""
    ctx = get_tenant_context()
    msgs = _deserialize_wire_messages(body.messages)
    budget = await asyncio.to_thread(lambda: tool_registry.compute_token_budget_report(ctx=ctx))
    skills = await asyncio.to_thread(lambda: load_tenant_skills(ctx, only_enabled=True))
    static_prompt = "You are nimAgents Assistant." + build_tier1_skill_directory_prompt(skills)

    projection = await context_engine.shape_context(
        static_system_prompt=static_prompt,
        messages=msgs,
        tool_schema_tokens=budget["deferred_startup_tokens"],
        policy=ContextPolicy(context_window=body.context_window),
        ctx=ctx,
        thread_id=thread_id,
        deferred_tools_count=budget["deferred_tools"],
        force_compact=True,
    )
    return {
        "ok": True,
        "thread_id": thread_id,
        "compacted": projection.compacted_this_turn,
        "total_tokens": projection.total_tokens,
        "static_prefix_tokens": projection.static_prefix_tokens,
        "session_prefix_tokens": projection.session_prefix_tokens,
        "dynamic_tail_tokens": projection.dynamic_tail_tokens,
        "kv_cache_hit_pct": projection.kv_cache_hit_pct,
        "pre_compact_facts": projection.pre_compact_facts,
        "checkpoint_summary": projection.updated_checkpoint,
    }


@router.get("/artifacts")
async def list_offloaded_artifacts(thread_id: str = Query(default="")):
    ctx = get_tenant_context()

    def _load() -> list[dict[str, Any]]:
        with get_db() as conn:
            if thread_id:
                rows = conn.execute(
                    """SELECT id, thread_id, tool_name, content_type, byte_size, token_count,
                              storage_uri, structural_summary, created_at
                       FROM context_artifacts
                       WHERE tenant_id = ? AND workspace_id = ? AND thread_id = ?
                       ORDER BY created_at DESC""",
                    (ctx.tenant_id, ctx.workspace_id, thread_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, thread_id, tool_name, content_type, byte_size, token_count,
                              storage_uri, structural_summary, created_at
                       FROM context_artifacts
                       WHERE tenant_id = ? AND workspace_id = ?
                       ORDER BY created_at DESC LIMIT 50""",
                    (ctx.tenant_id, ctx.workspace_id),
                ).fetchall()
        return [dict(r) for r in rows]

    return await asyncio.to_thread(_load)


@router.get("/artifacts/{artifact_id}")
async def read_artifact_endpoint(
    artifact_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=80, ge=1, le=250),
    grep: str = Query(default=""),
):
    ctx = get_tenant_context()
    text = await asyncio.to_thread(
        lambda: read_offloaded_artifact(artifact_id, offset=offset, limit=limit, grep=grep, ctx=ctx)
    )
    if text.endswith("not found in current tenant workspace."):
        raise HTTPException(status_code=404, detail=text)
    return {"artifact_id": artifact_id, "output": text}


@router.post("/simulate")
async def simulate_context_shaper(body: SimulateShaperRequest):
    ctx = get_tenant_context()
    msgs = _deserialize_wire_messages(body.messages)
    proj = await context_engine.shape_context(
        static_system_prompt="You are nimAgents Assistant.",
        messages=msgs,
        policy=ContextPolicy(
            context_window=body.context_window,
            offload_threshold_tokens=body.offload_threshold_tokens,
            clear_at_least_tokens=body.clear_at_least_tokens,
        ),
        ctx=ctx,
        force_compact=body.force_compact,
    )
    return {
        "prefix_hash": proj.prefix_hash,
        "static_prefix_tokens": proj.static_prefix_tokens,
        "session_prefix_tokens": proj.session_prefix_tokens,
        "dynamic_tail_tokens": proj.dynamic_tail_tokens,
        "total_tokens": proj.total_tokens,
        "context_window": proj.context_window,
        "utilization_pct": proj.utilization_pct,
        "kv_cache_hit_pct": proj.kv_cache_hit_pct,
        "offloaded_artifact_ids": proj.offloaded_artifact_ids,
        "cleared_thinking_tokens": proj.cleared_thinking_tokens,
        "cleared_tool_tokens": proj.cleared_tool_tokens,
        "compacted_this_turn": proj.compacted_this_turn,
        "pre_compact_facts": proj.pre_compact_facts,
        "checkpoint_summary": proj.updated_checkpoint,
        "shaped_message_count": len(proj.block3_messages),
    }

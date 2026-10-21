"""Control Plane API for Durable Sessions, `Last-Event-ID` SSE Replay, Crash Recovery & Forking (`/api/sessions`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.core.tenancy import get_tenant_context
from api.db import get_db
from api.services.run_manager import SessionLeaseBusyError, session_manager

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    title: str = Field(default="New Session", max_length=120)
    assistant_id: str = Field(default="agent", max_length=80)
    tags: list[str] = Field(default_factory=list)


class UpdateSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)
    pinned: bool | None = None
    archived: bool | None = None
    tags: list[str] | None = None


class StartRunRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    agent_config: dict[str, Any] = Field(default_factory=dict)


class ForkSessionRequest(BaseModel):
    up_to_seq: int | None = Field(default=None, ge=1)
    title: str = Field(default="", max_length=120)


@router.get("")
async def list_durable_sessions(
    search: str = Query(default=""),
    pinned_only: bool = Query(default=False),
    include_archived: bool = Query(default=False),
    tag: str = Query(default=""),
):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: session_manager.list_sessions(
            search=search,
            pinned_only=pinned_only,
            include_archived=include_archived,
            tag=tag,
            ctx=ctx,
        )
    )


@router.post("", status_code=201)
async def create_durable_session(body: CreateSessionRequest):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: session_manager.create_session(
            title=body.title,
            assistant_id=body.assistant_id,
            tags=body.tags,
            ctx=ctx,
        )
    )


@router.get("/{session_id}")
async def get_durable_session_detail(session_id: str):
    ctx = get_tenant_context()

    def _load() -> dict[str, Any]:
        sess = session_manager.get_session(session_id, ctx=ctx)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found")
        events = session_manager.get_events(session_id, after_seq=0, ctx=ctx)
        with get_db() as conn:
            sub_rows = conn.execute(
                """SELECT * FROM subagent_runs
                   WHERE parent_session_id = ? AND tenant_id = ? AND workspace_id = ?
                   ORDER BY created_at ASC""",
                (session_id, ctx.tenant_id, ctx.workspace_id),
            ).fetchall()
        subagents = [
            {
                **dict(r),
                "steps": json.loads(r["trace_json"] or "[]"),
            }
            for r in sub_rows
        ]
        return {**sess, "events": events, "subagents": subagents}

    return await asyncio.to_thread(_load)


@router.patch("/{session_id}")
async def patch_durable_session(session_id: str, body: UpdateSessionRequest):
    ctx = get_tenant_context()
    updated = await asyncio.to_thread(
        lambda: session_manager.update_session_metadata(
            session_id,
            title=body.title,
            pinned=body.pinned,
            archived=body.archived,
            tags=body.tags,
            ctx=ctx,
        )
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated


@router.delete("/{session_id}")
async def delete_durable_session(session_id: str):
    ctx = get_tenant_context()
    ok = await asyncio.to_thread(lambda: session_manager.delete_session(session_id, ctx=ctx))
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@router.post("/{session_id}/runs", status_code=202)
async def enqueue_durable_run(session_id: str, body: StartRunRequest):
    ctx = get_tenant_context()
    try:
        return await session_manager.start_background_run(
            session_id,
            body.prompt,
            agent_config=body.agent_config,
            ctx=ctx,
        )
    except SessionLeaseBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/{session_id}/stream")
async def stream_durable_session_events(
    session_id: str,
    after_seq: int = Query(default=0, ge=0),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    """Reconnectable SSE stream honoring standard `Last-Event-ID` header or `?after_seq=` cursor."""
    ctx = get_tenant_context()
    sess = await asyncio.to_thread(lambda: session_manager.get_session(session_id, ctx=ctx))
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    resume_cursor = after_seq
    if last_event_id is not None and last_event_id.strip().isdigit():
        resume_cursor = max(resume_cursor, int(last_event_id.strip()))

    return StreamingResponse(
        session_manager.subscribe_sse(session_id, last_event_id=resume_cursor, ctx=ctx),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{session_id}/cancel")
async def cancel_durable_session_run(session_id: str):
    ctx = get_tenant_context()
    sess = await asyncio.to_thread(lambda: session_manager.get_session(session_id, ctx=ctx))
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await session_manager.cancel_session(session_id, ctx=ctx)


@router.post("/{session_id}/wake")
async def wake_orphan_durable_session(session_id: str):
    ctx = get_tenant_context()
    try:
        return await session_manager.wake_session(session_id, ctx=ctx)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SessionLeaseBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{session_id}/fork", status_code=201)
async def fork_durable_session(session_id: str, body: ForkSessionRequest):
    ctx = get_tenant_context()
    try:
        return await asyncio.to_thread(
            lambda: session_manager.fork_session(
                session_id,
                up_to_seq=body.up_to_seq,
                title=body.title,
                ctx=ctx,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{session_id}/export")
async def export_durable_session(
    session_id: str,
    format: str = Query(default="markdown", pattern="^(markdown|jsonl)$"),
):
    ctx = get_tenant_context()
    try:
        content = await asyncio.to_thread(
            lambda: session_manager.export_session_trace(session_id, fmt=format, ctx=ctx)
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media = "application/x-ndjson" if format == "jsonl" else "text/markdown; charset=utf-8"
    return PlainTextResponse(content=content, media_type=media)

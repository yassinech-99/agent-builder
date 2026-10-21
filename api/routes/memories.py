"""Control Plane API for Phase 6 — 4-Tier Cognitive Memory, Bi-Temporal Lineage & Sleep-Time Consolidation (`/api/memories`)."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from agent.memory_engine import MAX_CORE_MEMORY_TOKENS, memory_engine
from api.core.tenancy import get_tenant_context
from api.services.memory_consolidator import memory_consolidator

router = APIRouter(prefix="/api/memories", tags=["memories"])


class UpsertBlockRequest(BaseModel):
    id: str | None = None
    block_type: str = Field(default="agent_notes", max_length=40)
    title: str = Field(default="", max_length=120)
    content: str = Field(..., min_length=1, max_length=16_000)
    agent_id: str = Field(default="default", max_length=120)
    pinned: bool = True


class UpsertSemanticFactRequest(BaseModel):
    subject: str = Field(..., min_length=1, max_length=160)
    predicate: str = Field(..., min_length=1, max_length=160)
    object: str = Field(..., min_length=1, max_length=500)
    category: str = Field(default="fact", max_length=60)
    confidence: float = Field(default=0.9, ge=0.05, le=1.0)
    importance: float = Field(default=0.75, ge=0.05, le=1.0)
    agent_id: str = Field(default="default", max_length=120)
    source_session_id: str = Field(default="", max_length=120)
    valid_from: str | None = None


class CreateEpisodicRequest(BaseModel):
    summary: str = Field(..., min_length=1, max_length=500)
    problem: str = Field(default="", max_length=1000)
    approach: str = Field(default="", max_length=1000)
    key_tool_steps: list[str] = Field(default_factory=list)
    user_corrections: str = Field(default="", max_length=1000)
    outcome: str = Field(default="succeeded", max_length=500)
    importance: float = Field(default=0.75, ge=0.05, le=1.0)
    session_id: str = Field(default="", max_length=120)
    agent_id: str = Field(default="default", max_length=120)


class ProposeProceduralRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    rule_prompt: str = Field(..., min_length=1, max_length=4000)
    rationale: str = Field(default="", max_length=1000)
    agent_id: str = Field(default="default", max_length=120)
    status: str = Field(default="proposed", max_length=30)


class UpdateProceduralRequest(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    rule_prompt: str | None = Field(default=None, max_length=4000)
    rationale: str | None = Field(default=None, max_length=1000)
    status: str | None = Field(default=None, max_length=30)


class RollbackProceduralRequest(BaseModel):
    target_version: int = Field(..., ge=1)


class RecallMemoriesRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=8, ge=1, le=25)
    agent_id: str | None = None


class ConsolidateMessagesItem(BaseModel):
    role: str = Field(default="user")
    content: str = Field(..., min_length=1)


class ConsolidateRequest(BaseModel):
    messages: list[ConsolidateMessagesItem] = Field(..., min_length=1)
    session_id: str = Field(default="", max_length=120)
    agent_id: str = Field(default="default", max_length=120)
    background: bool = True


@router.get("/summary")
async def get_memories_summary():
    ctx = get_tenant_context()

    def _compute() -> dict[str, Any]:
        blocks = memory_engine.list_blocks(ctx=ctx)
        all_facts = memory_engine.list_semantic_facts(include_historical=True, ctx=ctx)
        active_facts = [f for f in all_facts if f["invalidated_at"] is None]
        episodes = memory_engine.list_episodic_memories(ctx=ctx)
        procs = memory_engine.list_procedural_memories(ctx=ctx)
        pinned_tokens = sum(int(b["token_count"] or 0) for b in blocks if b["pinned"])
        return {
            "core_blocks_count": len(blocks),
            "pinned_core_tokens": pinned_tokens,
            "max_core_tokens": MAX_CORE_MEMORY_TOKENS,
            "active_semantic_facts": len(active_facts),
            "superseded_semantic_facts": len(all_facts) - len(active_facts),
            "episodic_memories_count": len(episodes),
            "procedural_active_count": sum(1 for p in procs if p["status"] == "active"),
            "procedural_proposed_count": sum(1 for p in procs if p["status"] == "proposed"),
        }

    return await asyncio.to_thread(_compute)


@router.get("/blocks")
async def list_core_memory_blocks(agent_id: str | None = Query(default=None)):
    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: memory_engine.list_blocks(agent_id=agent_id, ctx=ctx))


@router.post("/blocks", status_code=201)
async def upsert_core_memory_block(body: UpsertBlockRequest):
    ctx = get_tenant_context()
    try:
        return await asyncio.to_thread(
            lambda: memory_engine.upsert_block(
                block_id=body.id,
                block_type=body.block_type,
                title=body.title,
                content=body.content,
                agent_id=body.agent_id,
                pinned=body.pinned,
                ctx=ctx,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/blocks/{block_id}")
async def remove_core_memory_block(block_id: str):
    ctx = get_tenant_context()
    ok = await asyncio.to_thread(lambda: memory_engine.delete_block(block_id, ctx=ctx))
    if not ok:
        raise HTTPException(status_code=404, detail="Memory block not found")
    return {"ok": True}


@router.get("/facts")
async def list_bitemporal_facts(
    include_historical: bool = Query(default=False),
    as_of: str | None = Query(default=None),
    search: str = Query(default=""),
):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: memory_engine.list_semantic_facts(
            include_historical=include_historical,
            as_of=as_of,
            search=search,
            ctx=ctx,
        )
    )


@router.post("/facts", status_code=201)
async def upsert_bitemporal_fact(body: UpsertSemanticFactRequest):
    ctx = get_tenant_context()
    try:
        return await memory_engine.upsert_semantic_fact(
            subject=body.subject,
            predicate=body.predicate,
            obj=body.object,
            category=body.category,
            confidence=body.confidence,
            importance=body.importance,
            agent_id=body.agent_id,
            source_session_id=body.source_session_id,
            valid_from=body.valid_from,
            ctx=ctx,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.delete("/facts/{fact_id}")
async def delete_bitemporal_fact(fact_id: str):
    ctx = get_tenant_context()
    ok = await memory_engine.delete_semantic_fact(fact_id, ctx=ctx)
    if not ok:
        raise HTTPException(status_code=404, detail="Semantic memory fact not found")
    return {"ok": True}


@router.get("/episodes")
async def list_episodes():
    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: memory_engine.list_episodic_memories(ctx=ctx))


@router.post("/episodes", status_code=201)
async def create_episode(body: CreateEpisodicRequest):
    ctx = get_tenant_context()
    return await memory_engine.record_episodic_memory(
        summary=body.summary,
        problem=body.problem,
        approach=body.approach,
        key_tool_steps=body.key_tool_steps,
        user_corrections=body.user_corrections,
        outcome=body.outcome,
        importance=body.importance,
        session_id=body.session_id,
        agent_id=body.agent_id,
        ctx=ctx,
    )


@router.delete("/episodes/{ep_id}")
async def remove_episode(ep_id: str):
    ctx = get_tenant_context()
    ok = await memory_engine.delete_episodic_memory(ep_id, ctx=ctx)
    if not ok:
        raise HTTPException(status_code=404, detail="Episodic memory not found")
    return {"ok": True}


@router.get("/procedural")
async def list_procedural_rules(status: str | None = Query(default=None)):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: memory_engine.list_procedural_memories(status=status, ctx=ctx)
    )


@router.post("/procedural", status_code=201)
async def create_procedural_rule(body: ProposeProceduralRequest):
    ctx = get_tenant_context()
    return await asyncio.to_thread(
        lambda: memory_engine.propose_procedural_memory(
            title=body.title,
            rule_prompt=body.rule_prompt,
            rationale=body.rationale,
            agent_id=body.agent_id,
            status=body.status,
            ctx=ctx,
        )
    )


@router.patch("/procedural/{proc_id}")
async def patch_procedural_rule(proc_id: str, body: UpdateProceduralRequest):
    ctx = get_tenant_context()
    updated = await asyncio.to_thread(
        lambda: memory_engine.update_procedural_memory(
            proc_id,
            title=body.title,
            rule_prompt=body.rule_prompt,
            rationale=body.rationale,
            status=body.status,
            ctx=ctx,
        )
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Procedural memory not found")
    return updated


@router.post("/procedural/{proc_id}/rollback")
async def rollback_procedural_rule(proc_id: str, body: RollbackProceduralRequest):
    ctx = get_tenant_context()
    try:
        restored = await asyncio.to_thread(
            lambda: memory_engine.rollback_procedural_memory(
                proc_id,
                target_version=body.target_version,
                ctx=ctx,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if restored is None:
        raise HTTPException(status_code=404, detail="Procedural memory not found")
    return restored


@router.post("/recall")
async def recall_memories_endpoint(body: RecallMemoriesRequest):
    ctx = get_tenant_context()
    return await memory_engine.recall_memories(
        body.query,
        limit=body.limit,
        agent_id=body.agent_id,
        ctx=ctx,
    )


@router.post("/consolidate", status_code=202)
async def trigger_sleep_consolidation(body: ConsolidateRequest):
    ctx = get_tenant_context()
    lc_messages: list[BaseMessage] = []
    for item in body.messages:
        if item.role == "assistant":
            lc_messages.append(AIMessage(content=item.content))
        else:
            lc_messages.append(HumanMessage(content=item.content))

    if body.background:
        return memory_consolidator.enqueue_sleep_consolidation(
            lc_messages,
            session_id=body.session_id,
            agent_id=body.agent_id,
            ctx=ctx,
        )
    return await memory_consolidator.run_consolidation_now(
        lc_messages,
        session_id=body.session_id,
        agent_id=body.agent_id,
        ctx=ctx,
    )

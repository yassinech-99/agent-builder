"""Phase 6 — Zero-Latency "Sleep-Time" Cognitive Memory Consolidator (`api/services/memory_consolidator.py`).

Decouples all memory extraction, bi-temporal conflict invalidation, episodic trajectory synthesis,
and procedural prompt proposals completely from the interactive P0 chat hot path (`0ms` turn latency overhead!):
  - `enqueue_sleep_consolidation(...)` records a `background_jobs` entry and dispatches an async task
    onto **Priority Lane `1` (`Background Fair-Queued Pool`)** in `< 1ms`.
  - Extracts structured `(subject, predicate, object)` facts, user preferences, explicit user corrections
    (automatically proposing **Tier-4 Procedural Rules** when users say *"Always..."* / *"Never..."*!),
    and **Tier-3 Episodic Trajectories** (*Problem → Approach → Key Tool Steps → Outcome*).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import logging
import re
from typing import Any, Awaitable, Callable
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.memory_engine import memory_engine
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.memory_consolidator")

# High-precision structured patterns for zero-latency deterministic + LLM-assisted fact extraction
_EXPLICIT_FACT_PATTERNS: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    # "Our <slot> is/runs on/uses <val>" or "My <slot> is <val>"
    (
        re.compile(
            r"\b(?:our|my)\s+([a-zA-Z][a-zA-Z0-9 _/-]{2,42}?)\s+(?:is|runs\s+on|uses|migrated\s+to|switched\s+to)\s+([^\n.;!?]{2,90})",
            re.IGNORECASE,
        ),
        "user",
        "{slot}",
        "{val}",
    ),
    # "We migrated/switched <slot> to <val>"
    (
        re.compile(
            r"\bwe\s+(?:migrated|switched|moved|upgraded)\s+(?:our\s+)?([a-zA-Z][a-zA-Z0-9 _/-]{2,42}?)\s+(?:from\s+[^\n.;!?]+?\s+)?to\s+([^\n.;!?]{2,90})",
            re.IGNORECASE,
        ),
        "workspace",
        "{slot}",
        "{val}",
    ),
    # "I prefer <val> for <slot>" / "I prefer <val>"
    (
        re.compile(
            r"\bi\s+prefer\s+([^\n.;!?]{2,80})",
            re.IGNORECASE,
        ),
        "user",
        "prefers",
        "{slot}",
    ),
)

_PROCEDURAL_RULE_RE = re.compile(
    r"\b((?:always|never|from now on[, ]+always|please always|do not ever)\s+[^\n.!?]{8,180})",
    re.IGNORECASE,
)


def extract_structured_insights_from_messages(
    messages: list[BaseMessage],
) -> dict[str, Any]:
    """Deterministic high-signal extractor for facts, procedural corrections, and episodic trajectories."""
    facts: list[dict[str, Any]] = []
    procedural_rules: list[dict[str, str]] = []
    human_turns: list[str] = []
    assistant_turns: list[str] = []
    tool_names_used: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            txt = str(msg.content).strip()
            if not txt:
                continue
            human_turns.append(txt)

            for pat, default_subj, pred_tpl, obj_tpl in _EXPLICIT_FACT_PATTERNS:
                for m in pat.finditer(txt):
                    if obj_tpl == "{slot}":
                        facts.append(
                            {
                                "subject": default_subj,
                                "predicate": pred_tpl,
                                "object": m.group(1).strip(),
                                "category": "preference",
                                "confidence": 0.92,
                                "importance": 0.85,
                            }
                        )
                    else:
                        slot_name = m.group(1).strip()
                        val_str = m.group(2).strip()
                        facts.append(
                            {
                                "subject": default_subj,
                                "predicate": slot_name,
                                "object": val_str,
                                "category": "fact",
                                "confidence": 0.94,
                                "importance": 0.85,
                            }
                        )

            for pm in _PROCEDURAL_RULE_RE.finditer(txt):
                directive = pm.group(1).strip()
                words = directive.split()[:6]
                procedural_rules.append(
                    {
                        "title": " ".join(words).capitalize(),
                        "rule_prompt": directive,
                        "rationale": "Extracted automatically from explicit user standing instruction during sleep-time consolidation.",
                    }
                )

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tname = tc.get("name")
                    if tname and tname not in tool_names_used:
                        tool_names_used.append(tname)
            txt = str(msg.content).strip()
            if txt:
                assistant_turns.append(txt)
        elif isinstance(msg, ToolMessage):
            if msg.name and msg.name not in tool_names_used:
                tool_names_used.append(msg.name)

    episode: dict[str, Any] | None = None
    if human_turns and assistant_turns:
        problem = human_turns[0][:240]
        outcome_text = assistant_turns[-1][:260]
        corrections = "; ".join(r["rule_prompt"] for r in procedural_rules[:2])
        episode = {
            "summary": f"Resolved: {problem[:120]}",
            "problem": problem,
            "approach": (
                f"Executed {len(tool_names_used)} tool(s) ({', '.join(tool_names_used)})"
                if tool_names_used
                else "Direct synthesis & reasoning"
            ),
            "key_tool_steps": tool_names_used,
            "user_corrections": corrections,
            "outcome": outcome_text,
            "importance": 0.85 if tool_names_used or corrections else 0.65,
        }

    return {
        "facts": facts,
        "procedural_rules": procedural_rules,
        "episode": episode,
    }


class SleepTimeMemoryConsolidator:
    """Async Lane-P1 background memory consolidator (`0ms` blocking on interactive P0 chat)."""

    def __init__(self) -> None:
        self._inflight_tasks: set[asyncio.Task[Any]] = set()

    def _set_job_state(
        self,
        job_id: str,
        status: str,
        progress: int,
        detail: str,
        ctx: TenantContext,
    ) -> None:
        now_iso = datetime.now(tz=UTC).isoformat()
        with get_db() as conn:
            conn.execute(
                """INSERT INTO background_jobs
                   (id, tenant_id, workspace_id, lane, job_type, status, progress, detail, created_at, updated_at)
                   VALUES (?, ?, ?, 'background', 'memory_sleep_consolidation', ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     status = excluded.status,
                     progress = excluded.progress,
                     detail = excluded.detail,
                     updated_at = excluded.updated_at""",
                (job_id, ctx.tenant_id, ctx.workspace_id, status, progress, detail, now_iso, now_iso),
            )
            conn.commit()

    async def run_consolidation_now(
        self,
        messages: list[BaseMessage],
        *,
        session_id: str = "",
        agent_id: str = "default",
        llm_extractor: Callable[[list[BaseMessage]], Awaitable[dict[str, Any]]] | None = None,
        job_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Execute full 4-tier consolidation pass on Priority Lane 1 (`Background`)."""
        active = ctx or get_tenant_context()
        jid = job_id or f"memjob_{uuid.uuid4().hex[:10]}"
        await asyncio.to_thread(
            self._set_job_state, jid, "running", 20, f"Consolidating session {session_id or 'ephemeral'}", active
        )

        async def _lane_p1_work() -> dict[str, Any]:
            extracted = extract_structured_insights_from_messages(messages)
            if llm_extractor is not None:
                extra = await llm_extractor(messages)
                extracted["facts"].extend(extra.get("facts") or [])
                extracted["procedural_rules"].extend(extra.get("procedural_rules") or [])

            written_facts: list[dict[str, Any]] = []
            for fact in extracted["facts"]:
                rec = await memory_engine.upsert_semantic_fact(
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    obj=fact["object"],
                    category=fact.get("category", "fact"),
                    confidence=float(fact.get("confidence", 0.9)),
                    importance=float(fact.get("importance", 0.8)),
                    agent_id=agent_id,
                    source_session_id=session_id,
                    ctx=active,
                )
                written_facts.append(rec)

            proposed_rules: list[dict[str, Any]] = []
            existing_titles = {
                r["title"].lower()
                for r in await asyncio.to_thread(
                    lambda: memory_engine.list_procedural_memories(agent_id=agent_id, ctx=active)
                )
            }
            for rule in extracted["procedural_rules"]:
                if rule["title"].lower() in existing_titles:
                    continue
                created_rule = await asyncio.to_thread(
                    lambda r=rule: memory_engine.propose_procedural_memory(
                        title=r["title"],
                        rule_prompt=r["rule_prompt"],
                        rationale=r["rationale"],
                        agent_id=agent_id,
                        source_session_id=session_id,
                        ctx=active,
                    )
                )
                proposed_rules.append(created_rule)

            recorded_episode: dict[str, Any] | None = None
            ep_data = extracted.get("episode")
            if ep_data is not None:
                recorded_episode = await memory_engine.record_episodic_memory(
                    summary=ep_data["summary"],
                    problem=ep_data["problem"],
                    approach=ep_data["approach"],
                    key_tool_steps=ep_data["key_tool_steps"],
                    user_corrections=ep_data["user_corrections"],
                    outcome=ep_data["outcome"],
                    importance=ep_data["importance"],
                    session_id=session_id,
                    agent_id=agent_id,
                    ctx=active,
                )

            return {
                "job_id": jid,
                "facts_consolidated": len(written_facts),
                "facts": written_facts,
                "procedural_proposals": len(proposed_rules),
                "rules": proposed_rules,
                "episodic_recorded": recorded_episode is not None,
                "episode": recorded_episode,
            }

        result = await work_pool.submit_background(
            _lane_p1_work,
            job_type="memory_sleep_consolidation",
            ctx=active,
        )
        await asyncio.to_thread(
            self._set_job_state,
            jid,
            "completed",
            100,
            f"Consolidated {result['facts_consolidated']} fact(s), {result['procedural_proposals']} proposal(s)",
            active,
        )
        return result

    def enqueue_sleep_consolidation(
        self,
        messages: list[BaseMessage],
        *,
        session_id: str = "",
        agent_id: str = "default",
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Fire-and-forget dispatch returning immediately in `< 1ms` (`0ms` chat path stall!)."""
        active = ctx or get_tenant_context()
        jid = f"memjob_{uuid.uuid4().hex[:10]}"
        snapshot = list(messages)

        async def _runner() -> None:
            try:
                await self.run_consolidation_now(
                    snapshot,
                    session_id=session_id,
                    agent_id=agent_id,
                    job_id=jid,
                    ctx=active,
                )
            except Exception as exc:
                logger.warning("Background sleep-time consolidation %s failed: %s", jid, exc)

        task = asyncio.create_task(_runner())
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return {"job_id": jid, "status": "queued", "lane": "background_p1"}

    async def flush_inflight(self) -> None:
        if self._inflight_tasks:
            await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)


memory_consolidator = SleepTimeMemoryConsolidator()

"""Phase 5 — Context-Isolated Specialist Sub-Agents (`agent/subagents.py`).

Enables any parent `nimAgents` agent to spawn isolated specialist child sessions
(`researcher`, `coder`, `analyst`, `verifier`, or parallel multi-agent fan-out via `asyncio.gather`)
that execute complex multi-tool trajectories in a clean child context window and return ONLY:
  - A compact distilled `< 1,500`-token synthesis so the parent context window stays ultra-clean, PLUS
  - Structured step-by-step sub-agent telemetry persisted to `subagent_runs` and rendered as an
    expandable **Sub-Agent Execution Trace** card in the Studio Chat UI!
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from agent.registry import count_text_tokens
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db

MAX_SUBAGENT_DEPTH = 2
MAX_DISTILLED_TOKENS = 1500

SPECIALIST_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a Specialist Research Sub-Agent spawned by nimAgents. "
        "Gather high-signal evidence using your tools, cross-verify facts, and return a dense, "
        "structured synthesis under 1,200 tokens with concrete metrics and findings."
    ),
    "coder": (
        "You are a Specialist Engineering & Sandbox Sub-Agent spawned by nimAgents. "
        "Use `orchestrate_tools` / calculation / file tools to verify code and computations empirically, "
        "then return exact verified results and root causes concisely."
    ),
    "analyst": (
        "You are a Specialist Data & Telemetry Analyst Sub-Agent spawned by nimAgents. "
        "Inspect artifacts and data systematically, compute exact aggregates/percentiles, and return "
        "a crisp executive table + key takeaways."
    ),
    "verifier": (
        "You are an Independent Verification Sub-Agent spawned by nimAgents. "
        "Audit assumptions, check edge cases and arithmetic, and list confirmed facts vs defects."
    ),
}


def _truncate_to_token_budget(text: str, max_tokens: int = MAX_DISTILLED_TOKENS) -> str:
    if count_text_tokens(text) <= max_tokens:
        return text
    char_limit = max_tokens * 4
    return text[:char_limit].rstrip() + "\n\n...[Distilled to 1.5K-token sub-agent budget]"


async def run_isolated_subagent(
    *,
    specialist_role: str,
    objective: str,
    model_getter: Any,
    child_tools: list[BaseTool],
    parent_session_id: str = "",
    max_steps: int = 5,
    current_depth: int = 0,
    ctx: TenantContext | None = None,
) -> dict[str, Any]:
    """Execute a specialist child ReAct loop in a completely isolated context window."""
    if current_depth >= MAX_SUBAGENT_DEPTH:
        return {
            "type": "subagent_trace",
            "status": "blocked_max_depth",
            "specialist_role": specialist_role,
            "objective": objective,
            "distilled_output": f"Max sub-agent recursion depth ({MAX_SUBAGENT_DEPTH}) reached; handle directly.",
            "steps": [],
            "tokens_used": 0,
            "duration_ms": 0.0,
        }

    active = ctx or get_tenant_context()
    role_key = (specialist_role or "researcher").strip().lower()
    sys_prompt = SPECIALIST_PROMPTS.get(
        role_key,
        f"You are a Specialist `{specialist_role}` Sub-Agent spawned by nimAgents. Solve the assigned objective concisely.",
    )
    child_session_id = f"sub_{uuid.uuid4().hex[:10]}"
    run_pk = f"subrun_{uuid.uuid4().hex[:12]}"
    t0 = time.perf_counter()

    # Never pass sub-agent spawner tools recursively when near depth limit
    safe_tools = [
        t
        for t in child_tools
        if t.name not in ("delegate_to_subagent", "delegate_parallel_subagents")
    ]
    tools_by_name = {t.name: t for t in safe_tools}

    llm = model_getter()
    bound = llm.bind_tools(safe_tools) if safe_tools else llm

    messages: list[BaseMessage] = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=objective),
    ]
    trace_steps: list[dict[str, Any]] = []
    final_answer = ""

    for step_idx in range(1, max(1, min(max_steps, 8)) + 1):
        ai_msg: AIMessage = await bound.ainvoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if not tool_calls:
            final_answer = str(ai_msg.content or "").strip()
            trace_steps.append(
                {
                    "step": step_idx,
                    "action": "final_synthesis",
                    "summary": final_answer[:300],
                }
            )
            break

        for tc in tool_calls:
            t_name = tc.get("name", "")
            t_args = tc.get("args") or {}
            t_id = tc.get("id") or f"subcall_{step_idx}"
            target_tool = tools_by_name.get(t_name)
            if target_tool is None:
                obs = f"Tool '{t_name}' is not authorized in this sub-agent."
            else:
                try:
                    obs_raw = await target_tool.ainvoke(t_args)
                    obs = str(obs_raw)
                except Exception as exc:
                    obs = f"Tool execution error: {exc}"

            trace_steps.append(
                {
                    "step": step_idx,
                    "action": "tool_call",
                    "tool": t_name,
                    "args": t_args,
                    "observation_preview": obs[:280],
                }
            )
            messages.append(ToolMessage(content=obs, tool_call_id=t_id, name=t_name))

    if not final_answer:
        final_answer = str(messages[-1].content or "Sub-agent finished steps without text output.")

    distilled = _truncate_to_token_budget(final_answer, MAX_DISTILLED_TOKENS)
    raw_traj_tokens = sum(count_text_tokens(str(m.content)) for m in messages)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def _persist_subrun() -> None:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO subagent_runs
                   (id, tenant_id, workspace_id, parent_session_id, child_session_id,
                    specialist_role, objective, status, tokens_used, steps_count,
                    duration_ms, distilled_output, trace_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?)""",
                (
                    run_pk,
                    active.tenant_id,
                    active.workspace_id,
                    parent_session_id,
                    child_session_id,
                    role_key,
                    objective[:1000],
                    raw_traj_tokens,
                    len(trace_steps),
                    elapsed_ms,
                    distilled,
                    json.dumps(trace_steps, ensure_ascii=False),
                ),
            )
            conn.commit()

    await asyncio.to_thread(_persist_subrun)

    return {
        "type": "subagent_trace",
        "id": run_pk,
        "parent_session_id": parent_session_id,
        "child_session_id": child_session_id,
        "specialist_role": role_key,
        "objective": objective,
        "status": "completed",
        "child_context_tokens_isolated": raw_traj_tokens,
        "parent_tokens_returned": count_text_tokens(distilled),
        "steps_count": len(trace_steps),
        "duration_ms": elapsed_ms,
        "distilled_output": distilled,
        "steps": trace_steps,
    }


def build_subagent_delegation_tools(
    *,
    model_getter: Any,
    child_tools: list[BaseTool],
    parent_session_id: str = "",
    current_depth: int = 0,
    ctx: TenantContext | None = None,
) -> list[BaseTool]:
    """Create `delegate_to_subagent` and parallel fan-out `delegate_parallel_subagents` tools."""
    active = ctx or get_tenant_context()

    @tool
    async def delegate_to_subagent(
        specialist_role: str,
        objective: str,
        max_steps: int = 5,
    ) -> str:
        """Spawn a context-isolated specialist child agent (`researcher`, `coder`, `analyst`, `verifier`)
        so noisy intermediate research/code exploration stays out of your main context window!"""
        result = await run_isolated_subagent(
            specialist_role=specialist_role,
            objective=objective,
            model_getter=model_getter,
            child_tools=child_tools,
            parent_session_id=parent_session_id,
            max_steps=max_steps,
            current_depth=current_depth + 1,
            ctx=active,
        )
        return json.dumps(result, ensure_ascii=False)

    @tool
    async def delegate_parallel_subagents(tasks_json: str) -> str:
        """Run up to 4 independent specialist sub-agents concurrently (`tasks_json`: JSON array of
        `{"specialist_role": "researcher", "objective": "..."}`) and return their distilled findings."""
        try:
            parsed = json.loads(tasks_json)
            if not isinstance(parsed, list):
                return "Error: `tasks_json` must be a JSON list of `{specialist_role, objective}` objects."
        except json.JSONDecodeError as exc:
            return f"Invalid JSON in `tasks_json`: {exc}"

        coros = [
            run_isolated_subagent(
                specialist_role=str(item.get("specialist_role", "researcher")),
                objective=str(item.get("objective", "")),
                model_getter=model_getter,
                child_tools=child_tools,
                parent_session_id=parent_session_id,
                max_steps=int(item.get("max_steps", 4)),
                current_depth=current_depth + 1,
                ctx=active,
            )
            for item in parsed[:4]
            if isinstance(item, dict) and str(item.get("objective", "")).strip()
        ]
        results = await asyncio.gather(*coros)
        return json.dumps(
            {
                "type": "subagent_parallel_trace",
                "subagent_count": len(results),
                "results": results,
            },
            ensure_ascii=False,
        )

    return [delegate_to_subagent, delegate_parallel_subagents]

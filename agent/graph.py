"""LangGraph graph factory that builds a multi-tenant nimAgents ReAct agent with:
  - Unified `ToolRegistry` (`ToolSpec`, concrete `input_examples`, timeouts, circuit breakers & HITL)
  - Provider-agnostic KV-Cache-Safe Deferred Tool Loading (`tool_loading_mode = 'auto' | 'deferred' | 'eager'`)
    using dynamic `discovered_tool_ids` state updates (`Command`) via sub-5ms Hybrid BM25 + Cosine `tool_search`
  - Zero-Trust Sandbox Hands Programmatic Tool Calling (`orchestrate_tools`) over Unix socket RPC
  - Phase 4 3-Tier Progressive `SKILL.md` Skills (`activate_skill`, `read_skill_resource`)
  - Phase 4 5-Stage Graduated Context Engine (`read_artifact`, `<think>` stripping, batch tool clearing,
    pre-compact fact extraction, rolling `<compaction_checkpoint>`, and 95% hard ceiling guard)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
import re
import time
from typing import Annotated, Any

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from agent.context_engine import (
    ContextPolicy,
    build_read_artifact_tool,
    context_engine,
)
from agent.inference_gateway import compute_prefix_hash
from agent.model import get_model, sanitize_multimodal_messages
from agent.registry import (
    PINNED_CORE_TOOL_IDS,
    ToolSpec,
    build_deferred_directory_prompt,
    count_text_tokens,
    tool_registry,
)
from agent.sandbox_hands import build_orchestrate_tools_tool
from agent.skills import (
    build_skill_runtime_tools,
    build_tier1_skill_directory_prompt,
    build_tier2_active_skills_block,
    load_tenant_skills,
)
from api.core.tenancy import (
    TenantContext,
    build_meili_tenant_filter,
    get_tenant_context,
    sanitize_tenant_identifier,
    tenant_scope,
)
from api.db import ensure_tenant_guardrails_seeded, get_db

logger = logging.getLogger("nimagents.graph")

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY") or None

# Pooled MCP tool definitions with 60s TTL to avoid per-turn OS subprocess respawning
_MCP_TOOL_CACHE: dict[str, tuple[float, list[Any]]] = {}
_MCP_CACHE_TTL_S = 60.0

AUTO_DEFER_TOKEN_THRESHOLD = 1200
AUTO_DEFER_TOOL_COUNT_THRESHOLD = 6

# Byte-stable system prompt prefix to preserve GPU KV cache reuse across turns & users.
DEFAULT_SYSTEM_PROMPT = (
    "You are {agent_name}, a helpful and accurate AI assistant powered by nimAgents. "
    "Use the available tools thoughtfully when needed and cite retrieved knowledge clearly."
)


def _merge_ordered_unique_ids(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Append-only reducer preserving deterministic creation order for GPU KV cache stability!"""
    seen: list[str] = list(left or [])
    for item in right or []:
        if item and item not in seen:
            seen.append(item)
    return seen


class NimAgentState(MessagesState):
    """Multi-turn state extending `MessagesState` with KV-stable discovery, skills, and compaction."""

    discovered_tool_ids: Annotated[list[str], _merge_ordered_unique_ids]
    activated_skill_ids: Annotated[list[str], _merge_ordered_unique_ids]
    compaction_checkpoint: str


def _build_system_prompt(raw: str | None, agent_name: str = "nimAgents Assistant") -> str:
    """Build byte-stable Block 1 system prompt (never injecting per-second clock jitter into Block 1!)."""
    template = raw or DEFAULT_SYSTEM_PROMPT
    try:
        kwargs: dict[str, str] = {"agent_name": agent_name}
        if "{system_time}" in template:
            # Keep Block 1 100% deterministic across turns & users so GPU KV caches stay warm!
            kwargs["system_time"] = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


def _resolve_model_context_window(model_id: str | None, ctx: TenantContext, override: int | None = None) -> int:
    if override and override >= 1024:
        return int(override)
    if model_id:
        try:
            with get_db() as conn:
                row = conn.execute(
                    "SELECT context_window FROM models WHERE id = ? AND tenant_id = ?",
                    (model_id, ctx.tenant_id),
                ).fetchone()
            if row and row["context_window"]:
                return max(2048, int(row["context_window"]))
        except Exception:
            pass
    return 131072


def _pre_model_sanitize(state: dict[str, Any]) -> dict[str, Any]:
    """Sanitize multimodal inputs before forwarding messages to any model endpoint."""
    messages = state.get("messages", [])
    return {"llm_input_messages": sanitize_multimodal_messages(messages)}


def _extract_run_tenant(config: RunnableConfig) -> TenantContext:
    configurable = config.get("configurable", {})
    base = get_tenant_context()
    tid = configurable.get("x-tenant-id") or configurable.get("tenant_id") or base.tenant_id
    wid = configurable.get("x-workspace-id") or configurable.get("workspace_id") or base.workspace_id
    uid = configurable.get("x-user-id") or configurable.get("user_id") or base.user_id
    tier = configurable.get("x-plan-tier") or configurable.get("plan_tier") or base.plan_tier
    return TenantContext(
        tenant_id=sanitize_tenant_identifier(tid, base.tenant_id),
        workspace_id=sanitize_tenant_identifier(wid, base.workspace_id),
        user_id=sanitize_tenant_identifier(uid, base.user_id),
        plan_tier=sanitize_tenant_identifier(tier, base.plan_tier).lower(),
    )


def _parse_agent_config(config: RunnableConfig) -> dict[str, Any]:
    configurable = config.get("configurable", {})

    header_value = configurable.get("x-agent-config")
    if header_value:
        try:
            return json.loads(header_value)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "agent_name": configurable.get("agent_name", "nimAgents Assistant"),
        "system_prompt": configurable.get("system_prompt"),
        "enabled_prebuilt_tools": configurable.get("enabled_prebuilt_tools", []),
        "tool_configs": configurable.get("tool_configs", {}),
        "mcp_servers": configurable.get("mcp_servers", []),
        "model_id": configurable.get("model_id", ""),
        "rag_source_ids": configurable.get("rag_source_ids", []),
        "local_rag_source_ids": configurable.get("local_rag_source_ids", []),
        "lexical_rag_source_ids": configurable.get("lexical_rag_source_ids", []),
        "tool_loading_mode": configurable.get("tool_loading_mode", "auto"),
        "enable_programmatic_tools": configurable.get("enable_programmatic_tools", True),
        "approval_required_tools": configurable.get("approval_required_tools", []),
        "enabled_skill_ids": configurable.get("enabled_skill_ids"),
        "context_window": configurable.get("context_window"),
        "session_memory": configurable.get("session_memory", ""),
    }


async def _load_mcp_tools(mcp_servers: list[dict[str, Any]]) -> list[BaseTool]:
    if not mcp_servers:
        return []

    cache_key = "http:" + hashlib.sha256(
        json.dumps(mcp_servers, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cached = _MCP_TOOL_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and cached[0] > now:
        return list(cached[1])

    server_configs: dict[str, dict[str, Any]] = {}
    selected_by_server: dict[str, set[str]] = {}

    for idx, server in enumerate(mcp_servers):
        url = server.get("url", "")
        if not url:
            continue
        name = f"mcp_{idx}"
        server_configs[name] = {
            "transport": "streamable_http",
            "url": url,
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        }
        selected = server.get("selected_tools")
        if selected:
            selected_by_server[name] = set(selected)

    if not server_configs:
        return []

    client = MultiServerMCPClient(server_configs)
    all_tools = await client.get_tools()

    if selected_by_server:
        filtered = []
        for t in all_tools:
            for _srv_name, allowed in selected_by_server.items():
                if t.name in allowed:
                    filtered.append(t)
                    break
        result = filtered
    else:
        result = all_tools

    _MCP_TOOL_CACHE[cache_key] = (now + _MCP_CACHE_TTL_S, result)
    return list(result)


def _build_rag_tool(rag_source_ids: list[str], ctx: TenantContext):
    @tool
    async def rag_search(query: str) -> str:
        """Search external NVIDIA NIM RAG knowledge collections for relevant documents."""
        try:
            from api.routes.rag import query_nim_rag_row

            rows_to_query: list[dict[str, Any]] = []
            with get_db() as conn:
                for sid in rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM rag_sources WHERE id=? AND tenant_id=? AND status='ready'",
                        (sid, ctx.tenant_id),
                    ).fetchone()
                    if row:
                        rows_to_query.append(dict(row))

            all_results: list[str] = []
            for row_dict in rows_to_query:
                try:
                    docs = await asyncio.to_thread(query_nim_rag_row, row_dict, query, 5)
                    for item in docs:
                        text = item.get("page_content", "")
                        if text:
                            all_results.append(text)
                except Exception:
                    continue

            if not all_results:
                return "No relevant documents found in the knowledge base."
            return "\n---\n".join(all_results[:10])
        except Exception as exc:
            return f"RAG search error: {exc}"

    return rag_search


def _build_local_rag_tool(local_rag_source_ids: list[str], ctx: TenantContext):
    @tool
    async def local_rag_search(query: str) -> str:
        """Search tenant-isolated Qdrant vector documents using batched embeddings and NVIDIA reranking."""
        try:
            from api.routes.local_rag import search_local_source_async

            rows_to_query: list[dict[str, Any]] = []
            with get_db() as conn:
                for sid in local_rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM local_rag_sources WHERE id=? AND tenant_id=? AND status='ready'",
                        (sid, ctx.tenant_id),
                    ).fetchone()
                    if row:
                        rows_to_query.append(dict(row))

            all_results: list[str] = []
            for row_dict in rows_to_query:
                try:
                    docs = await search_local_source_async(
                        row_dict, query=query, k=4, ctx=ctx
                    )
                    for d in docs:
                        all_results.append(d["page_content"])
                except Exception:
                    continue

            if not all_results:
                return "No relevant documents found in local knowledge base."
            return "\n---\n".join(all_results[:8])
        except Exception as exc:
            return f"Local RAG search error: {exc}"

    return local_rag_search


def _build_lexical_rag_tool(lexical_rag_source_ids: list[str], ctx: TenantContext):
    @tool
    async def lexical_rag_search(query: str) -> str:
        """Search uploaded tenant documents using Meilisearch BM25 lexical matching."""
        try:
            import meilisearch

            client = meilisearch.Client(MEILI_URL, MEILI_MASTER_KEY)
            rows_to_query: list[dict[str, Any]] = []
            with get_db() as conn:
                for sid in lexical_rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM lexical_rag_sources WHERE id=? AND tenant_id=? AND status='ready'",
                        (sid, ctx.tenant_id),
                    ).fetchone()
                    if row:
                        rows_to_query.append(dict(row))

            all_results: list[str] = []
            for row_dict in rows_to_query:
                try:
                    index = client.index(row_dict["meili_index"])
                    filter_expr = build_meili_tenant_filter(ctx, source_id=row_dict["id"])
                    results = await asyncio.to_thread(
                        index.search, query, {"limit": 5, "filter": filter_expr}
                    )
                    for hit in results.get("hits", []):
                        all_results.append(hit.get("content", ""))
                except Exception:
                    continue

            if not all_results:
                return "No relevant documents found via keyword search."
            return "\n---\n".join(all_results[:5])
        except Exception as exc:
            return f"Lexical RAG search error: {exc}"

    return lexical_rag_search


async def _load_meilisearch_mcp_tools(
    tool_configs: dict[str, dict[str, str]],
) -> list[BaseTool]:
    """Load meilisearch-mcp tools with 60s pooled tool-list caching instead of per-turn spawns."""
    meili_cfg = tool_configs.get("meilisearch_mcp", {})
    meili_addr = meili_cfg.get("MEILI_HTTP_ADDR") or MEILI_URL
    meili_key = meili_cfg.get("MEILI_MASTER_KEY") or (MEILI_MASTER_KEY or "")

    cache_key = "meili_mcp:" + compute_prefix_hash(meili_addr, meili_key)
    cached = _MCP_TOOL_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and cached[0] > now:
        return list(cached[1])

    env = {**os.environ, "MEILI_HTTP_ADDR": meili_addr}
    if meili_key:
        env["MEILI_MASTER_KEY"] = meili_key

    server_configs = {
        "meilisearch": {
            "transport": "stdio",
            "command": "meilisearch-mcp",
            "args": [],
            "env": env,
        }
    }

    try:
        client = MultiServerMCPClient(server_configs)
        loaded = await client.get_tools()
        _MCP_TOOL_CACHE[cache_key] = (now + _MCP_CACHE_TTL_S, loaded)
        return list(loaded)
    except Exception as exc:
        logger.warning("Failed to load meilisearch-mcp tools: %s", exc)
        return []


def _check_guardrails(
    text: str,
    thread_id: str = "",
    assistant_id: str = "",
    ctx: TenantContext | None = None,
) -> dict:
    """Run text against active tenant guardrail rules. Returns {passed, hits, cleaned_text}."""
    import re as _re

    active = ctx or get_tenant_context()
    try:
        with get_db() as conn:
            ensure_tenant_guardrails_seeded(conn, active.tenant_id)
            rules = conn.execute(
                "SELECT * FROM guardrail_rules WHERE tenant_id = ? AND enabled = 1",
                (active.tenant_id,),
            ).fetchall()
    except Exception:
        return {"passed": True, "hits": [], "cleaned_text": text}

    hits = []
    cleaned = text
    for rule in rules:
        pattern = rule["pattern"]
        if not pattern:
            continue
        try:
            match = _re.search(pattern, text)
        except _re.error:
            continue
        if match:
            hits.append(
                {
                    "rule_id": rule["id"],
                    "rule_name": rule["name"],
                    "category": rule["category"],
                    "action": rule["action"],
                    "matched_text": match.group()[:200],
                }
            )
            if rule["action"] == "redact":
                cleaned = _re.sub(pattern, "[REDACTED]", cleaned)

    if hits:
        try:
            from datetime import datetime as _dt, timezone as _tz
            import uuid as _uuid

            now = _dt.now(_tz.utc).isoformat()
            with get_db() as conn:
                for hit in hits:
                    conn.execute(
                        """INSERT INTO guardrail_events
                           (id, tenant_id, workspace_id, rule_id, rule_name, category, action,
                            matched_text, thread_id, assistant_id, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(_uuid.uuid4()),
                            active.tenant_id,
                            active.workspace_id,
                            hit["rule_id"],
                            hit["rule_name"],
                            hit["category"],
                            hit["action"],
                            hit["matched_text"],
                            thread_id,
                            assistant_id,
                            now,
                        ),
                    )
                conn.commit()
        except Exception:
            pass

    has_block = any(h["action"] == "block" for h in hits)
    return {"passed": not has_block, "hits": hits, "cleaned_text": cleaned}


def _build_dynamic_tool_search_tool(
    deferred_specs: list[ToolSpec],
    ctx: TenantContext,
) -> BaseTool:
    """Create the `tool_search` LangGraph tool updating `discovered_tool_ids` via `Command`."""

    @tool
    def tool_search(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        mode: str = "hybrid",
        limit: int = 5,
    ) -> Command:
        """Discover and activate specialized deferred tools on demand (`mode`: 'hybrid', 'regex', or 'select').
        Activated tools immediately become directly callable on your very next step!"""
        try:
            hits = tool_registry.search_tools(
                query,
                mode=mode,
                limit=limit,
                candidates=deferred_specs,
                ctx=ctx,
            )
        except (ValueError, re.error) as exc:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Invalid tool_search query: {exc}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        activated_ids = [h["id"] for h in hits]
        lines = [
            f"Activated {len(hits)} tool(s) for immediate use ({mode} search):"
        ]
        for h in hits:
            lines.append(f"- `{h['name']}` (`{h['namespaced_id']}`): {h['description']}")

        return Command(
            update={
                "discovered_tool_ids": activated_ids,
                "messages": [
                    ToolMessage(
                        content="\n".join(lines),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return tool_search


async def make_graph(config: RunnableConfig):
    run_ctx = _extract_run_tenant(config)
    configurable = config.get("configurable", {})
    thread_id = str(configurable.get("thread_id") or "")

    with tenant_scope(
        tenant_id=run_ctx.tenant_id,
        workspace_id=run_ctx.workspace_id,
        user_id=run_ctx.user_id,
        plan_tier=run_ctx.plan_tier,
    ):
        agent_config = _parse_agent_config(config)
        agent_name = agent_config.get("agent_name", "nimAgents Assistant")

        enabled_ids: list[str] = list(agent_config.get("enabled_prebuilt_tools") or [])
        tool_configs: dict[str, dict[str, str]] = agent_config.get("tool_configs") or {}
        approval_ids = set(agent_config.get("approval_required_tools") or [])
        loading_mode = str(agent_config.get("tool_loading_mode") or "auto").lower()
        enable_programmatic = bool(agent_config.get("enable_programmatic_tools", True))

        has_meili_mcp = "meilisearch_mcp" in enabled_ids
        regular_tool_ids = [t for t in enabled_ids if t != "meilisearch_mcp"]
        if not regular_tool_ids and not agent_config.get("mcp_servers"):
            regular_tool_ids = ["calculator", "current_datetime"]

        all_candidate_specs = tool_registry.list_specs(
            ctx=run_ctx, only_ids=set(regular_tool_ids)
        )
        total_schema_tokens = sum(s.token_cost for s in all_candidate_specs)

        use_deferred = (
            loading_mode == "deferred"
            or (
                loading_mode == "auto"
                and (
                    total_schema_tokens > AUTO_DEFER_TOKEN_THRESHOLD
                    or len(all_candidate_specs) > AUTO_DEFER_TOOL_COUNT_THRESHOLD
                )
            )
        )

        pinned_ids: list[str] = []
        deferred_specs: list[ToolSpec] = []
        deferred_tools_by_id: dict[str, list[BaseTool]] = {}
        spec_for_tool_name: dict[str, ToolSpec] = {}
        spec_by_id = {s.id: s for s in all_candidate_specs}

        if use_deferred:
            for spec in all_candidate_specs:
                if not spec.defer_loading or spec.id in PINNED_CORE_TOOL_IDS:
                    pinned_ids.append(spec.id)
                else:
                    deferred_specs.append(spec)
                    insts = tool_registry.instantiate_tools(
                        [spec.id],
                        tool_configs=tool_configs,
                        approval_required_ids=approval_ids,
                        ctx=run_ctx,
                    )
                    for inst in insts:
                        spec_for_tool_name[inst.name] = spec
                    deferred_tools_by_id[spec.id] = insts
        else:
            pinned_ids = [s.id for s in all_candidate_specs]

        pinned_tools: list[BaseTool] = []
        for pid in pinned_ids:
            insts = tool_registry.instantiate_tools(
                [pid],
                tool_configs=tool_configs,
                approval_required_ids=approval_ids,
                ctx=run_ctx,
            )
            for inst in insts:
                spec_for_tool_name[inst.name] = spec_by_id[pid]
            pinned_tools.extend(insts)

        mcp_tools = await _load_mcp_tools(agent_config.get("mcp_servers", []))
        base_tools: list[BaseTool] = pinned_tools + mcp_tools

        rag_ids = agent_config.get("rag_source_ids", [])
        if rag_ids:
            base_tools.append(_build_rag_tool(rag_ids, run_ctx))

        local_rag_ids = agent_config.get("local_rag_source_ids", [])
        if local_rag_ids:
            base_tools.append(_build_local_rag_tool(local_rag_ids, run_ctx))

        lexical_ids = agent_config.get("lexical_rag_source_ids", [])
        if lexical_ids:
            base_tools.append(_build_lexical_rag_tool(lexical_ids, run_ctx))

        if has_meili_mcp:
            meili_mcp_tools = await _load_meilisearch_mcp_tools(tool_configs)
            base_tools.extend(meili_mcp_tools)

        @tool
        def check_guardrails(text: str) -> str:
            """Check text against security guardrail rules before processing. Returns 'ok' if safe, or a warning/block message."""
            result = _check_guardrails(text, thread_id=thread_id, ctx=run_ctx)
            if not result["passed"]:
                blocked = [h["rule_name"] for h in result["hits"] if h["action"] == "block"]
                return f"BLOCKED by guardrails: {', '.join(blocked)}. Do not process this content."
            if result["hits"]:
                warnings = [f"{h['rule_name']} ({h['action']})" for h in result["hits"]]
                return f"Guardrail warnings: {', '.join(warnings)}. Content was sanitized."
            return "ok"

        base_tools.append(check_guardrails)

        # Phase 4: JIT Artifact Reader (`read_artifact`) + 3-Tier Progressive `SKILL.md` Tools
        artifact_tool = build_read_artifact_tool(run_ctx)
        base_tools.append(artifact_tool)

        explicit_skill_ids = agent_config.get("enabled_skill_ids")
        allowed_skill_filter = set(explicit_skill_ids) if explicit_skill_ids is not None else None
        tenant_skills = load_tenant_skills(
            run_ctx,
            only_enabled=True,
            allowed_skill_ids_or_slugs=allowed_skill_filter,
        )
        skills_by_slug = {s["slug"]: s for s in tenant_skills}
        if tenant_skills:
            skill_tools = build_skill_runtime_tools(tenant_skills)
            base_tools.extend(skill_tools)

        # Build complete executable registry map for ToolNode AND Zero-Trust Sandbox Hands RPC!
        all_executable_tools: list[BaseTool] = list(base_tools)
        for tool_list in deferred_tools_by_id.values():
            all_executable_tools.extend(tool_list)

        if enable_programmatic:
            sandbox_allowed: dict[str, BaseTool] = {}
            for t in all_executable_tools:
                sp = spec_for_tool_name.get(t.name)
                if sp is not None:
                    if (
                        "code_execution" in sp.allowed_callers
                        and not sp.requires_approval
                        and sp.id not in approval_ids
                        and t.name not in approval_ids
                    ):
                        sandbox_allowed[t.name] = t
                elif t.name in ("rag_search", "local_rag_search", "lexical_rag_search", "read_artifact"):
                    sandbox_allowed[t.name] = t
            if sandbox_allowed:
                orch_tool = build_orchestrate_tools_tool(sandbox_allowed, ctx=run_ctx)
                base_tools.append(orch_tool)
                all_executable_tools.append(orch_tool)

        if deferred_specs:
            search_tool = _build_dynamic_tool_search_tool(deferred_specs, run_ctx)
            base_tools.append(search_tool)
            all_executable_tools.append(search_tool)

        # ── Block 1 Static Prefix (100% deterministic across ALL users & turns!) ──
        base_prompt = _build_system_prompt(
            agent_config.get("system_prompt"),
            agent_name,
        )
        skill_dir_suffix = build_tier1_skill_directory_prompt(tenant_skills)
        directory_suffix = build_deferred_directory_prompt(deferred_specs)
        system_prompt = f"{base_prompt}{skill_dir_suffix}{directory_suffix}"

        ctx_window = _resolve_model_context_window(
            agent_config.get("model_id"),
            run_ctx,
            override=agent_config.get("context_window"),
        )
        context_policy = ContextPolicy(context_window=ctx_window)
        session_memory_str = str(agent_config.get("session_memory") or "")

        prefix_hash = compute_prefix_hash(system_prompt)
        model = get_model(
            agent_config.get("model_id") or None,
            priority=1,
            prefix_hash=prefix_hash,
            thread_id=thread_id,
        )

        async def _agent_node(state: NimAgentState) -> dict[str, Any]:
            discovered = state.get("discovered_tool_ids") or []
            activated_skills = state.get("activated_skill_ids") or []
            prior_checkpoint = state.get("compaction_checkpoint") or ""

            active_turn_tools: list[BaseTool] = list(base_tools)
            seen_names = {t.name for t in active_turn_tools}
            for did in discovered:
                for dt in deferred_tools_by_id.get(did, []):
                    if dt.name not in seen_names:
                        active_turn_tools.append(dt)
                        seen_names.add(dt.name)

            bound_llm = model.bind_tools(active_turn_tools)
            clean_history = sanitize_multimodal_messages(state.get("messages", []))

            active_skills_md = build_tier2_active_skills_block(skills_by_slug, activated_skills)
            bound_schema_toks = sum(
                count_text_tokens(f"{t.name}:{getattr(t, 'description', '')}") + 32
                for t in active_turn_tools
            )

            projection = await context_engine.shape_context(
                static_system_prompt=system_prompt,
                session_memory_notes=session_memory_str,
                active_skills_block=active_skills_md,
                prior_compaction_checkpoint=prior_checkpoint,
                messages=clean_history,
                tool_schema_tokens=bound_schema_toks,
                policy=context_policy,
                ctx=run_ctx,
                thread_id=thread_id,
                agent_name=agent_name,
                deferred_tools_count=max(0, len(deferred_specs) - len(discovered)),
                active_skill_slugs=list(activated_skills),
            )

            response = await bound_llm.ainvoke(projection.to_model_messages())
            state_update: dict[str, Any] = {"messages": [response]}
            if projection.compacted_this_turn and projection.updated_checkpoint:
                state_update["compaction_checkpoint"] = projection.updated_checkpoint
            return state_update

        builder = StateGraph(NimAgentState)
        builder.add_node("agent", _agent_node)
        builder.add_node("tools", ToolNode(all_executable_tools))
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        return builder.compile()

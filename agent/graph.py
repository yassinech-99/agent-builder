"""LangGraph graph factory that builds a multi-tenant nimAgents ReAct agent from per-run config.

Config schema (passed via config["configurable"]):
    agent_name:              str
    system_prompt:           str
    enabled_prebuilt_tools:  list[str]
    tool_configs:            dict[str, dict[str, str]]  (per-tool secrets/params)
    mcp_servers:             list[dict]
    model_id:                str            (optional, falls back to DB default / .env)
    rag_source_ids:          list[str]      (optional, adds NVIDIA NIM RAG retriever tool)
    local_rag_source_ids:    list[str]      (optional, adds Qdrant vector RAG search tool)
    lexical_rag_source_ids:  list[str]      (optional, adds Meilisearch BM25 search tool)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
import os
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from agent.inference_gateway import compute_prefix_hash
from agent.model import get_model, sanitize_multimodal_messages
from agent.tools import get_tools_by_ids
from api.core.tenancy import (
    TenantContext,
    build_meili_tenant_filter,
    get_tenant_context,
    sanitize_tenant_identifier,
    tenant_scope,
)
from api.db import ensure_tenant_guardrails_seeded

logger = logging.getLogger("nimagents.graph")

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY") or None

# Pooled MCP tool definitions with 60s TTL to avoid per-turn OS subprocess respawning
_MCP_TOOL_CACHE: dict[str, tuple[float, list[Any]]] = {}
_MCP_CACHE_TTL_S = 60.0

# Byte-stable system prompt prefix to preserve GPU KV cache reuse across turns & users.
DEFAULT_SYSTEM_PROMPT = (
    "You are {agent_name}, a helpful and accurate AI assistant powered by nimAgents. "
    "Use the available tools thoughtfully when needed and cite retrieved knowledge clearly."
)


def _build_system_prompt(raw: str | None, agent_name: str = "nimAgents Assistant") -> str:
    template = raw or DEFAULT_SYSTEM_PROMPT
    try:
        kwargs: dict[str, str] = {"agent_name": agent_name}
        if "{system_time}" in template:
            kwargs["system_time"] = datetime.now(tz=UTC).isoformat()
        return template.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return template


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
    }


async def _load_mcp_tools(mcp_servers: list[dict[str, Any]]) -> list:
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
            from api.db import get_db
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
            from api.db import get_db
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
            from api.db import get_db

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
) -> list:
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
        from api.db import get_db

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
            from api.db import get_db

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

        enabled_tools = agent_config.get("enabled_prebuilt_tools", [])
        tool_configs = agent_config.get("tool_configs", {})

        has_meili_mcp = "meilisearch_mcp" in enabled_tools
        regular_tool_ids = [t for t in enabled_tools if t != "meilisearch_mcp"]

        prebuilt_tools = get_tools_by_ids(regular_tool_ids, tool_configs=tool_configs)
        mcp_tools = await _load_mcp_tools(agent_config.get("mcp_servers", []))

        tools = prebuilt_tools + mcp_tools

        rag_ids = agent_config.get("rag_source_ids", [])
        if rag_ids:
            tools.append(_build_rag_tool(rag_ids, run_ctx))

        local_rag_ids = agent_config.get("local_rag_source_ids", [])
        if local_rag_ids:
            tools.append(_build_local_rag_tool(local_rag_ids, run_ctx))

        lexical_ids = agent_config.get("lexical_rag_source_ids", [])
        if lexical_ids:
            tools.append(_build_lexical_rag_tool(lexical_ids, run_ctx))

        if has_meili_mcp:
            meili_mcp_tools = await _load_meilisearch_mcp_tools(tool_configs)
            tools.extend(meili_mcp_tools)

        if not tools:
            tools = get_tools_by_ids(["calculator", "current_datetime"])

        system_prompt = _build_system_prompt(
            agent_config.get("system_prompt"),
            agent_config.get("agent_name", "nimAgents Assistant"),
        )
        prefix_hash = compute_prefix_hash(system_prompt)
        model = get_model(
            agent_config.get("model_id") or None,
            priority=1,
            prefix_hash=prefix_hash,
            thread_id=thread_id,
        )

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

        tools.append(check_guardrails)

        return create_react_agent(
            model,
            tools,
            prompt=system_prompt,
            pre_model_hook=_pre_model_sanitize,
        )

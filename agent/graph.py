"""LangGraph graph factory that builds a ReAct agent from per-run config.

Config schema (passed via config["configurable"]):
    agent_name:              str
    system_prompt:           str
    enabled_prebuilt_tools:  list[str]
    tool_configs:            dict[str, dict[str, str]]  (per-tool secrets/params)
    mcp_servers:             list[dict]
    model_id:                str            (optional, falls back to .env)
    rag_source_ids:          list[str]      (optional, adds Bedrock KB retriever tool)
    local_rag_source_ids:    list[str]      (optional, adds local FAISS RAG search tool)
    lexical_rag_source_ids:  list[str]      (optional, adds lexical RAG search tool)
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

from agent.model import get_model
from agent.tools import get_tools_by_ids

MEILI_URL = os.getenv("MEILI_URL", "http://172.30.6.3:7700")

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant named {agent_name}. "
    "Current time: {system_time}. "
    "Use the tools available to you to answer the user's questions."
)


def _build_system_prompt(raw: str | None, agent_name: str = "NovaGrid Agents") -> str:
    template = raw or DEFAULT_SYSTEM_PROMPT
    try:
        return template.format(
            system_time=datetime.now(tz=UTC).isoformat(),
            agent_name=agent_name,
        )
    except (KeyError, IndexError):
        return template


def _parse_agent_config(config: RunnableConfig) -> dict[str, Any]:
    configurable = config.get("configurable", {})

    header_value = configurable.get("x-agent-config")
    if header_value:
        try:
            return json.loads(header_value)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "agent_name": configurable.get("agent_name", "Agent"),
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

    if not selected_by_server:
        return all_tools

    filtered = []
    for t in all_tools:
        for _srv_name, allowed in selected_by_server.items():
            if t.name in allowed:
                filtered.append(t)
                break
    return filtered


def _build_rag_tool(rag_source_ids: list[str]):
    @tool
    def rag_search(query: str) -> str:
        """Search the knowledge base for relevant information. Use this when the user asks about topics that might be in uploaded documents or knowledge bases."""
        try:
            import boto3
            from api.db import get_db_raw

            all_results = []
            default_region = os.getenv("AWS_REGION", "us-east-1")

            conn = get_db_raw()
            try:
                for sid in rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM rag_sources WHERE id=? AND status='ready'",
                        (sid,),
                    ).fetchone()
                    if not row:
                        continue
                    try:
                        region = row["region"] or default_region
                        client = boto3.client(
                            "bedrock-agent-runtime", region_name=region
                        )
                        response = client.retrieve(
                            knowledgeBaseId=row["knowledge_base_id"],
                            retrievalQuery={"text": query},
                            retrievalConfiguration={
                                "vectorSearchConfiguration": {"numberOfResults": 5}
                            },
                        )
                        for r in response.get("retrievalResults", []):
                            text = r.get("content", {}).get("text", "")
                            if text:
                                all_results.append(text)
                    except Exception:
                        continue
            finally:
                conn.close()

            if not all_results:
                return "No relevant documents found in the knowledge base."
            return "\n---\n".join(all_results[:10])
        except Exception as exc:
            return f"RAG search error: {exc}"

    return rag_search


def _build_local_rag_tool(local_rag_source_ids: list[str]):
    @tool
    def local_rag_search(query: str) -> str:
        """Search locally indexed documents using FAISS vector similarity. Use this for questions about uploaded or imported documents."""
        try:
            from api.db import get_db_raw
            from api.routes.local_rag import _get_embeddings
            from langchain_community.vectorstores import FAISS

            all_results = []
            conn = get_db_raw()
            try:
                for sid in local_rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM local_rag_sources WHERE id=? AND status='ready'",
                        (sid,),
                    ).fetchone()
                    if not row:
                        continue
                    try:
                        embeddings = _get_embeddings(row["embedding_model"])
                        store = FAISS.load_local(
                            row["index_path"],
                            embeddings,
                            allow_dangerous_deserialization=True,
                        )
                        docs = store.similarity_search(query, k=3)
                        for d in docs:
                            all_results.append(d.page_content)
                    except Exception:
                        continue
            finally:
                conn.close()

            if not all_results:
                return "No relevant documents found in local knowledge base."
            return "\n---\n".join(all_results[:8])
        except Exception as exc:
            return f"Local RAG search error: {exc}"

    return local_rag_search


def _build_lexical_rag_tool(lexical_rag_source_ids: list[str]):
    @tool
    def lexical_rag_search(query: str) -> str:
        """Search the knowledge base using keyword/lexical search via Meilisearch. Use this for exact term matching and keyword-based retrieval from uploaded documents."""
        try:
            import meilisearch
            from api.db import get_db_raw

            client = meilisearch.Client(MEILI_URL)
            all_results = []

            conn = get_db_raw()
            try:
                for sid in lexical_rag_source_ids:
                    row = conn.execute(
                        "SELECT * FROM lexical_rag_sources WHERE id=? AND status='ready'",
                        (sid,),
                    ).fetchone()
                    if not row:
                        continue
                    try:
                        index = client.index(row["meili_index"])
                        results = index.search(query, {"limit": 5})
                        for hit in results.get("hits", []):
                            all_results.append(hit.get("content", ""))
                    except Exception:
                        continue
            finally:
                conn.close()

            if not all_results:
                return "No relevant documents found via keyword search."
            return "\n---\n".join(all_results[:5])
        except Exception as exc:
            return f"Lexical RAG search error: {exc}"

    return lexical_rag_search


async def _load_meilisearch_mcp_tools(
    tool_configs: dict[str, dict[str, str]],
) -> list:
    """Spawn meilisearch-mcp as a stdio subprocess and return its tools."""
    meili_cfg = tool_configs.get("meilisearch_mcp", {})
    meili_addr = meili_cfg.get("MEILI_HTTP_ADDR", MEILI_URL)

    env = {**os.environ, "MEILI_HTTP_ADDR": meili_addr}
    meili_key = meili_cfg.get("MEILI_MASTER_KEY", "")
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
        return await client.get_tools()
    except Exception as exc:
        import logging
        logging.getLogger("novagrid.graph").warning(
            "Failed to load meilisearch-mcp tools: %s", exc
        )
        return []


def _check_guardrails(text: str, thread_id: str = "", assistant_id: str = "") -> dict:
    """Run text against active guardrail rules. Returns {passed, hits, cleaned_text}."""
    import re as _re
    try:
        from api.db import get_db_raw
        conn = get_db_raw()
        try:
            rules = conn.execute(
                "SELECT * FROM guardrail_rules WHERE enabled = 1"
            ).fetchall()
        finally:
            conn.close()
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
            hits.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "category": rule["category"],
                "action": rule["action"],
                "matched_text": match.group()[:200],
            })
            if rule["action"] == "redact":
                cleaned = _re.sub(pattern, "[REDACTED]", cleaned)

    if hits:
        try:
            import uuid as _uuid
            from datetime import datetime as _dt, timezone as _tz
            now = _dt.now(_tz.utc).isoformat()
            conn = get_db_raw()
            try:
                for hit in hits:
                    conn.execute(
                        """INSERT INTO guardrail_events
                           (id, rule_id, rule_name, category, action, matched_text, thread_id, assistant_id, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (str(_uuid.uuid4()), hit["rule_id"], hit["rule_name"],
                         hit["category"], hit["action"], hit["matched_text"],
                         thread_id, assistant_id, now),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    has_block = any(h["action"] == "block" for h in hits)
    return {"passed": not has_block, "hits": hits, "cleaned_text": cleaned}


async def make_graph(config: RunnableConfig):
    agent_config = _parse_agent_config(config)

    enabled_tools = agent_config.get("enabled_prebuilt_tools", [])
    tool_configs = agent_config.get("tool_configs", {})

    # Separate meilisearch_mcp from regular prebuilt tools (it uses stdio MCP, not a factory)
    has_meili_mcp = "meilisearch_mcp" in enabled_tools
    regular_tool_ids = [t for t in enabled_tools if t != "meilisearch_mcp"]

    prebuilt_tools = get_tools_by_ids(regular_tool_ids, tool_configs=tool_configs)
    mcp_tools = await _load_mcp_tools(agent_config.get("mcp_servers", []))

    tools = prebuilt_tools + mcp_tools

    # Bedrock Knowledge Bases
    rag_ids = agent_config.get("rag_source_ids", [])
    if rag_ids:
        tools.append(_build_rag_tool(rag_ids))

    # Local RAG (FAISS)
    local_rag_ids = agent_config.get("local_rag_source_ids", [])
    if local_rag_ids:
        tools.append(_build_local_rag_tool(local_rag_ids))

    # Lexical RAG
    lexical_ids = agent_config.get("lexical_rag_source_ids", [])
    if lexical_ids:
        tools.append(_build_lexical_rag_tool(lexical_ids))

    # Meilisearch MCP — chat with your data
    if has_meili_mcp:
        meili_mcp_tools = await _load_meilisearch_mcp_tools(tool_configs)
        tools.extend(meili_mcp_tools)

    if not tools:
        tools = get_tools_by_ids(["calculator", "current_datetime"])

    model = get_model(agent_config.get("model_id") or None)
    system_prompt = _build_system_prompt(
        agent_config.get("system_prompt"),
        agent_config.get("agent_name", "NovaGrid Agents"),
    )

    @tool
    def check_guardrails(text: str) -> str:
        """Check text against security guardrail rules before processing. Returns 'ok' if safe, or a warning/block message."""
        result = _check_guardrails(text)
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
    )

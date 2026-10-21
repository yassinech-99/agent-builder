"""Control Plane API for Unified `ToolRegistry`, Hybrid `ToolSearch`, Custom OpenAPI Tools & Sandbox Orchestration."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.registry import (
    ToolSpec,
    build_langchain_tool_from_custom_row,
    parse_openapi_spec_to_rows,
    sanitize_slug,
    tool_executor,
    tool_registry,
    validate_custom_tool_url,
)
from agent.sandbox_hands import execute_programmatic_script
from api.core.tenancy import get_tenant_context
from api.db import get_db

logger = logging.getLogger("nimagents.api.tools")
router = APIRouter(prefix="/api/tools", tags=["tools"])


class ToolSearchRequest(BaseModel):
    query: str = Field(default="", max_length=500)
    mode: str = Field(default="hybrid", pattern="^(hybrid|regex|select)$")
    limit: int = Field(default=8, ge=1, le=25)


class ToolExecuteRequest(BaseModel):
    tool_id: str = Field(..., min_length=1, max_length=128)
    args: dict[str, Any] = Field(default_factory=dict)
    tool_config: dict[str, str] = Field(default_factory=dict)


class ToolOrchestrateRequest(BaseModel):
    python_code: str = Field(..., min_length=1, max_length=16_000)
    authorized_tool_ids: list[str] = Field(
        default_factory=lambda: ["calculator", "current_datetime", "json_explorer"]
    )
    timeout_seconds: float = Field(default=15.0, ge=1.0, le=60.0)


class CustomToolCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(default="openapi", max_length=64)
    method: str = Field(default="GET", pattern="^(GET|POST|PUT|PATCH|DELETE)$")
    url_template: str = Field(..., min_length=8, max_length=1024)
    auth_header: str = Field(default="", max_length=512)
    input_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    input_examples: list[dict[str, Any]] = Field(default_factory=list)
    defer_loading: bool = True
    requires_approval: bool = False
    side_effect_level: str = Field(
        default="read", pattern="^(none|read|write|destructive)$"
    )


class OpenAPIImportRequest(BaseModel):
    spec: dict[str, Any]
    base_url_override: str = Field(default="", max_length=512)
    auth_header: str = Field(default="", max_length=512)
    namespace_prefix: str = Field(default="", max_length=64)


@router.get("/catalog")
async def list_unified_catalog():
    """Return all built-in, toolkit, and tenant custom tools with exact token costs & telemetry."""
    import asyncio

    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: tool_registry.list_catalog_dicts(ctx=ctx))


@router.get("/budget")
async def get_catalog_token_budget():
    """Return eager schema token cost vs deferred `tool_search` startup token cost."""
    import asyncio

    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: tool_registry.compute_token_budget_report(ctx=ctx))


@router.post("/search")
async def search_unified_tools(body: ToolSearchRequest):
    """Run hybrid BM25 + semantic cosine RRF (`hybrid`), `regex`, or `select` search over registered tools."""
    import asyncio

    ctx = get_tenant_context()
    t0 = time.perf_counter()
    try:
        hits = await asyncio.to_thread(
            lambda: tool_registry.search_tools(
                body.query,
                mode=body.mode,
                limit=body.limit,
                ctx=ctx,
            )
        )
    except (ValueError, re.error) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid tool search query: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    budget = await asyncio.to_thread(lambda: tool_registry.compute_token_budget_report(ctx=ctx))
    return {
        "mode": body.mode,
        "latency_ms": elapsed_ms,
        "budget": budget,
        "results": hits,
    }


@router.post("/execute")
async def execute_single_tool(body: ToolExecuteRequest):
    """Execute a single registered built-in or tenant custom tool via `ToolExecutor` for live Studio testing."""
    ctx = get_tenant_context()
    specs = tool_registry.list_specs(ctx=ctx, only_ids={body.tool_id})
    if not specs:
        raise HTTPException(status_code=404, detail=f"Tool '{body.tool_id}' not found")
    spec: ToolSpec = specs[0]

    # Instantiate raw underlying tool cleanly so telemetry records exact 1x call count
    from agent.tools import get_tools_by_ids

    if spec.source == "openapi":
        rows = {r["id"]: r for r in tool_registry.load_tenant_custom_rows(ctx)}
        _, raw_tool = build_langchain_tool_from_custom_row(rows[spec.id])
    else:
        raw_list = get_tools_by_ids(
            [body.tool_id],
            tool_configs={body.tool_id: body.tool_config} if body.tool_config else None,
        )
        if not raw_list:
            raise HTTPException(status_code=400, detail=f"Tool '{body.tool_id}' could not be initialized")
        raw_tool = raw_list[0]

    t0 = time.perf_counter()
    output = await tool_executor.execute_tool_instance(
        raw_tool,
        body.args,
        spec=spec,
        needs_approval=False,
    )
    return {
        "ok": True,
        "tool_id": spec.id,
        "tool_name": raw_tool.name,
        "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        "output": output,
        "telemetry": tool_executor.get_stats(spec.id),
    }


@router.post("/orchestrate")
async def orchestrate_sandbox_tools(body: ToolOrchestrateRequest):
    """Run an async Python orchestration script inside an isolated, secret-scrubbed Sandbox Hand."""
    ctx = get_tenant_context()
    specs = {s.id: s for s in tool_registry.list_specs(ctx=ctx)}
    allowed_ids = [
        tid
        for tid in body.authorized_tool_ids
        if tid in specs and "code_execution" in specs[tid].allowed_callers
    ]
    instances = tool_registry.instantiate_tools(allowed_ids, ctx=ctx)
    # Build approval-free inner tools for authorized programmatic RPC execution
    allowed_map = {}
    for tid in allowed_ids:
        spec = specs[tid]
        if tid in ("calculator", "current_datetime", "json_explorer") or spec.source == "openapi":
            for t in tool_registry.instantiate_tools([tid], ctx=ctx):
                allowed_map[t.name] = t
    if not allowed_map:
        for t in instances:
            allowed_map[t.name] = t

    res = await execute_programmatic_script(
        body.python_code,
        allowed_tools=allowed_map,
        ctx=ctx,
        timeout_seconds=body.timeout_seconds,
    )
    return res


@router.post("/custom", status_code=201)
async def create_custom_rest_tool(body: CustomToolCreateRequest):
    """Register a tenant-scoped custom HTTP/REST tool."""
    ctx = get_tenant_context()
    try:
        validated_url = validate_custom_tool_url(body.url_template)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    slug_name = sanitize_slug(body.name)[:48]
    cat_slug = sanitize_slug(body.category)[:32]
    tool_id = f"custom_{uuid.uuid4().hex[:12]}"
    headers = {"Authorization": body.auth_header.strip()} if body.auth_header.strip() else {}

    row = {
        "id": tool_id,
        "tenant_id": ctx.tenant_id,
        "workspace_id": ctx.workspace_id,
        "namespaced_id": f"{cat_slug}.{slug_name}",
        "name": slug_name,
        "description": body.description.strip(),
        "category": cat_slug,
        "method": body.method.upper(),
        "url_template": validated_url,
        "headers_json": json.dumps(headers),
        "input_schema_json": json.dumps(body.input_schema),
        "input_examples_json": json.dumps(body.input_examples),
        "defer_loading": int(body.defer_loading),
        "requires_approval": int(body.requires_approval),
        "side_effect_level": body.side_effect_level,
        "timeout_seconds": 20.0,
        "max_output_tokens": 2000,
        "source_spec_title": "Custom REST Tool",
    }

    with get_db() as conn:
        conn.execute(
            """INSERT INTO custom_tools
               (id, tenant_id, workspace_id, namespaced_id, name, description, category,
                method, url_template, headers_json, input_schema_json, input_examples_json,
                defer_loading, requires_approval, side_effect_level, timeout_seconds,
                max_output_tokens, source_spec_title)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(row.values()),
        )
        conn.commit()
        saved = dict(
            conn.execute(
                "SELECT * FROM custom_tools WHERE id = ? AND tenant_id = ?",
                (tool_id, ctx.tenant_id),
            ).fetchone()
        )

    spec, _ = build_langchain_tool_from_custom_row(saved)
    return spec.to_catalog_dict()


@router.post("/import-openapi", status_code=201)
async def import_openapi_tools(body: OpenAPIImportRequest):
    """Import an OpenAPI 3.x specification as tenant-scoped deferred `ToolSpec`s in 1 click."""
    ctx = get_tenant_context()
    try:
        rows = parse_openapi_spec_to_rows(
            body.spec,
            ctx=ctx,
            base_url_override=body.base_url_override,
            auth_header=body.auth_header,
            namespace_prefix=body.namespace_prefix,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(status_code=400, detail="No valid HTTP operations found in OpenAPI `paths`.")

    with get_db() as conn:
        for row in rows:
            conn.execute(
                """INSERT INTO custom_tools
                   (id, tenant_id, workspace_id, namespaced_id, name, description, category,
                    method, url_template, headers_json, input_schema_json, input_examples_json,
                    defer_loading, requires_approval, side_effect_level, timeout_seconds,
                    max_output_tokens, source_spec_title)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row.values()),
            )
        conn.commit()

    imported = [build_langchain_tool_from_custom_row(r)[0].to_catalog_dict() for r in rows]
    return {
        "imported_count": len(imported),
        "tools": imported,
    }


@router.delete("/custom/{tool_id}")
async def delete_custom_tool(tool_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM custom_tools WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (tool_id, ctx.tenant_id, ctx.workspace_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Custom tool not found")
    return {"ok": True}

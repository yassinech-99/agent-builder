"""MCP tool discovery endpoint."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("novagrid.api.mcp")
router = APIRouter(prefix="/api/mcp", tags=["mcp"])


class MCPDiscoverRequest(BaseModel):
    url: str = Field(..., min_length=1)


class MCPToolInfo(BaseModel):
    name: str
    description: str | None = None
    inputSchema: dict | None = None


class MCPDiscoverResponse(BaseModel):
    tools: list[MCPToolInfo]


@router.post("/discover", response_model=MCPDiscoverResponse)
async def discover_mcp_tools(req: MCPDiscoverRequest):
    """Send a JSON-RPC tools/list request to an MCP server and return discovered tools."""
    payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                req.url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"MCP server returned {exc.response.status_code}",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach MCP server: {exc}",
        ) from exc

    data = response.json()

    if "error" in data:
        raise HTTPException(
            status_code=502,
            detail=f"MCP server error: {data['error']}",
        )

    raw_tools = data.get("result", {}).get("tools", [])
    tools = [
        MCPToolInfo(
            name=t.get("name", ""),
            description=t.get("description"),
            inputSchema=t.get("inputSchema"),
        )
        for t in raw_tools
    ]
    return MCPDiscoverResponse(tools=tools)

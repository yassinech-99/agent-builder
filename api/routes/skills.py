"""Control Plane API for 3-Tier Progressive `SKILL.md` Agent Skills (`/api/skills`)."""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from agent.skills import (
    ensure_tenant_skills_seeded,
    load_tenant_skills,
    parse_skill_markdown,
    row_to_skill_dict,
    sanitize_skill_slug,
)
from api.core.tenancy import get_tenant_context
from api.db import get_db

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillCreateRequest(BaseModel):
    raw_skill_md: str = Field(default="", max_length=64_000)
    name: str = Field(default="", max_length=100)
    slug: str = Field(default="", max_length=64)
    description: str = Field(default="", max_length=400)
    category: str = Field(default="engineering", max_length=48)
    version: str = Field(default="1.0.0", max_length=24)
    tags: list[str] = Field(default_factory=list)
    allowed_tools: list[str] = Field(default_factory=list)
    body_markdown: str = Field(default="", max_length=60_000)
    resources: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class SkillUpdateRequest(BaseModel):
    raw_skill_md: str | None = Field(default=None, max_length=64_000)
    name: str | None = Field(default=None, max_length=100)
    description: str | None = Field(default=None, max_length=400)
    category: str | None = Field(default=None, max_length=48)
    version: str | None = Field(default=None, max_length=24)
    body_markdown: str | None = Field(default=None, max_length=60_000)
    resources: dict[str, str] | None = None
    enabled: bool | None = None


@router.get("")
async def list_skills():
    ctx = get_tenant_context()
    return await asyncio.to_thread(lambda: load_tenant_skills(ctx, only_enabled=False))


@router.post("", status_code=201)
async def create_skill(body: SkillCreateRequest):
    ctx = get_tenant_context()
    if body.raw_skill_md.strip():
        try:
            parsed = parse_skill_markdown(body.raw_skill_md)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        if not body.name.strip() or not body.body_markdown.strip():
            raise HTTPException(
                status_code=400,
                detail="Provide either `raw_skill_md` or both `name` and `body_markdown`.",
            )
        parsed = {
            "slug": sanitize_skill_slug(body.slug or body.name),
            "name": body.name.strip(),
            "description": (body.description.strip() or f"Specialized playbook for {body.name.strip()}"),
            "category": body.category.strip() or "engineering",
            "version": body.version.strip() or "1.0.0",
            "tags": body.tags,
            "allowed_tools": body.allowed_tools,
            "body_markdown": body.body_markdown.strip(),
            "resources": body.resources,
        }

    skill_id = f"skill_{uuid.uuid4().hex[:12]}"

    def _save() -> dict:
        with get_db() as conn:
            ensure_tenant_skills_seeded(conn, ctx)
            conn.execute(
                """INSERT INTO agent_skills
                   (id, tenant_id, workspace_id, slug, name, description, category, version,
                    tags_json, allowed_tools_json, body_markdown, resources_json, enabled)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    skill_id,
                    ctx.tenant_id,
                    ctx.workspace_id,
                    parsed["slug"],
                    parsed["name"],
                    parsed["description"],
                    parsed["category"],
                    parsed["version"],
                    json.dumps(parsed["tags"]),
                    json.dumps(parsed["allowed_tools"]),
                    parsed["body_markdown"],
                    json.dumps(parsed["resources"]),
                    int(body.enabled),
                ),
            )
            conn.commit()
            saved = conn.execute(
                "SELECT * FROM agent_skills WHERE id = ? AND tenant_id = ?",
                (skill_id, ctx.tenant_id),
            ).fetchone()
        return row_to_skill_dict(saved)

    return await asyncio.to_thread(_save)


@router.put("/{skill_id}")
async def update_skill(skill_id: str, body: SkillUpdateRequest):
    ctx = get_tenant_context()

    def _apply() -> dict:
        with get_db() as conn:
            existing = conn.execute(
                "SELECT * FROM agent_skills WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (skill_id, ctx.tenant_id, ctx.workspace_id),
            ).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Skill not found")
            cur = dict(existing)

            if body.raw_skill_md is not None and body.raw_skill_md.strip():
                parsed = parse_skill_markdown(body.raw_skill_md)
                cur.update(
                    {
                        "slug": parsed["slug"],
                        "name": parsed["name"],
                        "description": parsed["description"],
                        "category": parsed["category"],
                        "version": parsed["version"],
                        "tags_json": json.dumps(parsed["tags"]),
                        "allowed_tools_json": json.dumps(parsed["allowed_tools"]),
                        "body_markdown": parsed["body_markdown"],
                    }
                )
            if body.name is not None:
                cur["name"] = body.name.strip()
            if body.description is not None:
                cur["description"] = body.description.strip()
            if body.category is not None:
                cur["category"] = body.category.strip()
            if body.version is not None:
                cur["version"] = body.version.strip()
            if body.body_markdown is not None:
                cur["body_markdown"] = body.body_markdown.strip()
            if body.resources is not None:
                cur["resources_json"] = json.dumps(body.resources)
            if body.enabled is not None:
                cur["enabled"] = int(body.enabled)

            conn.execute(
                """UPDATE agent_skills
                   SET slug=?, name=?, description=?, category=?, version=?,
                       tags_json=?, allowed_tools_json=?, body_markdown=?, resources_json=?, enabled=?
                   WHERE id=? AND tenant_id=? AND workspace_id=?""",
                (
                    cur["slug"],
                    cur["name"],
                    cur["description"],
                    cur["category"],
                    cur["version"],
                    cur["tags_json"],
                    cur["allowed_tools_json"],
                    cur["body_markdown"],
                    cur["resources_json"],
                    cur["enabled"],
                    skill_id,
                    ctx.tenant_id,
                    ctx.workspace_id,
                ),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM agent_skills WHERE id = ? AND tenant_id = ?",
                (skill_id, ctx.tenant_id),
            ).fetchone()
        return row_to_skill_dict(updated)

    return await asyncio.to_thread(_apply)


@router.delete("/{skill_id}")
async def delete_skill(skill_id: str):
    ctx = get_tenant_context()

    def _del() -> int:
        with get_db() as conn:
            cur = conn.execute(
                "DELETE FROM agent_skills WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (skill_id, ctx.tenant_id, ctx.workspace_id),
            )
            conn.commit()
            return cur.rowcount

    deleted = await asyncio.to_thread(_del)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"ok": True}

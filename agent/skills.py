"""3-Tier Progressive Disclosure `SKILL.md` Engine (`agent/skills.py`).

Implements Anthropic's *Agent Skills (`SKILL.md`)* architecture for `nimAgents`:
  - **Tier 1 — Static Metadata Index (`~40 tokens/skill` in Block 1 System Prefix):**
    Exposes ONLY `slug`, `name`, and 1-sentence `description` across all turns and users,
    preserving cross-user GPU KV-cache prefixes while giving every agent instant awareness of
    dozens of specialized domain playbooks.
  - **Tier 2 — On-Demand Playbook Activation (`activate_skill(skill_slug)`):**
    Appends `skill_slug` to append-only thread state `activated_skill_ids` (`Command`),
    injecting the full structured `SKILL.md` playbook cleanly into **Block 2 (Session Prefix)**.
  - **Tier 3 — Zero-Overhead Linked Reference Assets (`read_skill_resource(skill_slug, resource_name)`):**
    Keeps large reference templates, SQL schemas, rubrics, and runbooks at **0 tokens** until read JIT.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Annotated, Any
import uuid

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.types import Command
import yaml

from agent.registry import count_text_tokens
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db

_SLUG_RE = re.compile(r"[^a-z0-9_-]+")


def sanitize_skill_slug(raw: str) -> str:
    cleaned = _SLUG_RE.sub("-", (raw or "").strip().lower()).strip("-")
    return cleaned[:64] or "custom-skill"


DEFAULT_STARTER_SKILLS: list[dict[str, Any]] = [
    {
        "slug": "nvidia-nim-architect",
        "name": "NVIDIA NIM & Dynamo Scaling Architect",
        "description": "Design high-throughput multi-user LLM/VLM deployments with NVIDIA NIM, Dynamo KV routing, and INT8/FP8 quantization.",
        "category": "infrastructure",
        "version": "1.2.0",
        "tags": ["nvidia", "nim", "dynamo", "kv-cache", "gpu"],
        "allowed_tools": ["calculator", "orchestrate_tools"],
        "body_markdown": (
            "## NVIDIA NIM & Dynamo Architecture Playbook\n"
            "1. **Prefix KV-Cache First:** Keep Block 1 system instructions + tool definitions 100% static across turns and users (`NIM_ENABLE_KV_CACHE_REUSE=1`). Never inject per-second timestamps into Block 1.\n"
            "2. **Aggregated vs Disaggregated Serving:** Below 8 GPUs or <2K prompt tokens, prefer aggregated NIM replicas with consistent prefix hashing; above 16 GPUs or 8K–64K agentic prefixes, split Prefill and Decode pools via NVIDIA Dynamo 1.0 + NIXL.\n"
            "3. **Sizing Formula:** Always compute KV cache memory per concurrent sequence with `calculator` (`2 * layers * kv_heads * head_dim * bytes_per_elem * seq_len`) before recommending GPU counts."
        ),
        "resources": {
            "kv_sizing_reference.md": (
                "# KV Cache Memory Quick Reference (FP8 = 1 byte/element)\n"
                "- Llama-3.3-70B / Nemotron-Super-49B at 32K context: ~2.6 GB FP8 KV per active sequence.\n"
                "- Reserve 15% HBM headroom for CUDA graphs + activation workspaces."
            )
        },
    },
    {
        "slug": "deep-research-synthesis",
        "name": "Multi-Source Research & Grounded Synthesis",
        "description": "Plan parallel evidence gathering, cross-verify claims across sources, and produce cited executive briefs.",
        "category": "research",
        "version": "1.0.0",
        "tags": ["research", "citations", "rag", "synthesis"],
        "allowed_tools": ["tool_search", "orchestrate_tools"],
        "body_markdown": (
            "## Grounded Multi-Source Research Protocol\n"
            "1. **Decompose First:** Break complex questions into 2–4 orthogonal sub-queries before invoking search or RAG tools.\n"
            "2. **Parallel Fan-Out:** Execute independent lookups concurrently via parallel tool calls or `orchestrate_tools` (`asyncio.gather`).\n"
            "3. **Conflict Audit:** Explicitly flag disagreements between primary sources with dates and quantitative metrics rather than averaging conflicting numbers.\n"
            "4. **Structured Output:** Lead with a 3-bullet executive answer, followed by a comparison table and grounded `[source]` citations."
        ),
        "resources": {
            "report_rubric.md": (
                "# Executive Research Quality Checklist\n"
                "- Every quantitative claim states unit + date + primary source.\n"
                "- Distinguish measured benchmarks from vendor projections."
            )
        },
    },
    {
        "slug": "python-data-investigation",
        "name": "Programmatic Data & API Investigation",
        "description": "Analyze large JSON/CSV datasets and multi-endpoint API responses inside `orchestrate_tools` without bloating context.",
        "category": "analytics",
        "version": "1.1.0",
        "tags": ["python", "analytics", "sandbox", "json"],
        "allowed_tools": ["orchestrate_tools", "json_explorer", "read_artifact"],
        "body_markdown": (
            "## Zero-Bloat Programmatic Investigation Workflow\n"
            "1. **Never Dump Raw Tables Into Chat Context:** Filter, join, group, and compute percentiles inside `orchestrate_tools` and print only compact summary JSON (`< 500` tokens).\n"
            "2. **Inspect Offloaded Artifacts Incrementally:** When a tool output is offloaded to `artifact://...`, inspect its structural schema first and use `read_artifact(artifact_id, grep=...)` instead of paging every line sequentially.\n"
            "3. **Sanity-Check Edge Cases:** Always report row counts, null rates, and p50/p95/max alongside averages."
        ),
        "resources": {
            "stats_template.py": (
                "def summary_stats(values):\n"
                "    s = sorted(values)\n"
                "    n = len(s)\n"
                "    return {'count': n, 'min': s[0], 'p50': s[n//2], 'p95': s[int(n*0.95)], 'max': s[-1]}\n"
            )
        },
    },
    {
        "slug": "incident-postmortem-triage",
        "name": "Production SRE Triage & Postmortem",
        "description": "Isolate root causes across latency/error spikes, verify blast radius, and draft blameless SEV postmortems.",
        "category": "operations",
        "version": "1.0.0",
        "tags": ["sre", "incident", "latency", "postmortem"],
        "allowed_tools": ["orchestrate_tools", "read_artifact"],
        "body_markdown": (
            "## SRE Root-Cause Triage & Postmortem Standard\n"
            "1. **Form Explicit Hypotheses:** Rank top 3 hypotheses (saturation, upstream dependency, recent config/deploy change) and falsify each with concrete telemetry.\n"
            "2. **Quantify Blast Radius:** Identify affected tenants, error budget burn, p95/p99 tail latency delta, and exact start/end timestamps in UTC.\n"
            "3. **Actionable Remediations:** Separate immediate mitigation (<1h) from structural prevention (automated regression test + circuit breaker)."
        ),
        "resources": {
            "sev_template.md": (
                "## Postmortem Template\n"
                "- **Summary & Impact**\n- **Timeline (UTC)**\n- **Root Cause & Trigger**\n- **What Went Well / Poorly**\n- **Action Items (Owner + Test)**"
            )
        },
    },
]


def ensure_tenant_skills_seeded(conn: sqlite3.Connection, ctx: TenantContext) -> None:
    """Seed the 4 starter `SKILL.md` playbooks once per `(tenant_id, workspace_id)`."""
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM agent_skills WHERE tenant_id = ? AND workspace_id = ?",
        (ctx.tenant_id, ctx.workspace_id),
    ).fetchone()
    if row and int(row["cnt"]) > 0:
        return

    for item in DEFAULT_STARTER_SKILLS:
        skill_id = f"skill_{sanitize_tenant_id_short(ctx)}_{item['slug']}"
        conn.execute(
            """INSERT OR IGNORE INTO agent_skills
               (id, tenant_id, workspace_id, slug, name, description, category, version,
                tags_json, allowed_tools_json, body_markdown, resources_json, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
            (
                skill_id,
                ctx.tenant_id,
                ctx.workspace_id,
                item["slug"],
                item["name"],
                item["description"],
                item["category"],
                item["version"],
                json.dumps(item["tags"]),
                json.dumps(item["allowed_tools"]),
                item["body_markdown"],
                json.dumps(item["resources"]),
            ),
        )
    conn.commit()


def sanitize_tenant_id_short(ctx: TenantContext) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", f"{ctx.tenant_id}_{ctx.workspace_id}")[:24]


def parse_skill_markdown(raw_skill_md: str) -> dict[str, Any]:
    """Parse standard `SKILL.md` (`---\\nYAML frontmatter\\n---\\nMarkdown body`) into structured fields."""
    text = (raw_skill_md or "").strip()
    if not text:
        raise ValueError("SKILL.md content cannot be empty")

    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            parsed = yaml.safe_load(parts[1]) or {}
            if not isinstance(parsed, dict):
                raise ValueError("SKILL.md YAML frontmatter must be a mapping")
            meta = parsed
            body = parts[2].strip()

    raw_name = str(meta.get("name") or "").strip()
    if not raw_name:
        heading_match = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
        raw_name = heading_match.group(1).strip() if heading_match else "Custom Agent Skill"

    slug = sanitize_skill_slug(str(meta.get("slug") or raw_name))
    description = str(meta.get("description") or "").strip()
    if not description:
        first_para = next((ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")), "")
        description = (first_para[:220] or f"Specialized domain playbook for {raw_name}")

    raw_tags = meta.get("tags") or []
    tags = [str(t).strip() for t in raw_tags] if isinstance(raw_tags, list) else [s.strip() for s in str(raw_tags).split(",") if s.strip()]
    raw_tools = meta.get("allowed_tools") or meta.get("allowed-tools") or []
    allowed_tools = (
        [str(t).strip() for t in raw_tools]
        if isinstance(raw_tools, list)
        else [s.strip() for s in str(raw_tools).split(",") if s.strip()]
    )
    resources = meta.get("resources") if isinstance(meta.get("resources"), dict) else {}

    return {
        "slug": slug,
        "name": raw_name[:100],
        "description": description[:400],
        "category": str(meta.get("category") or "engineering")[:48],
        "version": str(meta.get("version") or "1.0.0")[:24],
        "tags": tags,
        "allowed_tools": allowed_tools,
        "body_markdown": body,
        "resources": {str(k): str(v) for k, v in resources.items()},
    }


def render_skill_markdown(skill_row: dict[str, Any]) -> str:
    """Export any stored skill back into portable `SKILL.md` format."""
    tags = skill_row.get("tags")
    if tags is None:
        tags = json.loads(skill_row.get("tags_json") or "[]")
    allowed_tools = skill_row.get("allowed_tools")
    if allowed_tools is None:
        allowed_tools = json.loads(skill_row.get("allowed_tools_json") or "[]")
    frontmatter = {
        "name": skill_row["name"],
        "slug": skill_row["slug"],
        "description": skill_row["description"],
        "category": skill_row.get("category", "engineering"),
        "version": skill_row.get("version", "1.0.0"),
        "tags": tags,
        "allowed_tools": allowed_tools,
    }
    yaml_str = yaml.safe_dump(frontmatter, sort_keys=False, allow_unicode=True).strip()
    return f"---\n{yaml_str}\n---\n\n{skill_row.get('body_markdown', '').strip()}\n"


def row_to_skill_dict(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    d = dict(row)
    tags = json.loads(d.get("tags_json") or "[]")
    allowed_tools = json.loads(d.get("allowed_tools_json") or "[]")
    resources: dict[str, str] = json.loads(d.get("resources_json") or "{}")
    tier1_line = f"- `{d['slug']}` ({d['name']}): {d['description']}"
    tier1_tokens = count_text_tokens(tier1_line)
    tier2_tokens = count_text_tokens(d.get("body_markdown") or "")
    tier3_tokens = sum(count_text_tokens(v) for v in resources.values())
    total_eager = tier1_tokens + tier2_tokens + tier3_tokens
    saved_pct = round(((total_eager - tier1_tokens) / total_eager) * 100.0, 1) if total_eager > 0 else 0.0

    return {
        "id": d["id"],
        "slug": d["slug"],
        "name": d["name"],
        "description": d["description"],
        "category": d.get("category") or "engineering",
        "version": d.get("version") or "1.0.0",
        "tags": tags,
        "allowed_tools": allowed_tools,
        "body_markdown": d.get("body_markdown") or "",
        "resources": resources,
        "enabled": bool(d.get("enabled", 1)),
        "tier1_index_tokens": tier1_tokens,
        "tier2_playbook_tokens": tier2_tokens,
        "tier3_resources_tokens": tier3_tokens,
        "progressive_savings_pct": saved_pct,
        "raw_skill_md": render_skill_markdown({**d, "tags": tags, "allowed_tools": allowed_tools}),
        "created_at": d.get("created_at", ""),
    }


def load_tenant_skills(
    ctx: TenantContext | None = None,
    *,
    only_enabled: bool = True,
    allowed_skill_ids_or_slugs: set[str] | None = None,
) -> list[dict[str, Any]]:
    active = ctx or get_tenant_context()
    with get_db() as conn:
        ensure_tenant_skills_seeded(conn, active)
        sql = "SELECT * FROM agent_skills WHERE tenant_id = ? AND workspace_id = ?"
        params: list[Any] = [active.tenant_id, active.workspace_id]
        if only_enabled:
            sql += " AND enabled = 1"
        sql += " ORDER BY slug ASC"
        rows = conn.execute(sql, params).fetchall()

    items = [row_to_skill_dict(r) for r in rows]
    if allowed_skill_ids_or_slugs is not None:
        items = [
            s
            for s in items
            if s["id"] in allowed_skill_ids_or_slugs or s["slug"] in allowed_skill_ids_or_slugs
        ]
    return items


def build_tier1_skill_directory_prompt(skills: list[dict[str, Any]]) -> str:
    """Build deterministic Tier-1 metadata directory (`~40 tok/skill`) for Block 1 Static Prefix."""
    if not skills:
        return ""
    lines = [
        "\n\n## Available Agent Skills (`SKILL.md` Progressive Playbooks)",
        "Call `activate_skill(skill_slug)` before tackling tasks matching a specialized playbook below:",
    ]
    for s in sorted(skills, key=lambda x: x["slug"]):
        lines.append(f"- `{s['slug']}` ({s['name']}): {s['description']}")
    return "\n".join(lines)


def build_tier2_active_skills_block(
    skills_by_slug: dict[str, dict[str, Any]],
    activated_slugs: list[str],
) -> str:
    """Render activated Tier-2 `SKILL.md` instructions in deterministic activation order for Block 2."""
    sections: list[str] = []
    seen: set[str] = set()
    for slug in activated_slugs:
        if slug in seen or slug not in skills_by_slug:
            continue
        seen.add(slug)
        sk = skills_by_slug[slug]
        res_keys = list((sk.get("resources") or {}).keys())
        res_hint = (
            f"\nLinked Tier-3 Resources (load via `read_skill_resource('{slug}', name)`): {', '.join(res_keys)}"
            if res_keys
            else ""
        )
        sections.append(
            f"<skill slug=\"{slug}\" version=\"{sk.get('version', '1.0.0')}\">\n"
            f"{sk['body_markdown'].strip()}{res_hint}\n"
            f"</skill>"
        )
    if not sections:
        return ""
    return "## Active Domain Skills\n" + "\n\n".join(sections)


def build_skill_runtime_tools(
    skills: list[dict[str, Any]],
) -> list[BaseTool]:
    """Create `activate_skill` (Tier 2 state command) and `read_skill_resource` (Tier 3 JIT reader)."""
    by_slug = {s["slug"]: s for s in skills}
    by_id = {s["id"]: s for s in skills}

    @tool
    def activate_skill(
        skill_slug: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Activate a specialized `SKILL.md` domain playbook by `skill_slug` into session context."""
        target = by_slug.get(skill_slug.strip()) or by_id.get(skill_slug.strip())
        if target is None:
            available = ", ".join(sorted(by_slug.keys())) or "none"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Skill '{skill_slug}' not found. Available slugs: {available}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )
        slug = target["slug"]
        res_list = list((target.get("resources") or {}).keys())
        summary = (
            f"Activated skill `{slug}` ({target['name']}). "
            f"Full playbook loaded into session prefix."
            + (f" Available Tier-3 resources: {', '.join(res_list)}." if res_list else "")
        )
        return Command(
            update={
                "activated_skill_ids": [slug],
                "messages": [ToolMessage(content=summary, tool_call_id=tool_call_id)],
            }
        )

    @tool
    def read_skill_resource(skill_slug: str, resource_name: str) -> str:
        """Read a linked Tier-3 reference template, schema, or rubric inside an Agent Skill on demand."""
        target = by_slug.get(skill_slug.strip()) or by_id.get(skill_slug.strip())
        if target is None:
            return f"Skill '{skill_slug}' not found."
        resources = target.get("resources") or {}
        if resource_name not in resources:
            return (
                f"Resource '{resource_name}' not found in `{target['slug']}`. "
                f"Available: {', '.join(sorted(resources.keys())) or 'none'}"
            )
        return str(resources[resource_name])

    return [activate_skill, read_skill_resource]

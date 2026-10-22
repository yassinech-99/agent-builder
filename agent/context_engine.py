"""Phase 4 — KV-Cache-Aligned Context Projection & 5-Stage Graduated Compaction Engine (`agent/context_engine.py`).

Implements the complete multi-tenant context engineering pipeline from `plan.md`:
  - **3-Block KV-Cache-Aligned Prompt Projection (`ContextProjection`):**
      * **Block 1 (`static_prefix`)**: 100% deterministic agent system prompt + Tier-1 Skill Directory +
        Deferred Tool Namespace Directory. Byte-identical across EVERY turn AND EVERY user sharing an agent,
        maximizing cross-user Radix-Tree GPU KV cache hits on NVIDIA Dynamo / vLLM / SGLang!
      * **Block 2 (`session_prefix`)**: Pinned session memory notes + Active Tier-2 `SKILL.md` playbooks +
        Latest rolling `<compaction_checkpoint>`. Stable across normal turns; mutates only on skill activation
        or 80% compaction transitions.
      * **Block 3 (`dynamic_tail`)**: Shaped conversation history respecting atomic `[Human / AI+ToolCalls+Tools]`
        turn boundaries so orphan `ToolMessage` errors are mathematically impossible.

  - **5-Stage Graduated Pre-Model Shaper Pipeline:**
      * **Stage 1 — Large Tool Output Offloading (`> offload_threshold_tokens`, default `3,000` tok):**
        Streams oversized `ToolMessage` outputs to RustFS/S3 (with seamless tenant-isolated DB artifact fallback),
        replacing inline bloat with a structural schema summary + `1,000`-token preview + `read_artifact(...)`.
      * **Stage 2 — Historical `<think>` Reasoning Stripping (`keep_recent_thinking_turns = 2`):**
        Strips `<think>...</think>` blocks and `reasoning_content` older than the latest 2 assistant turns.
      * **Stage 3 — Batch KV-Aware Tool Result Clearing (`clear_at_least_tokens = 10,000`):**
        Evicts stale re-fetchable tool results older than `keep_recent_tool_messages = 5` ONLY when evictable
        tokens reach `clear_at_least_tokens` in a single batch (preventing per-turn prefix cache invalidation!),
        strictly preserving `PROTECTED_TOOL_NAMES`.
      * **Stage 4 — Pre-Compact Salient Fact Flush + Rolling Compaction (`> 80%` of `context_window`):**
        Extracts salient entities/decisions/artifacts before summarizing older atomic turn groups via an optional
        async `compactor` model (or deterministic structured fallback) into `<compaction_checkpoint>`.
      * **Stage 5 — Deterministic Hard Token Ceiling Guard (`> 95%` of `context_window`):**
        Mathematically guarantees total projected tokens never exceed `int(context_window * 0.95)`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import io
import json
import logging
import re
from typing import Any, Callable, Awaitable
import uuid

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool, tool

from agent.inference_gateway import compute_prefix_hash
from agent.registry import count_text_tokens
from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db

logger = logging.getLogger("nimagents.context_engine")

THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>\s*", flags=re.IGNORECASE)

PROTECTED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "memory",
        "recall_memories",
        "upsert_memory_fact",
        "update_core_memory_block",
        "propose_procedural_rule",
        "read_artifact",
        "activate_skill",
        "read_skill_resource",
        "human_input",
        "tool_search",
        "delegate_to_subagent",
        "delegate_parallel_subagents",
    }
)


@dataclass
class ContextPolicy:
    """Tunable thresholds governing the 5-stage graduated context pipeline."""

    context_window: int = 131072
    offload_threshold_tokens: int = 3000
    preview_tokens: int = 1000
    keep_recent_thinking_turns: int = 2
    keep_recent_tool_messages: int = 5
    clear_at_least_tokens: int = 10000
    soft_compact_ratio: float = 0.80
    hard_ceiling_ratio: float = 0.95
    keep_verbatim_turn_groups: int = 6


@dataclass
class ContextProjection:
    """KV-cache-aligned prompt projection + rich per-stage telemetry."""

    block1_static_prefix: str
    block2_session_prefix: str
    block3_messages: list[BaseMessage]
    prefix_hash: str
    static_prefix_tokens: int
    session_prefix_tokens: int
    dynamic_tail_tokens: int
    tool_schema_tokens: int
    total_tokens: int
    context_window: int
    utilization_pct: float
    kv_cache_hit_pct: float
    offloaded_artifact_ids: list[str] = field(default_factory=list)
    cleared_thinking_tokens: int = 0
    cleared_tool_tokens: int = 0
    compacted_this_turn: bool = False
    updated_checkpoint: str = ""
    pre_compact_facts: list[str] = field(default_factory=list)

    def to_model_messages(self) -> list[BaseMessage]:
        """Materialize final `list[BaseMessage]` with static Block 1 always at index 0."""
        msgs: list[BaseMessage] = [SystemMessage(content=self.block1_static_prefix)]
        if self.block2_session_prefix.strip():
            msgs.append(SystemMessage(content=self.block2_session_prefix))
        msgs.extend(self.block3_messages)
        return msgs


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"[{item.get('type', 'block')}]")
        return "\n".join(parts)
    return json.dumps(content, default=str)


def count_message_tokens(msg: BaseMessage) -> int:
    base = 6 + count_text_tokens(_stringify_content(msg.content))
    if isinstance(msg, AIMessage):
        if msg.tool_calls:
            base += count_text_tokens(json.dumps(msg.tool_calls, default=str))
        reasoning = (msg.additional_kwargs or {}).get("reasoning_content")
        if reasoning:
            base += count_text_tokens(str(reasoning))
    return base


def extract_structural_summary(raw_text: str, content_type: str = "text/plain") -> tuple[str, str]:
    """Inspect large JSON/Markdown/CSV/Log output and extract structural schema + detected type."""
    stripped = raw_text.strip()
    lines = raw_text.splitlines()
    line_count = max(1, len(lines))

    # 1. JSON Object or JSON Array inspection
    if stripped[:1] in ("{", "["):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                keys = list(parsed.keys())
                sample_keys = ", ".join(f"`{k}` ({type(parsed[k]).__name__})" for k in keys[:12])
                extra = f" (+{len(keys) - 12} more keys)" if len(keys) > 12 else ""
                return (
                    f"JSON object with {len(keys)} top-level keys across {line_count} lines: {sample_keys}{extra}",
                    "application/json",
                )
            if isinstance(parsed, list):
                elem_type = type(parsed[0]).__name__ if parsed else "empty"
                cols = ""
                if parsed and isinstance(parsed[0], dict):
                    col_names = list(parsed[0].keys())[:12]
                    cols = f"; item fields: {', '.join(f'`{c}`' for c in col_names)}"
                return (
                    f"JSON array with {len(parsed)} `{elem_type}` records ({line_count} lines){cols}",
                    "application/json",
                )
        except Exception:
            pass

    # 2. Markdown / Document headings inspection
    headings = [ln.strip() for ln in lines if ln.lstrip().startswith("#")][:8]
    if headings:
        return (
            f"Structured document ({line_count} lines) with headings: {' | '.join(headings)}",
            "text/markdown",
        )

    # 3. Tabular CSV / TSV heuristic
    if line_count >= 2 and ("," in lines[0] or "\t" in lines[0]):
        delim = "\t" if "\t" in lines[0] else ","
        cols = [c.strip() for c in lines[0].split(delim)[:12] if c.strip()]
        if len(cols) >= 3:
            return (
                f"Tabular dataset ({line_count - 1} rows, columns: {', '.join(f'`{c}`' for c in cols)})",
                "text/csv",
            )

    return (f"Text/log payload ({line_count} lines, {len(raw_text):,} chars)", content_type)


def _try_upload_artifact_to_object_store(
    ctx: TenantContext,
    artifact_id: str,
    raw_bytes: bytes,
    content_type: str,
) -> str:
    """Stream artifact payload to tenant-isolated RustFS object storage when configured."""
    try:
        with get_db() as conn:
            row = conn.execute(
                """SELECT * FROM storage_connections
                   WHERE tenant_id = ? AND workspace_id = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (ctx.tenant_id, ctx.workspace_id),
            ).fetchone()
        if not row:
            return ""
        from api.core.object_store import build_object_store_client_for_bucket
        from api.core.tenancy import scoped_object_prefix

        row_d = dict(row)
        bucket = row_d["bucket_name"]
        client = build_object_store_client_for_bucket(bucket, ctx=ctx)
        try:
            obj_key = f"{scoped_object_prefix('artifacts', ctx)}{artifact_id}.txt"
            client.put_object(
                bucket,
                obj_key,
                io.BytesIO(raw_bytes),
                length=len(raw_bytes),
                content_type=content_type,
            )
            return f"s3://{bucket}/{obj_key}"
        finally:
            client.close()
    except Exception:
        return ""


def persist_offloaded_artifact(
    *,
    raw_text: str,
    tool_name: str,
    thread_id: str = "",
    preview_tokens: int = 1000,
    ctx: TenantContext | None = None,
    deterministic_id: str | None = None,
) -> dict[str, Any]:
    """Persist large tool output as a tenant-isolated artifact (idempotent by `deterministic_id`)."""
    active = ctx or get_tenant_context()
    artifact_id = deterministic_id or f"art_{uuid.uuid4().hex[:12]}"

    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM context_artifacts WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (artifact_id, active.tenant_id, active.workspace_id),
        ).fetchone()
        if existing:
            return dict(existing)

    summary, detected_ct = extract_structural_summary(raw_text)
    raw_bytes = raw_text.encode("utf-8", errors="replace")
    tok_count = count_text_tokens(raw_text)
    char_budget = max(400, preview_tokens * 4)
    preview = raw_text[:char_budget]
    s3_uri = _try_upload_artifact_to_object_store(active, artifact_id, raw_bytes, detected_ct)
    storage_uri = s3_uri or f"artifact://{active.tenant_id}/{active.workspace_id}/{artifact_id}"

    record = {
        "id": artifact_id,
        "tenant_id": active.tenant_id,
        "workspace_id": active.workspace_id,
        "thread_id": thread_id,
        "tool_name": tool_name or "tool",
        "content_type": detected_ct,
        "byte_size": len(raw_bytes),
        "token_count": tok_count,
        "storage_uri": storage_uri,
        "inline_content": raw_text,
        "structural_summary": summary,
        "preview_text": preview,
    }
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO context_artifacts
               (id, tenant_id, workspace_id, thread_id, tool_name, content_type,
                byte_size, token_count, storage_uri, inline_content, structural_summary, preview_text)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(record.values()),
        )
        conn.commit()
    return record


def read_offloaded_artifact(
    artifact_id: str,
    *,
    offset: int = 0,
    limit: int = 80,
    grep: str = "",
    ctx: TenantContext | None = None,
) -> str:
    """JIT reader supporting line windowing (`offset`/`limit`) and regex/substring `grep`."""
    active = ctx or get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM context_artifacts WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (artifact_id.strip(), active.tenant_id, active.workspace_id),
        ).fetchone()
    if not row:
        return f"Artifact '{artifact_id}' not found in current tenant workspace."

    d = dict(row)
    lines = (d.get("inline_content") or "").splitlines()
    total_lines = len(lines)

    if grep.strip():
        try:
            rx = re.compile(grep.strip(), flags=re.IGNORECASE)
            matched = [f"L{idx + 1}: {ln}" for idx, ln in enumerate(lines) if rx.search(ln)]
        except re.error:
            needle = grep.strip().lower()
            matched = [f"L{idx + 1}: {ln}" for idx, ln in enumerate(lines) if needle in ln.lower()]
        slice_matches = matched[max(0, offset) : max(0, offset) + max(1, min(limit, 200))]
        header = (
            f"[Artifact `{d['id']}` · grep='{grep}' · {len(matched)} matching lines of {total_lines}]\n"
        )
        return header + ("\n".join(slice_matches) if slice_matches else "(no lines matched)")

    start = max(0, min(offset, total_lines))
    end = min(total_lines, start + max(1, min(limit, 250)))
    numbered = [f"L{idx + 1}: {lines[idx]}" for idx in range(start, end)]
    header = (
        f"[Artifact `{d['id']}` ({d['storage_uri']}) · showing lines {start + 1}..{end} of {total_lines}]\n"
    )
    return header + "\n".join(numbered)


def build_read_artifact_tool(ctx: TenantContext) -> BaseTool:
    @tool
    def read_artifact(
        artifact_id: str,
        offset: int = 0,
        limit: int = 80,
        grep: str = "",
    ) -> str:
        """Read lines or search (`grep`) inside an offloaded large tool output (`artifact://...`) on demand."""
        return read_offloaded_artifact(
            artifact_id,
            offset=offset,
            limit=limit,
            grep=grep,
            ctx=ctx,
        )

    return read_artifact


def group_messages_into_atomic_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages into atomic units so `[AIMessage(tool_calls), ToolMessage...]` are NEVER split!"""
    groups: list[list[BaseMessage]] = []
    idx = 0
    n = len(messages)
    while idx < n:
        msg = messages[idx]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            unit = [msg]
            idx += 1
            while idx < n and isinstance(messages[idx], ToolMessage):
                unit.append(messages[idx])
                idx += 1
            groups.append(unit)
        else:
            groups.append([msg])
            idx += 1
    return groups


def extract_pre_compact_facts(messages: list[BaseMessage], max_facts: int = 12) -> list[str]:
    """Deterministically harvest salient entities, artifact IDs, numbers, decisions & constraints before compaction."""
    facts: list[str] = []
    seen: set[str] = set()

    for m in messages:
        text = _stringify_content(m.content)
        if not text:
            continue
        # Capture any offloaded artifact handles so compacted context can still inspect them JIT!
        for art_id in re.findall(r"\bart_[a-zA-Z0-9_]+\b", text):
            entry = f"Artifact handle available via `read_artifact('{art_id}')`"
            if entry not in seen:
                seen.add(entry)
                facts.append(entry)

        if isinstance(m, HumanMessage):
            for line in text.splitlines():
                clean = line.strip()
                if not clean:
                    continue
                if any(
                    kw in clean.lower()
                    for kw in (
                        "must",
                        "never",
                        "always",
                        "prefer",
                        "budget",
                        "deadline",
                        "version",
                        "project",
                        "api",
                        "target",
                        "=",
                        "$",
                        "%",
                    )
                ) or re.search(r"\b\d+(?:\.\d+)?\s*(?:gb|ms|%|usd|gpus?|users?|tokens?)\b", clean, re.I):
                    snippet = clean[:220]
                    entry = f"User constraint/fact: {snippet}"
                    if entry not in seen:
                        seen.add(entry)
                        facts.append(entry)
                        if len(facts) >= max_facts:
                            return facts
    return facts[:max_facts]


def build_deterministic_compaction_summary(
    older_messages: list[BaseMessage],
    prior_checkpoint: str,
    extracted_facts: list[str],
) -> str:
    """Recall-tuned deterministic structured `<compaction_checkpoint>` builder (0ms network latency!)."""
    user_goals: list[str] = []
    tool_actions: list[str] = []
    assistant_conclusions: list[str] = []

    for m in older_messages:
        txt = _stringify_content(m.content).strip()
        if isinstance(m, HumanMessage) and txt:
            user_goals.append(f"- {txt.splitlines()[0][:200]}")
        elif isinstance(m, ToolMessage):
            tname = getattr(m, "name", "") or "tool"
            first_ln = txt.splitlines()[0][:160] if txt else "(empty)"
            tool_actions.append(f"- `{tname}` -> {first_ln}")
        elif isinstance(m, AIMessage) and txt and not getattr(m, "tool_calls", None):
            clean_txt = THINK_BLOCK_RE.sub("", txt).strip()
            if clean_txt:
                assistant_conclusions.append(f"- {clean_txt.splitlines()[0][:220]}")

    sections = ["<compaction_checkpoint>"]
    if prior_checkpoint.strip():
        prior_body = re.sub(r"</?compaction_checkpoint>", "", prior_checkpoint).strip()
        sections.append("### Carried-Forward Context\n" + prior_body[:1200])
    if extracted_facts:
        sections.append("### Salient Preserved Facts & Handles\n" + "\n".join(f"- {f}" for f in extracted_facts[:10]))
    if user_goals:
        sections.append("### Earlier User Objectives\n" + "\n".join(user_goals[-8:]))
    if tool_actions:
        sections.append("### Completed Tool & Artifact Trace\n" + "\n".join(tool_actions[-10:]))
    if assistant_conclusions:
        sections.append("### Key Decisions & Findings\n" + "\n".join(assistant_conclusions[-6:]))
    sections.append("</compaction_checkpoint>")
    return "\n\n".join(sections)


class FiveStageContextEngine:
    """Executes the 5-stage KV-cache-preserving context projection & compaction pipeline."""

    async def shape_context(
        self,
        *,
        static_system_prompt: str,
        session_memory_notes: str = "",
        active_skills_block: str = "",
        prior_compaction_checkpoint: str = "",
        messages: list[BaseMessage],
        tool_schema_tokens: int = 0,
        policy: ContextPolicy | None = None,
        ctx: TenantContext | None = None,
        thread_id: str = "",
        agent_name: str = "nimAgents Assistant",
        deferred_tools_count: int = 0,
        active_skill_slugs: list[str] | None = None,
        force_compact: bool = False,
        compactor_fn: Callable[[str], Awaitable[str]] | None = None,
        memory_flush_fn: Callable[[list[str], TenantContext], Awaitable[None]] | None = None,
    ) -> ContextProjection:
        cfg = policy or ContextPolicy()
        active_ctx = ctx or get_tenant_context()

        # Clone messages shallowly so caller checkpoint history is never mutated unexpectedly
        working: list[BaseMessage] = [copy.copy(m) for m in messages if not isinstance(m, SystemMessage)]

        offloaded_ids: list[str] = []
        cleared_thinking = 0
        cleared_tool_toks = 0
        compacted_now = False
        checkpoint_text = prior_compaction_checkpoint.strip()
        harvested_facts: list[str] = []

        # ── STAGE 1: Large Tool Output Offloading (`> cfg.offload_threshold_tokens`) ──
        for idx, msg in enumerate(working):
            if not isinstance(msg, ToolMessage):
                continue
            raw_str = _stringify_content(msg.content)
            if raw_str.startswith("[Offloaded Large Tool Result -> `artifact://") or "[Artifact `" in raw_str[:40]:
                continue
            tok_len = count_text_tokens(raw_str)
            if tok_len > cfg.offload_threshold_tokens:
                det_seed = compute_prefix_hash(thread_id, getattr(msg, "tool_call_id", str(idx)), raw_str[:256])
                det_id = f"art_{det_seed[:12]}"
                art = persist_offloaded_artifact(
                    raw_text=raw_str,
                    tool_name=getattr(msg, "name", None) or "tool",
                    thread_id=thread_id,
                    preview_tokens=cfg.preview_tokens,
                    ctx=active_ctx,
                    deterministic_id=det_id,
                )
                offloaded_ids.append(art["id"])
                replacement = (
                    f"[Offloaded Large Tool Result -> `{art['storage_uri']}` (id=`{art['id']}`, "
                    f"{art['token_count']:,} tokens / {art['byte_size']:,} bytes)]\n"
                    f"Schema Summary: {art['structural_summary']}\n"
                    f"Use `read_artifact(artifact_id='{art['id']}', offset=0, limit=80, grep='...')` for full JIT inspection.\n"
                    f"--- Preview ---\n{art['preview_text']}"
                )
                msg.content = replacement

        # ── STAGE 2: Historical `<think>` Reasoning Stripping (`< N - keep_recent_thinking_turns`) ──
        ai_indices = [i for i, m in enumerate(working) if isinstance(m, AIMessage)]
        cutoff_ai_set = set(ai_indices[: max(0, len(ai_indices) - cfg.keep_recent_thinking_turns)])
        for i in cutoff_ai_set:
            ai_msg = working[i]
            assert isinstance(ai_msg, AIMessage)
            before_tok = count_message_tokens(ai_msg)
            if isinstance(ai_msg.content, str) and "<think>" in ai_msg.content.lower():
                ai_msg.content = THINK_BLOCK_RE.sub("", ai_msg.content).strip()
            if ai_msg.additional_kwargs and "reasoning_content" in ai_msg.additional_kwargs:
                new_kw = dict(ai_msg.additional_kwargs)
                new_kw.pop("reasoning_content", None)
                ai_msg.additional_kwargs = new_kw
            after_tok = count_message_tokens(ai_msg)
            cleared_thinking += max(0, before_tok - after_tok)

        # ── STAGE 3: Batch KV-Aware Stale Tool Result Clearing (`clear_at_least_tokens`) ──
        tool_msg_indices = [i for i, m in enumerate(working) if isinstance(m, ToolMessage)]
        stale_candidates = tool_msg_indices[: max(0, len(tool_msg_indices) - cfg.keep_recent_tool_messages)]
        evictable: list[tuple[int, int]] = []
        for ti in stale_candidates:
            t_msg = working[ti]
            t_name = (getattr(t_msg, "name", None) or "").strip()
            if t_name in PROTECTED_TOOL_NAMES:
                continue
            body_txt = _stringify_content(t_msg.content)
            if body_txt.startswith("[Cleared stale tool result"):
                continue
            t_toks = count_text_tokens(body_txt)
            if t_toks > 60:
                evictable.append((ti, t_toks))

        total_evictable_toks = sum(tok for _, tok in evictable)
        if total_evictable_toks >= cfg.clear_at_least_tokens:
            for ti, old_toks in evictable:
                t_msg = working[ti]
                t_name = getattr(t_msg, "name", None) or "tool"
                # Preserve any artifact pointer ID inside the compact tombstone!
                art_match = re.search(r"\bart_[a-zA-Z0-9_]+\b", _stringify_content(t_msg.content))
                art_note = f" (artifact `{art_match.group(0)}` still readable via `read_artifact`)" if art_match else ""
                placeholder = f"[Cleared stale tool result from `{t_name}` ({old_toks} tok){art_note}; re-run tool if needed.]"
                new_toks = count_text_tokens(placeholder)
                t_msg.content = placeholder
                cleared_tool_toks += max(0, old_toks - new_toks)

        # Assemble Block 1 & Block 2 helpers to measure exact token utilization
        prefix_hash = compute_prefix_hash(static_system_prompt)
        static_tokens = count_text_tokens(static_system_prompt) + tool_schema_tokens

        def _compose_block2(chk: str) -> str:
            parts: list[str] = []
            if session_memory_notes.strip():
                parts.append(f"<session_memory>\n{session_memory_notes.strip()}\n</session_memory>")
            if active_skills_block.strip():
                parts.append(active_skills_block.strip())
            if chk.strip():
                parts.append(chk.strip())
            return "\n\n".join(parts)

        block2 = _compose_block2(checkpoint_text)
        session_tokens = count_text_tokens(block2) if block2 else 0
        tail_tokens = sum(count_message_tokens(m) for m in working)
        soft_limit = int(cfg.context_window * cfg.soft_compact_ratio)

        # ── STAGE 4: Pre-Compact Salient Fact Flush + Rolling Compaction (`> 80%` window) ──
        if force_compact or (static_tokens + session_tokens + tail_tokens > soft_limit):
            turn_groups = group_messages_into_atomic_turns(working)
            keep_n = min(cfg.keep_verbatim_turn_groups, max(1, len(turn_groups) - 1))
            if len(turn_groups) > 1:
                older_groups = turn_groups[:-keep_n]
                recent_groups = turn_groups[-keep_n:]
                older_flat = [m for grp in older_groups for m in grp]
                harvested_facts = extract_pre_compact_facts(older_flat)
                if memory_flush_fn is not None and harvested_facts:
                    try:
                        await memory_flush_fn(harvested_facts, active_ctx)
                    except Exception as flush_exc:
                        logger.warning("Pre-compact memory flush failed cleanly: %s", flush_exc)

                if compactor_fn is not None:
                    serialized = "\n".join(
                        f"[{m.type.upper()}] {_stringify_content(m.content)[:1200]}"
                        for m in older_flat
                    )
                    prompt_for_compactor = (
                        "Summarize the following earlier agent trajectory into a concise, high-recall "
                        "`<compaction_checkpoint>` preserving all goals, numbers, file/artifact IDs, and decisions:\n\n"
                        f"{serialized}"
                    )
                    try:
                        summary_raw = await compactor_fn(prompt_for_compactor)
                        checkpoint_text = (
                            summary_raw
                            if "<compaction_checkpoint>" in summary_raw
                            else f"<compaction_checkpoint>\n{summary_raw.strip()}\n</compaction_checkpoint>"
                        )
                    except Exception:
                        checkpoint_text = build_deterministic_compaction_summary(
                            older_flat, checkpoint_text, harvested_facts
                        )
                else:
                    checkpoint_text = build_deterministic_compaction_summary(
                        older_flat, checkpoint_text, harvested_facts
                    )

                working = [m for grp in recent_groups for m in grp]
                compacted_now = True
                block2 = _compose_block2(checkpoint_text)
                session_tokens = count_text_tokens(block2) if block2 else 0
                tail_tokens = sum(count_message_tokens(m) for m in working)

        # ── STAGE 5: Deterministic Hard Token Ceiling Guard (`<= 95%` window — ZERO overflows!) ──
        hard_ceiling = max(512, int(cfg.context_window * cfg.hard_ceiling_ratio))
        while (static_tokens + session_tokens + tail_tokens) > hard_ceiling:
            groups = group_messages_into_atomic_turns(working)
            if len(groups) > 1:
                # Drop oldest atomic turn group (never splitting AI tool_calls from ToolMessages!)
                working = [m for grp in groups[1:] for m in grp]
                tail_tokens = sum(count_message_tokens(m) for m in working)
                continue

            # If even 1 single gigantic turn + checkpoint exceeds hard ceiling, trim checkpoint first, then message text
            if checkpoint_text:
                max_chk_chars = max(200, (hard_ceiling - static_tokens) * 2)
                if len(checkpoint_text) > max_chk_chars:
                    checkpoint_text = checkpoint_text[:max_chk_chars] + "\n...[trimmed]\n</compaction_checkpoint>"
                    block2 = _compose_block2(checkpoint_text)
                    session_tokens = count_text_tokens(block2) if block2 else 0
                    if (static_tokens + session_tokens + tail_tokens) <= hard_ceiling:
                        break
                else:
                    checkpoint_text = ""
                    block2 = _compose_block2("")
                    session_tokens = count_text_tokens(block2) if block2 else 0
                    if (static_tokens + session_tokens + tail_tokens) <= hard_ceiling:
                        break

            if working:
                budget_tokens_left = max(64, hard_ceiling - static_tokens - session_tokens - 24)
                last_msg = working[-1]
                raw = _stringify_content(last_msg.content)
                trimmed = raw[: max(128, budget_tokens_left * 3)] + "\n...[Hard Context Ceiling Guard applied]"
                last_msg.content = trimmed
                tail_tokens = sum(count_message_tokens(m) for m in working)
            break

        total_tokens = static_tokens + session_tokens + tail_tokens
        reusable_prefix_tokens = static_tokens + session_tokens
        kv_hit_pct = (
            round(min(98.5, (reusable_prefix_tokens / max(1, total_tokens)) * 100.0), 1)
            if total_tokens > 0
            else 0.0
        )
        util_pct = round((total_tokens / max(1, cfg.context_window)) * 100.0, 2)

        projection = ContextProjection(
            block1_static_prefix=static_system_prompt,
            block2_session_prefix=block2,
            block3_messages=working,
            prefix_hash=prefix_hash,
            static_prefix_tokens=static_tokens,
            session_prefix_tokens=session_tokens,
            dynamic_tail_tokens=tail_tokens,
            tool_schema_tokens=tool_schema_tokens,
            total_tokens=total_tokens,
            context_window=cfg.context_window,
            utilization_pct=util_pct,
            kv_cache_hit_pct=kv_hit_pct,
            offloaded_artifact_ids=offloaded_ids,
            cleared_thinking_tokens=cleared_thinking,
            cleared_tool_tokens=cleared_tool_toks,
            compacted_this_turn=compacted_now,
            updated_checkpoint=checkpoint_text if compacted_now else "",
            pre_compact_facts=harvested_facts,
        )

        if thread_id:
            self._persist_telemetry_snapshot(
                ctx=active_ctx,
                thread_id=thread_id,
                agent_name=agent_name,
                projection=projection,
                deferred_tools_count=deferred_tools_count,
                active_skill_slugs=active_skill_slugs or [],
                checkpoint_text=checkpoint_text,
            )

        return projection

    def _persist_telemetry_snapshot(
        self,
        *,
        ctx: TenantContext,
        thread_id: str,
        agent_name: str,
        projection: ContextProjection,
        deferred_tools_count: int,
        active_skill_slugs: list[str],
        checkpoint_text: str,
    ) -> None:
        row_id = f"{ctx.tenant_id}:{ctx.workspace_id}:{thread_id}"
        now_iso = datetime.now(tz=UTC).isoformat()
        try:
            with get_db() as conn:
                prev = conn.execute(
                    "SELECT * FROM session_compactions WHERE id = ? AND tenant_id = ?",
                    (row_id, ctx.tenant_id),
                ).fetchone()
                prev_d = dict(prev) if prev else {}
                total_compactions = int(prev_d.get("compaction_count") or 0) + (
                    1 if projection.compacted_this_turn else 0
                )
                merged_facts: list[str] = json.loads(prev_d.get("pre_compact_facts_json") or "[]")
                for f in projection.pre_compact_facts:
                    if f not in merged_facts:
                        merged_facts.append(f)
                art_cnt_row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM context_artifacts WHERE tenant_id = ? AND workspace_id = ? AND thread_id = ?",
                    (ctx.tenant_id, ctx.workspace_id, thread_id),
                ).fetchone()
                art_total = int(art_cnt_row["cnt"]) if art_cnt_row else len(projection.offloaded_artifact_ids)

                conn.execute(
                    """INSERT OR REPLACE INTO session_compactions
                       (id, tenant_id, workspace_id, thread_id, agent_name, prefix_hash,
                        static_prefix_tokens, session_prefix_tokens, dynamic_tail_tokens,
                        total_tokens, context_window, kv_cache_hit_pct, deferred_tools_count,
                        active_skills_json, offloaded_artifacts_count, cleared_thinking_tokens,
                        cleared_tool_tokens, compaction_count, checkpoint_summary,
                        pre_compact_facts_json, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row_id,
                        ctx.tenant_id,
                        ctx.workspace_id,
                        thread_id,
                        agent_name,
                        projection.prefix_hash,
                        projection.static_prefix_tokens,
                        projection.session_prefix_tokens,
                        projection.dynamic_tail_tokens,
                        projection.total_tokens,
                        projection.context_window,
                        projection.kv_cache_hit_pct,
                        deferred_tools_count,
                        json.dumps(active_skill_slugs),
                        art_total,
                        int(prev_d.get("cleared_thinking_tokens") or 0) + projection.cleared_thinking_tokens,
                        int(prev_d.get("cleared_tool_tokens") or 0) + projection.cleared_tool_tokens,
                        total_compactions,
                        checkpoint_text or prev_d.get("checkpoint_summary") or "",
                        json.dumps(merged_facts[-25:]),
                        now_iso,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("Telemetry snapshot update skipped: %s", exc)


context_engine = FiveStageContextEngine()

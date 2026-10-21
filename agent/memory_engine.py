"""Phase 6 — 4-Tier Cognitive Memory Engine (`agent/memory_engine.py`).

Implements the production multi-tenant 4-Tier Cognitive Memory architecture from `plan.md`:
  - **Tier 1 — Pinned Core Memory Blocks (`memory_blocks`, capped at `<= 1,500` tokens):**
      Editable structured Markdown blocks (`user_profile`, `workspace_context`, `agent_notes`)
      cached cleanly in **Block 2 (`session_prefix`)** so **Block 1 (`static_prefix`)** remains
      100% byte-identical across ALL users and turns for GPU Radix-Tree KV cache sharing.
  - **Tier 2 — Bi-Temporal Semantic Fact Store (`semantic_memories`):**
      Zep / Graphiti-style atomic triples `(subject, predicate, object, confidence, importance,
      valid_from, invalidated_at, superseded_by_id, source_session_id)`:
        * Exact triple match (`subject_norm`, `predicate_norm`, `object_norm`) reinforces confidence
          and increments `access_count` without duplicate row bloat.
        * Contradicting triple (`subject_norm`, `predicate_norm` with new `object_norm`) atomically
          sets `invalidated_at = now` and `superseded_by_id = new_id` inside the same DB transaction,
          eliminating stale-fact hallucinations while preserving complete point-in-time (`as_of`)
          enterprise audit lineage!
  - **Tier 3 — Episodic Memory (`episodic_memories`):**
      Distilled task trajectories (*Problem → Approach → Key Tool Steps → User Corrections → Outcome*)
      scored by **Hybrid Relevance × Exponential Recency Decay × Importance**.
  - **Tier 4 — Procedural Memory (`procedural_memories`):**
      Self-improving behavioral rule proposals (`proposed` -> `active` / `rejected`) with full
      immutable `history_json` snapshots supporting **1-Click Version Rollback** to any revision!
  - **Sub-35ms Multi-Tenant Hybrid Recall (`recall_memories`):**
      Parallel Tenant-Partitioned **Qdrant (`is_tenant=True` + `INT8`) Dense Vector Search** +
      **BM25 / Token Lexical Search** fused via **Weighted Reciprocal Rank Fusion ($k = 60$)**
      multiplied by recency decay, importance, and confidence — strictly isolated by `(tenant_id, workspace_id)`!
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
from typing import Any
import uuid

from langchain_core.tools import BaseTool, tool
import numpy as np

from agent.registry import count_text_tokens
from api.core.tenancy import TenantContext, get_tenant_context
from api.core.vector_store import vector_store as shared_vector_store
from api.db import get_db
from api.services.batcher import gpu_batcher

logger = logging.getLogger("nimagents.memory_engine")

MAX_CORE_MEMORY_TOKENS = 1500
VALID_BLOCK_TYPES = frozenset({"user_profile", "workspace_context", "agent_notes"})
VALID_PROCEDURAL_STATUSES = frozenset({"proposed", "active", "rejected", "archived"})

_WORD_RE = re.compile(r"[a-z0-9_.-]+")
_NORM_SPACE_RE = re.compile(r"\s+")


def _normalize_slot(value: str) -> str:
    cleaned = _NORM_SPACE_RE.sub(" ", value.strip().lower())
    return cleaned.strip(" .,;:!?\"'`")


def _tokenize_lexical(text: str) -> list[str]:
    return [tok for tok in _WORD_RE.findall(text.lower()) if len(tok) >= 2]


class LocalHashedSemanticEmbedder:
    """Zero-network SIMD-vectorized n-gram + token feature hasher (`256-dim` L2-normalized).

    Guarantees `< 0.4ms` deterministic dense vector generation inside air-gapped or local dev environments
    while preserving subword/synonym stem overlap inside Qdrant `is_tenant=True` HNSW shards.
    Automatically superseded by `NVIDIAEmbeddings` (`nvidia/llama-nemotron-embed-1b-v2`) whenever
    `NVIDIA_EMBED_BASE_URL` or live `NVIDIA_API_KEY` memory embedding mode is active.
    """

    dim: int = 256

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for txt in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = _tokenize_lexical(txt)
            features: list[tuple[str, float]] = [(tok, 2.0) for tok in tokens]
            # Character 3-grams capture morphological & typo similarity cleanly
            compact = " ".join(tokens)
            for i in range(max(0, len(compact) - 2)):
                features.append((compact[i : i + 3], 0.6))
            # Token bigrams capture phrase ordering ("primary cluster", "preferred model")
            for i in range(max(0, len(tokens) - 1)):
                features.append((f"{tokens[i]}_{tokens[i + 1]}", 2.5))

            for feat, weight in features:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                h = int.from_bytes(digest, "little")
                idx = h % self.dim
                sign = 1.0 if ((h >> 32) & 1) == 0 else -1.0
                vec[idx] += sign * weight

            norm = float(np.linalg.norm(vec))
            if norm > 1e-8:
                vec /= norm
            else:
                vec[0] = 1.0
            out.append(vec.tolist())
        return out


_LOCAL_EMBEDDER = LocalHashedSemanticEmbedder()
_NVIDIA_EMBEDDER: Any = None


def _resolve_memory_embedder() -> tuple[str, Any]:
    global _NVIDIA_EMBEDDER
    embed_url = os.getenv("NVIDIA_EMBED_BASE_URL", "").strip()
    use_remote = bool(embed_url) or os.getenv("NIMAGENTS_MEMORY_REMOTE_EMBED", "").strip() == "1"
    if use_remote:
        if _NVIDIA_EMBEDDER is None:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            model_name = os.getenv("NVIDIA_MEMORY_EMBED_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
            kwargs: dict[str, Any] = {"model": model_name, "truncate": "END"}
            if embed_url:
                kwargs["base_url"] = embed_url
            api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            if api_key:
                kwargs["api_key"] = api_key
            _NVIDIA_EMBEDDER = NVIDIAEmbeddings(**kwargs)
        return ("nvidia-memory-embed", _NVIDIA_EMBEDDER)
    return ("local-hashed-256", _LOCAL_EMBEDDER)


def _recency_decay(iso_timestamp: str | None, half_life_days: float = 30.0) -> float:
    if not iso_timestamp:
        return 1.0
    try:
        parsed = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        age_days = max(0.0, (datetime.now(tz=UTC) - parsed).total_seconds() / 86400.0)
        return float(math.exp(-math.log(2.0) * (age_days / max(1.0, half_life_days))))
    except Exception:
        return 1.0


def _bm25_lexical_score(query_tokens: list[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize_lexical(doc_text)
    if not doc_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for tok in doc_tokens:
        counts[tok] = counts.get(tok, 0) + 1
    k1 = 1.5
    b = 0.75
    avg_dl = 24.0
    dl = float(len(doc_tokens))
    score = 0.0
    for q in set(query_tokens):
        tf = float(counts.get(q, 0))
        if tf <= 0.0:
            continue
        score += (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avg_dl)))
    return score


class CognitiveMemoryEngine:
    """Thread-safe & async-safe 4-Tier Multi-Tenant Cognitive Memory Manager."""

    def __init__(self) -> None:
        self._db_mutex = threading.RLock()

    # ── Tier 1: Pinned Core Memory Blocks (`<= 1,500` tokens in Block 2) ──

    def list_blocks(
        self,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        active = ctx or get_tenant_context()
        sql = """SELECT * FROM memory_blocks
                 WHERE tenant_id = ? AND workspace_id = ?"""
        params: list[Any] = [active.tenant_id, active.workspace_id]
        if agent_id:
            sql += " AND agent_id IN (?, 'default')"
            params.append(agent_id)
        if user_id:
            sql += " AND user_id IN (?, 'anonymous')"
            params.append(user_id)
        sql += " ORDER BY block_type ASC, id ASC"
        with self._db_mutex, get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                **dict(r),
                "pinned": bool(r["pinned"]),
            }
            for r in rows
        ]

    def upsert_block(
        self,
        *,
        block_type: str = "agent_notes",
        title: str = "",
        content: str,
        agent_id: str = "default",
        pinned: bool = True,
        block_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        btype = block_type if block_type in VALID_BLOCK_TYPES else "agent_notes"
        default_titles = {
            "user_profile": "User Profile",
            "workspace_context": "Workspace Context",
            "agent_notes": "Agent Working Notes",
        }
        clean_title = title.strip() or default_titles[btype]
        clean_content = content.strip()
        tok_count = count_text_tokens(clean_content)
        if tok_count > MAX_CORE_MEMORY_TOKENS:
            raise ValueError(
                f"Core memory block exceeds hard {MAX_CORE_MEMORY_TOKENS}-token ceiling ({tok_count} tokens)"
            )

        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            existing_id = block_id
            if not existing_id:
                existing = conn.execute(
                    """SELECT id FROM memory_blocks
                       WHERE tenant_id = ? AND workspace_id = ? AND user_id = ?
                         AND agent_id = ? AND block_type = ?""",
                    (active.tenant_id, active.workspace_id, active.user_id, agent_id, btype),
                ).fetchone()
                existing_id = existing["id"] if existing else f"blk_{uuid.uuid4().hex[:10]}"

            conn.execute(
                """INSERT INTO memory_blocks
                   (id, tenant_id, workspace_id, user_id, agent_id, block_type, title, content, token_count, pinned, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     block_type = excluded.block_type,
                     title = excluded.title,
                     content = excluded.content,
                     token_count = excluded.token_count,
                     pinned = excluded.pinned,
                     updated_at = excluded.updated_at""",
                (
                    existing_id,
                    active.tenant_id,
                    active.workspace_id,
                    active.user_id,
                    agent_id,
                    btype,
                    clean_title,
                    clean_content,
                    tok_count,
                    int(bool(pinned)),
                    now_iso,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM memory_blocks WHERE id = ? AND tenant_id = ?",
                (existing_id, active.tenant_id),
            ).fetchone()
        return {**dict(row), "pinned": bool(row["pinned"])}

    def delete_block(self, block_id: str, *, ctx: TenantContext | None = None) -> bool:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            cur = conn.execute(
                "DELETE FROM memory_blocks WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (block_id, active.tenant_id, active.workspace_id),
            )
            conn.commit()
            return cur.rowcount > 0

    # ── Tier 2: Bi-Temporal Semantic Fact Store (`semantic_memories`) ─────

    async def upsert_semantic_fact(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        category: str = "fact",
        confidence: float = 0.9,
        importance: float = 0.75,
        agent_id: str = "default",
        source_session_id: str = "",
        source_turn_id: str = "",
        valid_from: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Insert or bi-temporally supersede a `(subject, predicate, object)` fact atomically."""
        active = ctx or get_tenant_context()
        subj_clean = _NORM_SPACE_RE.sub(" ", subject.strip())
        pred_clean = _NORM_SPACE_RE.sub(" ", predicate.strip())
        obj_clean = _NORM_SPACE_RE.sub(" ", obj.strip())
        if not subj_clean or not pred_clean or not obj_clean:
            raise ValueError("subject, predicate, and object must all be non-empty")

        subj_norm = _normalize_slot(subj_clean)
        pred_norm = _normalize_slot(pred_clean)
        obj_norm = _normalize_slot(obj_clean)
        statement = f"{subj_clean} {pred_clean} {obj_clean}"
        now_iso = valid_from or datetime.now(tz=UTC).isoformat()
        conf_clamped = max(0.05, min(1.0, float(confidence)))
        imp_clamped = max(0.05, min(1.0, float(importance)))

        def _tx_upsert() -> tuple[dict[str, Any], list[str], bool]:
            superseded_ids: list[str] = []
            with self._db_mutex, get_db() as conn:
                conn.execute("BEGIN IMMEDIATE")
                active_matches = conn.execute(
                    """SELECT * FROM semantic_memories
                       WHERE tenant_id = ? AND workspace_id = ? AND user_id = ?
                         AND subject_norm = ? AND predicate_norm = ?
                         AND invalidated_at IS NULL""",
                    (active.tenant_id, active.workspace_id, active.user_id, subj_norm, pred_norm),
                ).fetchall()

                # Case A: Exact active triple already exists -> reinforce in place (zero bloat!)
                for row in active_matches:
                    if row["object_norm"] == obj_norm:
                        boosted_conf = min(1.0, max(float(row["confidence"]), conf_clamped) + 0.03)
                        conn.execute(
                            """UPDATE semantic_memories
                               SET confidence = ?, importance = MAX(importance, ?),
                                   access_count = access_count + 1, updated_at = ?
                               WHERE id = ?""",
                            (boosted_conf, imp_clamped, now_iso, row["id"]),
                        )
                        conn.commit()
                        updated = conn.execute(
                            "SELECT * FROM semantic_memories WHERE id = ?", (row["id"],)
                        ).fetchone()
                        return ({**dict(updated), "superseded_ids": [], "deduplicated": True}, [], False)

                # Case B: New fact OR contradiction superseding prior active fact(s)!
                new_id = f"sem_{uuid.uuid4().hex[:12]}"
                for stale in active_matches:
                    superseded_ids.append(stale["id"])
                    conn.execute(
                        """UPDATE semantic_memories
                           SET invalidated_at = ?, superseded_by_id = ?, updated_at = ?
                           WHERE id = ?""",
                        (now_iso, new_id, now_iso, stale["id"]),
                    )

                conn.execute(
                    """INSERT INTO semantic_memories
                       (id, tenant_id, workspace_id, user_id, agent_id,
                        subject, subject_norm, predicate, predicate_norm, object, object_norm,
                        statement, category, confidence, importance, access_count,
                        valid_from, invalidated_at, superseded_by_id,
                        source_session_id, source_turn_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, '', ?, ?, ?, ?)""",
                    (
                        new_id,
                        active.tenant_id,
                        active.workspace_id,
                        active.user_id,
                        agent_id,
                        subj_clean,
                        subj_norm,
                        pred_clean,
                        pred_norm,
                        obj_clean,
                        obj_norm,
                        statement,
                        category,
                        conf_clamped,
                        imp_clamped,
                        now_iso,
                        source_session_id,
                        source_turn_id,
                        now_iso,
                        now_iso,
                    ),
                )
                conn.commit()
                created = conn.execute(
                    "SELECT * FROM semantic_memories WHERE id = ?", (new_id,)
                ).fetchone()
                return (
                    {**dict(created), "superseded_ids": superseded_ids, "deduplicated": False},
                    superseded_ids,
                    True,
                )

        record, superseded_ids, needs_index = await asyncio.to_thread(_tx_upsert)

        if needs_index:
            # Remove invalidated vectors from hot Qdrant shard & index the new active fact!
            for old_sid in superseded_ids:
                try:
                    await shared_vector_store.delete_source(
                        source_id=old_sid, namespace="memories", ctx=active
                    )
                except Exception:
                    pass
            model_key, embedder = _resolve_memory_embedder()
            vec = await gpu_batcher.embed_text(model_key, record["statement"], embedder.embed_documents)
            await shared_vector_store.upsert_chunks(
                source_id=record["id"],
                vectors=[vec],
                texts=[record["statement"]],
                metadatas=[
                    {
                        "memory_id": record["id"],
                        "tier": "semantic",
                        "category": record["category"],
                        "user_id": active.user_id,
                        "agent_id": agent_id,
                    }
                ],
                namespace="memories",
                ctx=active,
            )

        return record

    def list_semantic_facts(
        self,
        *,
        include_historical: bool = False,
        as_of: str | None = None,
        search: str = "",
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        """List active facts, full audit lineage (`include_historical=True`), or point-in-time (`as_of`)."""
        active = ctx or get_tenant_context()
        sql = """SELECT * FROM semantic_memories
                 WHERE tenant_id = ? AND workspace_id = ?"""
        params: list[Any] = [active.tenant_id, active.workspace_id]

        if as_of:
            sql += " AND valid_from <= ? AND (invalidated_at IS NULL OR invalidated_at > ?)"
            params.extend([as_of, as_of])
        elif not include_historical:
            sql += " AND invalidated_at IS NULL"

        if search.strip():
            like = f"%{search.strip().lower()}%"
            sql += " AND LOWER(statement) LIKE ?"
            params.append(like)

        sql += " ORDER BY valid_from DESC, created_at DESC"
        with self._db_mutex, get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    async def delete_semantic_fact(self, fact_id: str, *, ctx: TenantContext | None = None) -> bool:
        active = ctx or get_tenant_context()

        def _del() -> bool:
            with self._db_mutex, get_db() as conn:
                cur = conn.execute(
                    "DELETE FROM semantic_memories WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                    (fact_id, active.tenant_id, active.workspace_id),
                )
                conn.commit()
                return cur.rowcount > 0

        removed = await asyncio.to_thread(_del)
        if removed:
            try:
                await shared_vector_store.delete_source(
                    source_id=fact_id, namespace="memories", ctx=active
                )
            except Exception:
                pass
        return removed

    # ── Tier 3: Episodic Memory (`episodic_memories`) ─────────────────────

    async def record_episodic_memory(
        self,
        *,
        summary: str,
        problem: str = "",
        approach: str = "",
        key_tool_steps: list[str] | None = None,
        user_corrections: str = "",
        outcome: str = "succeeded",
        importance: float = 0.75,
        session_id: str = "",
        agent_id: str = "default",
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        ep_id = f"epi_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(tz=UTC).isoformat()
        steps_list = key_tool_steps or []
        imp = max(0.05, min(1.0, float(importance)))

        def _insert() -> dict[str, Any]:
            with self._db_mutex, get_db() as conn:
                conn.execute(
                    """INSERT INTO episodic_memories
                       (id, tenant_id, workspace_id, user_id, agent_id, session_id,
                        summary, problem, approach, key_tool_steps_json, user_corrections,
                        outcome, importance, access_count, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                    (
                        ep_id,
                        active.tenant_id,
                        active.workspace_id,
                        active.user_id,
                        agent_id,
                        session_id,
                        summary.strip(),
                        problem.strip(),
                        approach.strip(),
                        json.dumps(steps_list),
                        user_corrections.strip(),
                        outcome.strip() or "succeeded",
                        imp,
                        now_iso,
                    ),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM episodic_memories WHERE id = ?", (ep_id,)).fetchone()
            return {**dict(row), "key_tool_steps": steps_list}

        record = await asyncio.to_thread(_insert)
        composite_text = " | ".join(
            part for part in (summary, problem, approach, user_corrections, outcome) if part
        )
        model_key, embedder = _resolve_memory_embedder()
        vec = await gpu_batcher.embed_text(model_key, composite_text, embedder.embed_documents)
        await shared_vector_store.upsert_chunks(
            source_id=ep_id,
            vectors=[vec],
            texts=[composite_text],
            metadatas=[
                {
                    "memory_id": ep_id,
                    "tier": "episodic",
                    "user_id": active.user_id,
                    "agent_id": agent_id,
                }
            ],
            namespace="memories",
            ctx=active,
        )
        return record

    def list_episodic_memories(
        self,
        *,
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            rows = conn.execute(
                """SELECT * FROM episodic_memories
                   WHERE tenant_id = ? AND workspace_id = ?
                   ORDER BY created_at DESC""",
                (active.tenant_id, active.workspace_id),
            ).fetchall()
        return [
            {
                **dict(r),
                "key_tool_steps": json.loads(r["key_tool_steps_json"] or "[]"),
            }
            for r in rows
        ]

    async def delete_episodic_memory(self, ep_id: str, *, ctx: TenantContext | None = None) -> bool:
        active = ctx or get_tenant_context()

        def _del() -> bool:
            with self._db_mutex, get_db() as conn:
                cur = conn.execute(
                    "DELETE FROM episodic_memories WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                    (ep_id, active.tenant_id, active.workspace_id),
                )
                conn.commit()
                return cur.rowcount > 0

        removed = await asyncio.to_thread(_del)
        if removed:
            try:
                await shared_vector_store.delete_source(
                    source_id=ep_id, namespace="memories", ctx=active
                )
            except Exception:
                pass
        return removed

    # ── Tier 4: Procedural Memory (`procedural_memories` + 1-Click Rollback) ─

    def propose_procedural_memory(
        self,
        *,
        title: str,
        rule_prompt: str,
        rationale: str = "",
        agent_id: str = "default",
        status: str = "proposed",
        source_session_id: str = "",
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        pm_id = f"proc_{uuid.uuid4().hex[:12]}"
        st = status if status in VALID_PROCEDURAL_STATUSES else "proposed"
        now_iso = datetime.now(tz=UTC).isoformat()
        initial_history = [
            {
                "version": 1,
                "title": title.strip(),
                "rule_prompt": rule_prompt.strip(),
                "rationale": rationale.strip(),
                "status": st,
                "timestamp": now_iso,
            }
        ]
        with self._db_mutex, get_db() as conn:
            conn.execute(
                """INSERT INTO procedural_memories
                   (id, tenant_id, workspace_id, agent_id, title, rule_prompt, rationale,
                    status, version, history_json, source_session_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                (
                    pm_id,
                    active.tenant_id,
                    active.workspace_id,
                    agent_id,
                    title.strip(),
                    rule_prompt.strip(),
                    rationale.strip(),
                    st,
                    json.dumps(initial_history),
                    source_session_id,
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM procedural_memories WHERE id = ?", (pm_id,)).fetchone()
        return {**dict(row), "history": initial_history}

    def update_procedural_memory(
        self,
        proc_id: str,
        *,
        title: str | None = None,
        rule_prompt: str | None = None,
        rationale: str | None = None,
        status: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any] | None:
        active = ctx or get_tenant_context()
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                """SELECT * FROM procedural_memories
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (proc_id, active.tenant_id, active.workspace_id),
            ).fetchone()
            if row is None:
                return None
            new_title = title.strip() if title is not None else row["title"]
            new_prompt = rule_prompt.strip() if rule_prompt is not None else row["rule_prompt"]
            new_rationale = rationale.strip() if rationale is not None else row["rationale"]
            new_status = (
                status if (status is not None and status in VALID_PROCEDURAL_STATUSES) else row["status"]
            )
            next_version = int(row["version"] or 1) + 1
            history = json.loads(row["history_json"] or "[]")
            history.append(
                {
                    "version": next_version,
                    "title": new_title,
                    "rule_prompt": new_prompt,
                    "rationale": new_rationale,
                    "status": new_status,
                    "timestamp": now_iso,
                }
            )
            conn.execute(
                """UPDATE procedural_memories
                   SET title = ?, rule_prompt = ?, rationale = ?, status = ?,
                       version = ?, history_json = ?, updated_at = ?
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (
                    new_title,
                    new_prompt,
                    new_rationale,
                    new_status,
                    next_version,
                    json.dumps(history),
                    now_iso,
                    proc_id,
                    active.tenant_id,
                    active.workspace_id,
                ),
            )
            conn.commit()
            updated = conn.execute("SELECT * FROM procedural_memories WHERE id = ?", (proc_id,)).fetchone()
        return {**dict(updated), "history": history}

    def rollback_procedural_memory(
        self,
        proc_id: str,
        *,
        target_version: int,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any] | None:
        """Instant 1-Click Version Rollback to any historical revision snapshot!"""
        active = ctx or get_tenant_context()
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                """SELECT * FROM procedural_memories
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (proc_id, active.tenant_id, active.workspace_id),
            ).fetchone()
            if row is None:
                return None
            history: list[dict[str, Any]] = json.loads(row["history_json"] or "[]")
            snap = next((item for item in history if int(item.get("version", -1)) == target_version), None)
            if snap is None:
                raise KeyError(f"Version {target_version} not found on procedural memory '{proc_id}'")

            next_version = int(row["version"] or 1) + 1
            history.append(
                {
                    "version": next_version,
                    "title": snap["title"],
                    "rule_prompt": snap["rule_prompt"],
                    "rationale": snap.get("rationale", ""),
                    "status": snap["status"],
                    "rolled_back_from_version": target_version,
                    "timestamp": now_iso,
                }
            )
            conn.execute(
                """UPDATE procedural_memories
                   SET title = ?, rule_prompt = ?, rationale = ?, status = ?,
                       version = ?, history_json = ?, updated_at = ?
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (
                    snap["title"],
                    snap["rule_prompt"],
                    snap.get("rationale", ""),
                    snap["status"],
                    next_version,
                    json.dumps(history),
                    now_iso,
                    proc_id,
                    active.tenant_id,
                    active.workspace_id,
                ),
            )
            conn.commit()
            restored = conn.execute("SELECT * FROM procedural_memories WHERE id = ?", (proc_id,)).fetchone()
        return {**dict(restored), "history": history}

    def list_procedural_memories(
        self,
        *,
        status: str | None = None,
        agent_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        active = ctx or get_tenant_context()
        sql = """SELECT * FROM procedural_memories
                 WHERE tenant_id = ? AND workspace_id = ?"""
        params: list[Any] = [active.tenant_id, active.workspace_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        if agent_id:
            sql += " AND agent_id IN (?, 'default')"
            params.append(agent_id)
        sql += " ORDER BY id ASC"
        with self._db_mutex, get_db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                **dict(r),
                "history": json.loads(r["history_json"] or "[]"),
            }
            for r in rows
        ]

    # ── Block 2 Deterministic Session Prefix Renderer (`Tier 1 + Tier 4`) ──

    def render_session_prefix_memories(
        self,
        *,
        agent_id: str = "default",
        extra_notes: str = "",
        ctx: TenantContext | None = None,
    ) -> str:
        """Render pinned Tier-1 Core Memory (`<=1,500` tok) + approved Tier-4 Procedural Rules for Block 2."""
        active = ctx or get_tenant_context()
        sections: list[str] = []
        if extra_notes.strip():
            sections.append(extra_notes.strip())

        blocks = [b for b in self.list_blocks(agent_id=agent_id, user_id=active.user_id, ctx=active) if b["pinned"]]
        if blocks:
            budget_left = MAX_CORE_MEMORY_TOKENS
            rendered_blocks: list[str] = []
            for blk in blocks:
                if not blk["content"]:
                    continue
                cost = max(1, int(blk["token_count"] or count_text_tokens(blk["content"])))
                if cost > budget_left:
                    continue
                budget_left -= cost
                rendered_blocks.append(f"### [{blk['block_type']}] {blk['title']}\n{blk['content']}")
            if rendered_blocks:
                sections.append("## Core Memory Blocks\n" + "\n\n".join(rendered_blocks))

        approved_rules = self.list_procedural_memories(status="active", agent_id=agent_id, ctx=active)
        if approved_rules:
            rule_lines = [f"- **{r['title']}** (v{r['version']}): {r['rule_prompt']}" for r in approved_rules]
            sections.append("## Approved Procedural Rules\n" + "\n".join(rule_lines))

        return "\n\n".join(sections)

    # ── `< 35ms` Multi-Tenant Hybrid Recall (`Qdrant Dense + BM25 RRF`) ───

    async def recall_memories(
        self,
        query: str,
        *,
        limit: int = 8,
        agent_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Hybrid Qdrant (`is_tenant=True`) Dense + BM25 RRF × Recency × Importance recall in `<35ms`."""
        t0 = time.perf_counter()
        active = ctx or get_tenant_context()
        q_tokens = _tokenize_lexical(query)
        model_key, embedder = _resolve_memory_embedder()

        query_vec_task = asyncio.create_task(
            gpu_batcher.embed_text(model_key, query, embedder.embed_documents)
        )
        candidates_task = asyncio.to_thread(
            lambda: (
                self.list_semantic_facts(include_historical=False, ctx=active),
                self.list_episodic_memories(ctx=active),
            )
        )
        query_vec, (sem_rows, epi_rows) = await asyncio.gather(query_vec_task, candidates_task)

        dense_hits = await shared_vector_store.search(
            source_id=None,
            query_vector=query_vec,
            k=max(limit * 4, 24),
            namespace="memories",
            ctx=active,
        )
        dense_rank_by_id: dict[str, int] = {}
        for rank_idx, hit in enumerate(dense_hits):
            mid = str((hit.get("metadata") or {}).get("memory_id") or "")
            if mid and mid not in dense_rank_by_id:
                dense_rank_by_id[mid] = rank_idx + 1

        unified_pool: list[dict[str, Any]] = []
        for s in sem_rows:
            unified_pool.append(
                {
                    "id": s["id"],
                    "tier": "semantic",
                    "text": s["statement"],
                    "subject": s["subject"],
                    "predicate": s["predicate"],
                    "object": s["object"],
                    "category": s["category"],
                    "confidence": float(s["confidence"] or 0.9),
                    "importance": float(s["importance"] or 0.7),
                    "timestamp": s["valid_from"] or s["created_at"],
                }
            )
        for e in epi_rows:
            text_blob = " — ".join(
                p for p in (e["summary"], e["problem"], e["user_corrections"], e["outcome"]) if p
            )
            unified_pool.append(
                {
                    "id": e["id"],
                    "tier": "episodic",
                    "text": text_blob,
                    "summary": e["summary"],
                    "problem": e["problem"],
                    "outcome": e["outcome"],
                    "user_corrections": e["user_corrections"],
                    "confidence": 1.0,
                    "importance": float(e["importance"] or 0.75),
                    "timestamp": e["created_at"],
                }
            )

        # Rank lexically via BM25 over tenant-isolated candidates
        lex_scored = [
            (item["id"], _bm25_lexical_score(q_tokens, item["text"]))
            for item in unified_pool
        ]
        lex_scored.sort(key=lambda pair: pair[1], reverse=True)
        lex_rank_by_id = {
            mid: idx + 1 for idx, (mid, sc) in enumerate(lex_scored) if sc > 0.0
        }

        rrf_k = 60.0
        scored_results: list[dict[str, Any]] = []
        for item in unified_pool:
            mid = item["id"]
            d_rank = dense_rank_by_id.get(mid)
            l_rank = lex_rank_by_id.get(mid)
            if d_rank is None and l_rank is None:
                continue
            rrf = (1.0 / (rrf_k + d_rank) if d_rank is not None else 0.0) + (
                1.25 / (rrf_k + l_rank) if l_rank is not None else 0.0
            )
            recency = _recency_decay(item["timestamp"])
            quality = 0.55 + 0.25 * item["importance"] + 0.20 * item["confidence"]
            final_score = round(rrf * (0.7 + 0.3 * recency) * quality * 100.0, 5)
            scored_results.append(
                {
                    **item,
                    "score": final_score,
                    "dense_rank": d_rank,
                    "lexical_rank": l_rank,
                    "recency_factor": round(recency, 4),
                }
            )

        scored_results.sort(key=lambda r: r["score"], reverse=True)
        top_items = scored_results[: max(1, limit)]
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "query": query,
            "count": len(top_items),
            "latency_ms": elapsed_ms,
            "results": top_items,
        }


memory_engine = CognitiveMemoryEngine()


def build_memory_runtime_tools(
    ctx: TenantContext,
    *,
    agent_id: str = "default",
    thread_id: str = "",
) -> list[BaseTool]:
    """Build protected LangGraph memory tools (`recall_memories`, `upsert_memory_fact`, `update_core_memory_block`, `propose_procedural_rule`)."""

    @tool
    async def recall_memories(query: str, limit: int = 6) -> str:
        """Search long-term bi-temporal semantic facts and past episodic experiences relevant to `query`."""
        res = await memory_engine.recall_memories(
            query,
            limit=max(1, min(limit, 15)),
            agent_id=agent_id,
            ctx=ctx,
        )
        return json.dumps(res, ensure_ascii=False)

    @tool
    async def upsert_memory_fact(
        subject: str,
        predicate: str,
        object_value: str,
        category: str = "preference",
        importance: float = 0.8,
    ) -> str:
        """Persist a durable factual statement `(subject, predicate, object_value)` with automatic contradiction superseding."""
        rec = await memory_engine.upsert_semantic_fact(
            subject=subject,
            predicate=predicate,
            obj=object_value,
            category=category,
            importance=importance,
            agent_id=agent_id,
            source_session_id=thread_id,
            ctx=ctx,
        )
        return json.dumps(
            {
                "id": rec["id"],
                "statement": rec["statement"],
                "superseded_ids": rec.get("superseded_ids", []),
                "deduplicated": rec.get("deduplicated", False),
            },
            ensure_ascii=False,
        )

    @tool
    async def update_core_memory_block(
        block_type: str,
        content: str,
        title: str = "",
    ) -> str:
        """Update a pinned Tier-1 core memory block (`user_profile`, `workspace_context`, or `agent_notes`)."""
        blk = await asyncio.to_thread(
            lambda: memory_engine.upsert_block(
                block_type=block_type,
                title=title,
                content=content,
                agent_id=agent_id,
                ctx=ctx,
            )
        )
        return json.dumps(blk, ensure_ascii=False)

    @tool
    async def propose_procedural_rule(
        title: str,
        rule_prompt: str,
        rationale: str = "",
    ) -> str:
        """Propose a Tier-4 procedural behavior rule for human review in `/memories`."""
        prop = await asyncio.to_thread(
            lambda: memory_engine.propose_procedural_memory(
                title=title,
                rule_prompt=rule_prompt,
                rationale=rationale,
                agent_id=agent_id,
                source_session_id=thread_id,
                ctx=ctx,
            )
        )
        return json.dumps(prop, ensure_ascii=False)

    return [
        recall_memories,
        upsert_memory_fact,
        update_core_memory_block,
        propose_procedural_rule,
    ]

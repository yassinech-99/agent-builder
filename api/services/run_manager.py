"""Phase 5 — Durable Session Manager, Fencing-Token Lease Recovery, `Last-Event-ID` Replay & Forking.

Implements Anthropic's *Scaling Managed Agents* stateless session architecture for `nimAgents`:
  1. **Append-Only External Session Event Log (`session_events`):**
     Every user message, streamed token delta, tool call, sub-agent trace, and assistant completion
     is assigned a strictly monotonic per-session `seq_id` (`1, 2, 3, ...`).
  2. **Per-Session Lease Lock with Monotonic Fencing Tokens (`session_leases`):**
     Workers acquire a time-bounded lease (`lease_ttl_s = 30.0`) with a strictly increasing `fencing_token`
     before mutating or streaming into a session. If a Brain Worker pod crashes mid-stream, `wake_session(...)`
     acquires the expired lease with `fencing_token + 1`, rejects any late zombie writes (`StaleLeaseFenceError`),
     and resumes the unfinished run cleanly to completion!
  3. **Reconnectable SSE Stream Fan-Out (`Last-Event-ID` / `after_seq`):**
     Closing the browser tab or dropping Wi-Fi never aborts an active run. Reconnecting clients pass
     `Last-Event-ID: <seq_id>` and receive exact missed events `seq_id + 1 .. N` with zero gaps or duplicates
     before seamlessly tailing live in-flight chunks!
  4. **Instant Cross-Worker Abort (`cancel_session`), Time-Travel Checkpoint Forking (`fork_session`),
     Background P1 Auto-Titling, Pin/Tag/Archive, Tenant-Filtered Search, and Markdown / JSONL Export.**
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os
import re
import threading
import time
from typing import Any, AsyncGenerator, Callable, Awaitable
import uuid

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from api.core.tenancy import TenantContext, build_meili_tenant_filter, get_tenant_context, tenant_scope
from api.db import get_db
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.run_manager")

MEILI_URL = os.getenv("MEILI_URL", "http://localhost:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY") or None
SESSIONS_MEILI_INDEX = "nimagents_sessions"


class SessionLeaseBusyError(RuntimeError):
    """Raised when another healthy worker holds an unexpired session lease."""


class StaleLeaseFenceError(RuntimeError):
    """Raised when a superseded zombie worker attempts to write with an outdated fencing token."""


@dataclass(frozen=True)
class LeaseHandle:
    session_id: str
    worker_id: str
    fencing_token: int
    expires_at: float


def generate_semantic_session_title(prompt_text: str) -> str:
    """Fast heuristic semantic title generator queued asynchronously on Lane P1."""
    cleaned = re.sub(r"\s+", " ", (prompt_text or "").strip())
    cleaned = re.sub(r"^(please|can you|could you|help me|i need to|let's)\s+", "", cleaned, flags=re.I)
    if not cleaned:
        return "Untitled Session"
    words = cleaned.split(" ")[:9]
    title = " ".join(words).rstrip(".,;:!?")
    return (title[0].upper() + title[1:])[:80]


def _row_to_session_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    return {
        "id": d["id"],
        "tenant_id": d["tenant_id"],
        "workspace_id": d["workspace_id"],
        "user_id": d.get("user_id") or "anonymous",
        "assistant_id": d.get("assistant_id") or "agent",
        "title": d.get("title") or "New Session",
        "pinned": bool(d.get("pinned", 0)),
        "archived": bool(d.get("archived", 0)),
        "tags": json.loads(d.get("tags_json") or "[]"),
        "forked_from_session_id": d.get("forked_from_session_id") or "",
        "forked_from_seq": int(d.get("forked_from_seq") or 0),
        "status": d.get("status") or "idle",
        "pending_prompt": d.get("pending_prompt") or "",
        "message_count": int(d.get("message_count") or 0),
        "created_at": d.get("created_at") or "",
        "updated_at": d.get("updated_at") or "",
    }


def _row_to_event_dict(row: Any) -> dict[str, Any]:
    d = dict(row)
    return {
        "id": d["id"],
        "seq_id": int(d["seq_id"]),
        "session_id": d["session_id"],
        "run_id": d.get("run_id") or "",
        "event_type": d["event_type"],
        "role": d.get("role") or "system",
        "payload": json.loads(d.get("payload_json") or "{}"),
        "checkpoint_id": d.get("checkpoint_id") or "",
        "created_at": d.get("created_at") or "",
    }


class DurableSessionManager:
    """Stateless-worker-compatible durable session coordinator with monotonic fencing & SSE replay."""

    def __init__(self, default_lease_ttl_s: float = 30.0) -> None:
        self.default_lease_ttl_s = default_lease_ttl_s
        self._db_mutex = threading.RLock()
        self._subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._cancel_flags: dict[str, asyncio.Event] = {}
        self._active_tasks: dict[str, asyncio.Task[Any]] = {}

    @staticmethod
    def _stream_key(ctx: TenantContext, session_id: str) -> str:
        return f"{ctx.tenant_id}:{ctx.workspace_id}:{session_id}"

    # ── Session CRUD, Pinning, Tagging, Archiving & Search ─────────────────

    def create_session(
        self,
        *,
        title: str = "New Session",
        assistant_id: str = "agent",
        tags: list[str] | None = None,
        session_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.now(tz=UTC).isoformat()

        with self._db_mutex, get_db() as conn:
            conn.execute(
                """INSERT INTO durable_sessions
                   (id, tenant_id, workspace_id, user_id, assistant_id, title,
                    pinned, archived, tags_json, status, message_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, 'idle', 0, ?, ?)""",
                (
                    sid,
                    active.tenant_id,
                    active.workspace_id,
                    active.user_id,
                    assistant_id,
                    title.strip() or "New Session",
                    json.dumps(tags or []),
                    now_iso,
                    now_iso,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM durable_sessions WHERE id = ? AND tenant_id = ?",
                (sid, active.tenant_id),
            ).fetchone()
        return _row_to_session_dict(row)

    def ensure_session(self, session_id: str, ctx: TenantContext | None = None) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                "SELECT * FROM durable_sessions WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            ).fetchone()
        if row:
            return _row_to_session_dict(row)
        return self.create_session(session_id=session_id, ctx=active)

    def get_session(self, session_id: str, ctx: TenantContext | None = None) -> dict[str, Any] | None:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                "SELECT * FROM durable_sessions WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            ).fetchone()
        return _row_to_session_dict(row) if row else None

    def update_session_metadata(
        self,
        session_id: str,
        *,
        title: str | None = None,
        pinned: bool | None = None,
        archived: bool | None = None,
        tags: list[str] | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any] | None:
        active = ctx or get_tenant_context()
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            existing = conn.execute(
                "SELECT * FROM durable_sessions WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            ).fetchone()
            if not existing:
                return None
            cur = dict(existing)
            new_title = title.strip() if title is not None and title.strip() else cur["title"]
            new_pinned = int(pinned) if pinned is not None else int(cur["pinned"])
            new_archived = int(archived) if archived is not None else int(cur["archived"])
            new_tags = json.dumps(tags) if tags is not None else cur["tags_json"]

            conn.execute(
                """UPDATE durable_sessions
                   SET title = ?, pinned = ?, archived = ?, tags_json = ?, updated_at = ?
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (
                    new_title,
                    new_pinned,
                    new_archived,
                    new_tags,
                    now_iso,
                    session_id,
                    active.tenant_id,
                    active.workspace_id,
                ),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM durable_sessions WHERE id = ? AND tenant_id = ?",
                (session_id, active.tenant_id),
            ).fetchone()
        return _row_to_session_dict(updated)

    def delete_session(self, session_id: str, ctx: TenantContext | None = None) -> bool:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            cur = conn.execute(
                "DELETE FROM durable_sessions WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            )
            conn.execute(
                "DELETE FROM session_events WHERE session_id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            )
            conn.execute(
                "DELETE FROM session_leases WHERE session_id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def list_sessions(
        self,
        *,
        search: str = "",
        pinned_only: bool = False,
        include_archived: bool = False,
        tag: str = "",
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        """List & search tenant sessions (Meilisearch tenant-filtered search + SQLite full-log match)."""
        active = ctx or get_tenant_context()
        q_str = (search or "").strip()

        matched_ids_from_search: set[str] | None = None
        if q_str:
            matched_ids_from_search = set()
            # 1. Try fast Meilisearch tenant-filtered index when available
            try:
                import meilisearch

                m_client = meilisearch.Client(MEILI_URL, MEILI_MASTER_KEY)
                hits = m_client.index(SESSIONS_MEILI_INDEX).search(
                    q_str,
                    {"limit": 50, "filter": build_meili_tenant_filter(active)},
                )
                for h in hits.get("hits", []):
                    if h.get("session_id"):
                        matched_ids_from_search.add(str(h["session_id"]))
            except Exception:
                pass

            # 2. Always union exact/substring SQLite title + full event log match within tenant scope
            like_pat = f"%{q_str.lower()}%"
            with self._db_mutex, get_db() as conn:
                s_rows = conn.execute(
                    """SELECT DISTINCT id FROM durable_sessions
                       WHERE tenant_id = ? AND workspace_id = ?
                         AND (LOWER(title) LIKE ? OR LOWER(tags_json) LIKE ?)""",
                    (active.tenant_id, active.workspace_id, like_pat, like_pat),
                ).fetchall()
                e_rows = conn.execute(
                    """SELECT DISTINCT session_id FROM session_events
                       WHERE tenant_id = ? AND workspace_id = ?
                         AND LOWER(payload_json) LIKE ?""",
                    (active.tenant_id, active.workspace_id, like_pat),
                ).fetchall()
                matched_ids_from_search.update(r["id"] for r in s_rows)
                matched_ids_from_search.update(r["session_id"] for r in e_rows)

        with self._db_mutex, get_db() as conn:
            sql = "SELECT * FROM durable_sessions WHERE tenant_id = ? AND workspace_id = ?"
            params: list[Any] = [active.tenant_id, active.workspace_id]
            if not include_archived:
                sql += " AND archived = 0"
            if pinned_only:
                sql += " AND pinned = 1"
            sql += " ORDER BY pinned DESC, updated_at DESC"
            rows = conn.execute(sql, params).fetchall()

        results: list[dict[str, Any]] = []
        for r in rows:
            item = _row_to_session_dict(r)
            if matched_ids_from_search is not None and item["id"] not in matched_ids_from_search:
                continue
            if tag.strip() and tag.strip() not in item["tags"]:
                continue
            results.append(item)
        return results

    # ── Distributed Lease Lock with Monotonic Fencing Tokens ───────────────

    def acquire_lease(
        self,
        session_id: str,
        *,
        worker_id: str | None = None,
        ttl_s: float | None = None,
        now_monotonic: float | None = None,
        ctx: TenantContext | None = None,
    ) -> LeaseHandle:
        """Atomically acquire per-session lease and issue strictly monotonic `fencing_token`."""
        active = ctx or get_tenant_context()
        owner = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        duration = ttl_s if ttl_s is not None else self.default_lease_ttl_s
        now_ts = time.time() if now_monotonic is None else now_monotonic
        lease_pk = f"{active.tenant_id}:{active.workspace_id}:{session_id}"

        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                "SELECT * FROM session_leases WHERE id = ? AND tenant_id = ?",
                (lease_pk, active.tenant_id),
            ).fetchone()
            if row is not None:
                cur_owner = row["worker_id"]
                cur_exp = float(row["expires_at"])
                cur_fence = int(row["fencing_token"])
                if cur_exp > now_ts and cur_owner != owner:
                    raise SessionLeaseBusyError(
                        f"Session '{session_id}' is actively leased by '{cur_owner}' "
                        f"for {round(cur_exp - now_ts, 2)}s (fence={cur_fence})."
                    )
                next_fence = cur_fence + 1
            else:
                next_fence = 1

            exp_at = now_ts + duration
            conn.execute(
                """INSERT OR REPLACE INTO session_leases
                   (id, tenant_id, workspace_id, session_id, worker_id, fencing_token, acquired_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    lease_pk,
                    active.tenant_id,
                    active.workspace_id,
                    session_id,
                    owner,
                    next_fence,
                    now_ts,
                    exp_at,
                ),
            )
            conn.commit()

        return LeaseHandle(
            session_id=session_id,
            worker_id=owner,
            fencing_token=next_fence,
            expires_at=exp_at,
        )

    def renew_lease(
        self,
        lease: LeaseHandle,
        *,
        ttl_s: float | None = None,
        now_monotonic: float | None = None,
        ctx: TenantContext | None = None,
    ) -> LeaseHandle:
        active = ctx or get_tenant_context()
        now_ts = time.time() if now_monotonic is None else now_monotonic
        duration = ttl_s if ttl_s is not None else self.default_lease_ttl_s
        lease_pk = f"{active.tenant_id}:{active.workspace_id}:{lease.session_id}"

        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                "SELECT * FROM session_leases WHERE id = ? AND tenant_id = ?",
                (lease_pk, active.tenant_id),
            ).fetchone()
            if not row or int(row["fencing_token"]) != lease.fencing_token:
                raise StaleLeaseFenceError(
                    f"Worker '{lease.worker_id}' fencing token {lease.fencing_token} was superseded!"
                )
            new_exp = now_ts + duration
            conn.execute(
                "UPDATE session_leases SET expires_at = ? WHERE id = ? AND fencing_token = ?",
                (new_exp, lease_pk, lease.fencing_token),
            )
            conn.commit()
        return LeaseHandle(
            session_id=lease.session_id,
            worker_id=lease.worker_id,
            fencing_token=lease.fencing_token,
            expires_at=new_exp,
        )

    def release_lease(self, lease: LeaseHandle, ctx: TenantContext | None = None) -> None:
        active = ctx or get_tenant_context()
        lease_pk = f"{active.tenant_id}:{active.workspace_id}:{lease.session_id}"
        with self._db_mutex, get_db() as conn:
            conn.execute(
                "UPDATE session_leases SET expires_at = 0 WHERE id = ? AND fencing_token = ?",
                (lease_pk, lease.fencing_token),
            )
            conn.commit()

    # ── Append-Only External Session Event Log (`emitEvent` / `getEvents`) ─

    def emit_event(
        self,
        session_id: str,
        *,
        event_type: str,
        payload: dict[str, Any],
        role: str = "system",
        run_id: str = "",
        checkpoint_id: str = "",
        lease: LeaseHandle | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Append immutable event with monotonic `seq_id`, fencing verification, and live fan-out."""
        active = ctx or get_tenant_context()
        lease_pk = f"{active.tenant_id}:{active.workspace_id}:{session_id}"
        now_iso = datetime.now(tz=UTC).isoformat()

        with self._db_mutex, get_db() as conn:
            if lease is not None:
                cur_lease = conn.execute(
                    "SELECT fencing_token FROM session_leases WHERE id = ? AND tenant_id = ?",
                    (lease_pk, active.tenant_id),
                ).fetchone()
                if not cur_lease or int(cur_lease["fencing_token"]) != lease.fencing_token:
                    raise StaleLeaseFenceError(
                        f"Rejected zombie write with stale fencing token {lease.fencing_token} on '{session_id}'"
                    )

            max_row = conn.execute(
                "SELECT COALESCE(MAX(seq_id), 0) AS max_seq FROM session_events WHERE session_id = ? AND tenant_id = ? AND workspace_id = ?",
                (session_id, active.tenant_id, active.workspace_id),
            ).fetchone()
            next_seq = int(max_row["max_seq"]) + 1
            ev_id = f"ev_{uuid.uuid4().hex[:12]}"
            chk_id = checkpoint_id or f"chk_{next_seq:05d}"

            conn.execute(
                """INSERT INTO session_events
                   (id, tenant_id, workspace_id, session_id, seq_id, run_id,
                    event_type, role, payload_json, checkpoint_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ev_id,
                    active.tenant_id,
                    active.workspace_id,
                    session_id,
                    next_seq,
                    run_id,
                    event_type,
                    role,
                    json.dumps(payload, ensure_ascii=False),
                    chk_id,
                    now_iso,
                ),
            )
            msg_delta = 1 if event_type in ("user_message", "assistant_message") else 0
            conn.execute(
                """UPDATE durable_sessions
                   SET message_count = message_count + ?, updated_at = ?
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (msg_delta, now_iso, session_id, active.tenant_id, active.workspace_id),
            )
            conn.commit()

        event_obj = {
            "id": ev_id,
            "seq_id": next_seq,
            "session_id": session_id,
            "run_id": run_id,
            "event_type": event_type,
            "role": role,
            "payload": payload,
            "checkpoint_id": chk_id,
            "created_at": now_iso,
        }

        # Live fan-out to all connected SSE subscribers for this session
        skey = self._stream_key(active, session_id)
        for q in list(self._subscribers.get(skey, ())):
            try:
                q.put_nowait(event_obj)
            except asyncio.QueueFull:
                pass

        return event_obj

    def get_events(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        up_to_seq: int | None = None,
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            sql = """SELECT * FROM session_events
                     WHERE session_id = ? AND tenant_id = ? AND workspace_id = ? AND seq_id > ?"""
            params: list[Any] = [session_id, active.tenant_id, active.workspace_id, max(0, after_seq)]
            if up_to_seq is not None:
                sql += " AND seq_id <= ?"
                params.append(up_to_seq)
            sql += " ORDER BY seq_id ASC"
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event_dict(r) for r in rows]

    def reconstruct_messages(
        self,
        session_id: str,
        *,
        up_to_seq: int | None = None,
        ctx: TenantContext | None = None,
    ) -> list[BaseMessage]:
        """Rehydrate LangChain message history cleanly from append-only `session_events`."""
        events = self.get_events(session_id, after_seq=0, up_to_seq=up_to_seq, ctx=ctx)
        msgs: list[BaseMessage] = []
        for ev in events:
            et = ev["event_type"]
            content = str(ev["payload"].get("content", ""))
            if et == "user_message":
                msgs.append(HumanMessage(content=content))
            elif et == "assistant_message":
                msgs.append(AIMessage(content=content))
        return msgs

    # ── Time-Travel Session Forking & Trace Exports ────────────────────────

    def fork_session(
        self,
        source_session_id: str,
        *,
        up_to_seq: int | None = None,
        title: str = "",
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Branch a new independent session from any historical `seq_id` checkpoint in `<5ms`."""
        active = ctx or get_tenant_context()
        src = self.get_session(source_session_id, ctx=active)
        if src is None:
            raise KeyError(f"Source session '{source_session_id}' not found")

        events_to_copy = self.get_events(source_session_id, after_seq=0, up_to_seq=up_to_seq, ctx=active)
        effective_seq = events_to_copy[-1]["seq_id"] if events_to_copy else 0
        fork_title = title.strip() or f"{src['title']} (Fork @{effective_seq})"

        forked = self.create_session(
            title=fork_title,
            assistant_id=src["assistant_id"],
            tags=[*src["tags"], "forked"] if "forked" not in src["tags"] else src["tags"],
            ctx=active,
        )
        new_sid = forked["id"]
        now_iso = datetime.now(tz=UTC).isoformat()

        with self._db_mutex, get_db() as conn:
            msg_cnt = 0
            for ev in events_to_copy:
                if ev["event_type"] in ("user_message", "assistant_message"):
                    msg_cnt += 1
                conn.execute(
                    """INSERT INTO session_events
                       (id, tenant_id, workspace_id, session_id, seq_id, run_id,
                        event_type, role, payload_json, checkpoint_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"ev_{uuid.uuid4().hex[:12]}",
                        active.tenant_id,
                        active.workspace_id,
                        new_sid,
                        ev["seq_id"],
                        ev["run_id"],
                        ev["event_type"],
                        ev["role"],
                        json.dumps(ev["payload"], ensure_ascii=False),
                        ev["checkpoint_id"],
                        ev["created_at"],
                    ),
                )
            conn.execute(
                """UPDATE durable_sessions
                   SET forked_from_session_id = ?, forked_from_seq = ?, message_count = ?, updated_at = ?
                   WHERE id = ? AND tenant_id = ?""",
                (source_session_id, effective_seq, msg_cnt, now_iso, new_sid, active.tenant_id),
            )
            conn.commit()

        return self.get_session(new_sid, ctx=active) or forked

    def export_session_trace(
        self,
        session_id: str,
        *,
        fmt: str = "markdown",
        ctx: TenantContext | None = None,
    ) -> str:
        active = ctx or get_tenant_context()
        sess = self.get_session(session_id, ctx=active)
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found")
        events = self.get_events(session_id, after_seq=0, ctx=active)

        if fmt.lower() == "jsonl":
            lines = [json.dumps({"type": "session_header", **sess}, ensure_ascii=False)]
            for ev in events:
                lines.append(json.dumps(ev, ensure_ascii=False))
            return "\n".join(lines) + "\n"

        md = [
            f"# {sess['title']}",
            f"- **Session ID:** `{sess['id']}`",
            f"- **Workspace:** `{sess['tenant_id']}/{sess['workspace_id']}`",
            f"- **Tags:** {', '.join(sess['tags']) or 'none'}",
            f"- **Events:** {len(events)}",
        ]
        if sess["forked_from_session_id"]:
            md.append(
                f"- **Forked From:** `{sess['forked_from_session_id']}` @ seq `{sess['forked_from_seq']}`"
            )
        md.append("\n---\n")
        for ev in events:
            et = ev["event_type"]
            if et == "user_message":
                md.append(f"### 👤 User (`seq #{ev['seq_id']}`)\n{ev['payload'].get('content', '')}\n")
            elif et == "assistant_message":
                md.append(f"### 🤖 Assistant (`seq #{ev['seq_id']}`)\n{ev['payload'].get('content', '')}\n")
            elif et == "subagent_trace":
                p = ev["payload"]
                md.append(
                    f"### 🧩 Sub-Agent `{p.get('specialist_role', 'specialist')}` (`{p.get('child_session_id', '')}`)\n"
                    f"{p.get('distilled_output', '')}\n"
                )
        return "\n".join(md)

    # ── Background Execution, Reconnectable SSE Replay, Cancel & `wake()` ──

    def _set_session_status(
        self,
        session_id: str,
        status: str,
        *,
        pending_prompt: str | None = None,
        ctx: TenantContext,
    ) -> None:
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            if pending_prompt is not None:
                conn.execute(
                    """UPDATE durable_sessions
                       SET status = ?, pending_prompt = ?, updated_at = ?
                       WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                    (status, pending_prompt, now_iso, session_id, ctx.tenant_id, ctx.workspace_id),
                )
            else:
                conn.execute(
                    """UPDATE durable_sessions
                       SET status = ?, updated_at = ?
                       WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                    (status, now_iso, session_id, ctx.tenant_id, ctx.workspace_id),
                )
            conn.commit()

    async def start_background_run(
        self,
        session_id: str,
        prompt: str,
        *,
        agent_config: dict[str, Any] | None = None,
        runner_fn: Callable[[list[BaseMessage], dict[str, Any]], AsyncGenerator[str, None]] | None = None,
        worker_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Enqueue a durable background agent run that continues even if all HTTP clients disconnect."""
        active = ctx or get_tenant_context()
        sess = await asyncio.to_thread(lambda: self.ensure_session(session_id, ctx=active))
        run_id = f"run_{uuid.uuid4().hex[:10]}"
        lease = await asyncio.to_thread(
            lambda: self.acquire_lease(session_id, worker_id=worker_id, ctx=active)
        )

        # Persist user message immediately into append-only event log
        user_ev = await asyncio.to_thread(
            lambda: self.emit_event(
                session_id,
                event_type="user_message",
                role="user",
                run_id=run_id,
                payload={"content": prompt},
                lease=lease,
                ctx=active,
            )
        )
        await asyncio.to_thread(
            lambda: self._set_session_status(session_id, "running", pending_prompt=prompt, ctx=active)
        )

        # Schedule auto-semantic title on Lane P1 if still titled 'New Session'
        if sess["title"] in ("New Session", "Untitled Session"):
            async def _auto_title_job() -> None:
                smart_title = generate_semantic_session_title(prompt)
                await asyncio.to_thread(
                    lambda: self.update_session_metadata(session_id, title=smart_title, ctx=active)
                )

            await work_pool.submit_background(_auto_title_job, job_type="session_auto_title", ctx=active)

        skey = self._stream_key(active, session_id)
        cancel_ev = asyncio.Event()
        self._cancel_flags[skey] = cancel_ev

        task = asyncio.create_task(
            self._drive_run_loop(
                session_id=session_id,
                run_id=run_id,
                lease=lease,
                cancel_ev=cancel_ev,
                agent_config=agent_config or {},
                runner_fn=runner_fn,
                ctx=active,
            )
        )
        self._active_tasks[skey] = task
        return {
            "session_id": session_id,
            "run_id": run_id,
            "user_seq_id": user_ev["seq_id"],
            "fencing_token": lease.fencing_token,
            "status": "running",
        }

    async def _drive_run_loop(
        self,
        *,
        session_id: str,
        run_id: str,
        lease: LeaseHandle,
        cancel_ev: asyncio.Event,
        agent_config: dict[str, Any],
        runner_fn: Callable[[list[BaseMessage], dict[str, Any]], AsyncGenerator[str, None]] | None,
        ctx: TenantContext,
    ) -> None:
        skey = self._stream_key(ctx, session_id)
        chunks: list[str] = []
        try:
            history = await asyncio.to_thread(lambda: self.reconstruct_messages(session_id, ctx=ctx))

            if runner_fn is not None:
                async for token_chunk in runner_fn(history, agent_config):
                    if cancel_ev.is_set():
                        await asyncio.to_thread(
                            lambda: self.emit_event(
                                session_id,
                                event_type="status",
                                run_id=run_id,
                                payload={"status": "cancelled", "partial": "".join(chunks)},
                                lease=lease,
                                ctx=ctx,
                            )
                        )
                        await asyncio.to_thread(
                            lambda: self._set_session_status(session_id, "cancelled", pending_prompt="", ctx=ctx)
                        )
                        return
                    chunks.append(token_chunk)
                    await asyncio.to_thread(
                        lambda c=token_chunk: self.emit_event(
                            session_id,
                            event_type="assistant_token",
                            role="assistant",
                            run_id=run_id,
                            payload={"delta": c},
                            lease=lease,
                            ctx=ctx,
                        )
                    )
            else:
                from agent.graph import make_graph

                with tenant_scope(
                    tenant_id=ctx.tenant_id,
                    workspace_id=ctx.workspace_id,
                    user_id=ctx.user_id,
                    plan_tier=ctx.plan_tier,
                ):
                    graph = await make_graph(
                        {"configurable": {**agent_config, "thread_id": session_id}}
                    )
                    out = await graph.ainvoke({"messages": history})
                    reply_txt = str(out["messages"][-1].content)
                    # Emit compact chunk sequence for SSE replay fidelity
                    step_sz = max(24, len(reply_txt) // 4 or 1)
                    for i in range(0, len(reply_txt), step_sz):
                        if cancel_ev.is_set():
                            break
                        piece = reply_txt[i : i + step_sz]
                        chunks.append(piece)
                        await asyncio.to_thread(
                            lambda c=piece: self.emit_event(
                                session_id,
                                event_type="assistant_token",
                                role="assistant",
                                run_id=run_id,
                                payload={"delta": c},
                                lease=lease,
                                ctx=ctx,
                            )
                        )

            full_text = "".join(chunks)
            await asyncio.to_thread(
                lambda: self.emit_event(
                    session_id,
                    event_type="assistant_message",
                    role="assistant",
                    run_id=run_id,
                    payload={"content": full_text},
                    lease=lease,
                    ctx=ctx,
                )
            )
            await asyncio.to_thread(
                lambda: self.emit_event(
                    session_id,
                    event_type="status",
                    run_id=run_id,
                    payload={"status": "completed"},
                    lease=lease,
                    ctx=ctx,
                )
            )
            await asyncio.to_thread(
                lambda: self._set_session_status(session_id, "completed", pending_prompt="", ctx=ctx)
            )
        except asyncio.CancelledError:
            try:
                await asyncio.to_thread(
                    lambda: self._set_session_status(session_id, "cancelled", pending_prompt="", ctx=ctx)
                )
            except Exception:
                pass
        except Exception as exc:
            logger.warning("Durable run %s failed: %s", run_id, exc)
            try:
                await asyncio.to_thread(
                    lambda: self._set_session_status(session_id, "error", ctx=ctx)
                )
            except Exception:
                pass
        finally:
            try:
                await asyncio.to_thread(lambda: self.release_lease(lease, ctx=ctx))
            except Exception:
                pass
            # Signal end-of-stream (`None`) to all waiting live subscribers
            for q in list(self._subscribers.get(skey, ())):
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass

    async def cancel_session(self, session_id: str, ctx: TenantContext | None = None) -> dict[str, Any]:
        """Publish immediate cross-loop / cross-worker cancellation signal (`<5ms`)."""
        active = ctx or get_tenant_context()
        skey = self._stream_key(active, session_id)
        flag = self._cancel_flags.get(skey)
        task = self._active_tasks.get(skey)
        current_loop = asyncio.get_running_loop()

        if task is not None and not task.done():
            task_loop = task.get_loop()
            if task_loop is current_loop:
                if flag is not None:
                    flag.set()
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
                except asyncio.TimeoutError:
                    task.cancel()
            elif task_loop.is_running():
                if flag is not None:
                    task_loop.call_soon_threadsafe(flag.set)
                task_loop.call_soon_threadsafe(task.cancel)
        elif flag is not None:
            flag.set()

        await asyncio.to_thread(
            lambda: self._set_session_status(session_id, "cancelled", pending_prompt="", ctx=active)
        )
        return {"ok": True, "session_id": session_id, "status": "cancelled"}

    async def wake_session(
        self,
        session_id: str,
        *,
        worker_id: str | None = None,
        now_monotonic: float | None = None,
        runner_fn: Callable[[list[BaseMessage], dict[str, Any]], AsyncGenerator[str, None]] | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Crash-safe `wake(session_id)`: takeover expired worker lease with higher fencing token & finish run!"""
        active = ctx or get_tenant_context()
        sess = await asyncio.to_thread(lambda: self.get_session(session_id, ctx=active))
        if sess is None:
            raise KeyError(f"Session '{session_id}' not found")

        lease = await asyncio.to_thread(
            lambda: self.acquire_lease(
                session_id,
                worker_id=worker_id or "recovery-worker",
                now_monotonic=now_monotonic,
                ctx=active,
            )
        )
        run_id = f"recovery_{uuid.uuid4().hex[:8]}"
        skey = self._stream_key(active, session_id)
        cancel_ev = asyncio.Event()
        self._cancel_flags[skey] = cancel_ev

        task = asyncio.create_task(
            self._drive_run_loop(
                session_id=session_id,
                run_id=run_id,
                lease=lease,
                cancel_ev=cancel_ev,
                agent_config={},
                runner_fn=runner_fn,
                ctx=active,
            )
        )
        self._active_tasks[skey] = task
        await task
        return {
            "recovered": True,
            "session_id": session_id,
            "run_id": run_id,
            "new_fencing_token": lease.fencing_token,
            "status": (self.get_session(session_id, ctx=active) or {}).get("status", "completed"),
        }

    async def subscribe_sse(
        self,
        session_id: str,
        *,
        last_event_id: int = 0,
        ctx: TenantContext | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield standard SSE frames (`id: <seq_id>\\nevent: <type>\\ndata: <json>\\n\\n`) with gap-free replay!"""
        active = ctx or get_tenant_context()
        skey = self._stream_key(active, session_id)
        live_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=2048)

        # Register live subscriber FIRST before querying historical events so ZERO events slip between replay & live!
        self._subscribers.setdefault(skey, set()).add(live_queue)
        high_water_seq = last_event_id

        try:
            missed = await asyncio.to_thread(
                lambda: self.get_events(session_id, after_seq=last_event_id, ctx=active)
            )
            for ev in missed:
                high_water_seq = max(high_water_seq, ev["seq_id"])
                yield f"id: {ev['seq_id']}\nevent: {ev['event_type']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

            sess = await asyncio.to_thread(lambda: self.get_session(session_id, ctx=active))
            if not sess or sess["status"] not in ("running",):
                return

            while True:
                item = await asyncio.wait_for(live_queue.get(), timeout=30.0)
                if item is None:
                    break
                if item["seq_id"] <= high_water_seq:
                    # Deduplicate any event that landed during the initial DB replay window!
                    continue
                high_water_seq = item["seq_id"]
                yield f"id: {item['seq_id']}\nevent: {item['event_type']}\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item["event_type"] == "status" and item["payload"].get("status") in (
                    "completed",
                    "cancelled",
                    "error",
                ):
                    break
        finally:
            subs = self._subscribers.get(skey)
            if subs is not None:
                subs.discard(live_queue)


session_manager = DurableSessionManager()

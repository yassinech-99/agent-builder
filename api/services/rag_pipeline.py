"""Phase 7 — Production Multi-Tenant Contextual, Parent-Child & Hybrid Agentic RAG Pipeline (`api/services/rag_pipeline.py`).

Implements the end-to-end enterprise retrieval architecture specified in `plan.md` Phase 7:
  1. **Streaming `O(1)` Upload + SHA-256 Content Deduplication (`rag_documents`):**
     - Streams uploaded bytes in `64 KiB` chunks directly to staging / RustFS object storage without
       buffering large files in event-loop RAM, skipping unchanged files instantaneously (`deduplicated=True`).
  2. **Async `202 Accepted` Ingestion on Fair-Queued Priority Lane `P1` + Live SSE Progress:**
     - Decouples heavy parsing, contextual enrichment, and embedding completely from interactive chat (`P0`),
       streaming per-stage progress (`queued` -> `chunking` -> `contextualizing` -> `indexing` -> `ready`).
  3. **Structure-Aware Parent-Child Chunking (`rag_parent_chunks` + `rag_child_chunks`):**
     - Preserves full Markdown heading breadcrumbs (`Doc > H1 > H2 > H3`) AND keeps Markdown tables intact
       inside atomic blocks so column headers are NEVER severed from rows!
     - Stores full rich `~1,400`-char parent sections linked `1:N` to focused `~380`-char searchable children.
  4. **Anthropic-Style Contextual Retrieval Enrichment (`contextual_enrich=True`):**
     - Prepends document thesis + hierarchical breadcrumb + key entity summary (`context_prefix`) to every
       child chunk prior to BOTH Dense Vector Indexing AND Contextual BM25 Indexing!
  5. **6-Stage Hybrid + Cross-Encoder Reranked + Parent-Deduplicated + Self-Grading Agentic Query Engine:**
     - Parallel **Qdrant `is_tenant=True` (`INT8`) Dense Search** + **Contextual BM25 Search**
       -> **Weighted Reciprocal Rank Fusion ($k = 60$)**
       -> **Batched `NVIDIARerank` (`nvidia/llama-nemotron-rerank-1b-v2`) / Precision Cross-Scorer**
       -> **Parent-Section Expansion & Sibling Deduplication (`1` citation per unique `parent_id`)**
       -> **Self-Grading Relevance Check & Automatic Query Rewrite Fallback**
       -> **Grounded `[1]`, `[2]` Source Evidence Citations**!
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any, AsyncGenerator, Awaitable, Callable
import uuid

from fastapi import UploadFile
from langchain_core.documents import Document

from agent.memory_engine import LocalHashedSemanticEmbedder
from agent.registry import count_text_tokens
from api.core.object_store import build_object_store_client_for_bucket
from api.core.tenancy import (
    TenantContext,
    get_tenant_context,
    scoped_object_prefix,
)
from api.core.vector_store import vector_store as shared_vector_store
from api.db import get_db
from api.services.batcher import gpu_batcher
from api.utils.doc_loader import MAX_FILE_SIZE, load_documents
from api.workers.priority_queue import work_pool

logger = logging.getLogger("nimagents.rag_pipeline")

RAG_VECTOR_NAMESPACE = "contextual_rag"
STREAM_CHUNK_BYTES = 65_536

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_WORD_RE = re.compile(r"[a-z0-9_.-]+")
_STOPWORDS = frozenset(
    {
        "what",
        "which",
        "where",
        "when",
        "how",
        "does",
        "can",
        "you",
        "please",
        "tell",
        "about",
        "the",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "our",
        "are",
        "was",
        "were",
        "have",
        "has",
        "explain",
        "describe",
        "give",
        "find",
    }
)

_LOCAL_RAG_EMBEDDER = LocalHashedSemanticEmbedder()
_NVIDIA_RAG_EMBEDDER_CACHE: dict[str, Any] = {}


def _tokenize_terms(text: str) -> list[str]:
    return [tok for tok in _WORD_RE.findall(text.lower()) if len(tok) >= 2]


def _resolve_rag_embedder(model_name: str = "nvidia/llama-nemotron-embed-1b-v2") -> tuple[str, Any]:
    embed_url = os.getenv("NVIDIA_EMBED_BASE_URL", "").strip()
    use_remote = bool(embed_url) or os.getenv("NIMAGENTS_RAG_REMOTE_EMBED", "").strip() == "1"
    if use_remote:
        cached = _NVIDIA_RAG_EMBEDDER_CACHE.get(model_name)
        if cached is None:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

            kwargs: dict[str, Any] = {"model": model_name, "truncate": "END"}
            if embed_url:
                kwargs["base_url"] = embed_url
            api_key = os.getenv("NVIDIA_API_KEY", "").strip()
            if api_key:
                kwargs["api_key"] = api_key
            cached = NVIDIAEmbeddings(**kwargs)
            _NVIDIA_RAG_EMBEDDER_CACHE[model_name] = cached
        return (model_name, cached)
    return ("local-contextual-256", _LOCAL_RAG_EMBEDDER)


async def stream_upload_with_sha256(
    upload: UploadFile,
    dest_path: Path,
    max_bytes: int = MAX_FILE_SIZE,
) -> tuple[int, str]:
    """Stream `UploadFile` to `dest_path` in `64 KiB` chunks computing `SHA-256` in `O(1)` memory."""
    hasher = hashlib.sha256()
    total = 0
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with dest_path.open("wb") as handle:
        while True:
            chunk = await upload.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Uploaded file exceeds maximum allowed size ({max_bytes} bytes)")
            hasher.update(chunk)
            await asyncio.to_thread(handle.write, chunk)
    return total, hasher.hexdigest()


# ── Structure-Aware Markdown / Table-Preserving Parent-Child Chunker ───────


def _split_blocks_preserving_tables(body: str) -> list[str]:
    """Split section text into logical paragraph / intact-table / code-fence blocks."""
    lines = body.splitlines()
    blocks: list[str] = []
    cur: list[str] = []
    in_fence = False
    in_table = False

    def _flush() -> None:
        nonlocal cur
        joined = "\n".join(cur).strip()
        if joined:
            blocks.append(joined)
        cur = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                cur.append(line)
                _flush()
                in_fence = False
            else:
                _flush()
                in_fence = True
                cur.append(line)
            continue

        if in_fence:
            cur.append(line)
            continue

        looks_like_table_row = stripped.startswith("|") and stripped.endswith("|") and "|" in stripped[1:-1]
        if looks_like_table_row:
            if not in_table:
                _flush()
                in_table = True
            cur.append(line)
            continue
        elif in_table:
            _flush()
            in_table = False

        if not stripped:
            _flush()
        else:
            cur.append(line)

    _flush()
    return blocks


def split_structured_parent_child(
    raw_text: str,
    *,
    doc_name: str,
    parent_max_chars: int = 1400,
    child_max_chars: int = 400,
) -> list[dict[str, Any]]:
    """Parse structured text into hierarchical Parent Sections + `1:N` atomic Child Chunks.

    Guarantees:
      - Heading breadcrumbs (`{doc_name} > H1 > H2 > H3`) attach to EVERY parent AND child chunk.
      - Markdown tables (`| col | val |`) NEVER split mid-table if under `parent_max_chars`
        (and if split across children, preserve table header + separator rows automatically!).
    """
    clean_doc = Path(doc_name).stem.replace("_", " ").replace("-", " ").strip() or doc_name
    heading_stack: dict[int, str] = {}
    raw_sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []
    current_path = clean_doc

    for line in raw_text.splitlines():
        hm = _HEADING_RE.match(line)
        if hm:
            if current_lines:
                raw_sections.append((current_path, current_lines))
                current_lines = []
            lvl = len(hm.group(1))
            title = hm.group(2).strip()
            for deeper in [k for k in heading_stack if k >= lvl]:
                heading_stack.pop(deeper, None)
            heading_stack[lvl] = title
            ordered = [heading_stack[k] for k in sorted(heading_stack)]
            current_path = " > ".join([clean_doc, *ordered])
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        raw_sections.append((current_path, current_lines))

    parents_out: list[dict[str, Any]] = []
    for heading_path, sec_lines in raw_sections:
        sec_text = "\n".join(sec_lines).strip()
        if not sec_text:
            continue

        blocks = _split_blocks_preserving_tables(sec_text)
        # Pack blocks into parent windows (`~parent_max_chars`) without breaking tables
        parent_windows: list[list[str]] = []
        cur_p: list[str] = []
        cur_len = 0
        for blk in blocks:
            if cur_p and (cur_len + len(blk) + 2 > parent_max_chars):
                parent_windows.append(cur_p)
                cur_p = [blk]
                cur_len = len(blk)
            else:
                cur_p.append(blk)
                cur_len += len(blk) + 2
        if cur_p:
            parent_windows.append(cur_p)

        for p_blocks in parent_windows:
            parent_md = "\n\n".join(p_blocks).strip()
            children_texts: list[str] = []
            c_acc: list[str] = []
            c_len = 0

            for blk in p_blocks:
                is_table = blk.lstrip().startswith("|") and "\n|" in blk
                if is_table:
                    # Keep entire Markdown table atomic as a dedicated child chunk!
                    if c_acc:
                        children_texts.append("\n\n".join(c_acc))
                        c_acc = []
                        c_len = 0
                    children_texts.append(blk)
                    continue

                if len(blk) <= child_max_chars:
                    if c_acc and (c_len + len(blk) + 2 > child_max_chars):
                        children_texts.append("\n\n".join(c_acc))
                        c_acc = [blk]
                        c_len = len(blk)
                    else:
                        c_acc.append(blk)
                        c_len += len(blk) + 2
                else:
                    if c_acc:
                        children_texts.append("\n\n".join(c_acc))
                        c_acc = []
                        c_len = 0
                    # Sentence-boundary split for long prose paragraphs
                    sentences = re.split(r"(?<=[.!?])\s+", blk)
                    s_acc: list[str] = []
                    s_len = 0
                    for sent in sentences:
                        if s_acc and (s_len + len(sent) + 1 > child_max_chars):
                            children_texts.append(" ".join(s_acc))
                            s_acc = [sent]
                            s_len = len(sent)
                        else:
                            s_acc.append(sent)
                            s_len += len(sent) + 1
                    if s_acc:
                        children_texts.append(" ".join(s_acc))

            if c_acc:
                children_texts.append("\n\n".join(c_acc))

            if children_texts:
                parents_out.append(
                    {
                        "heading_path": heading_path,
                        "parent_markdown": parent_md,
                        "children": children_texts,
                    }
                )

    return parents_out


def synthesize_contextual_prefix(
    *,
    doc_name: str,
    doc_overview: str,
    heading_path: str,
    parent_markdown: str,
    child_text: str,
) -> str:
    """Build situating contextual header per Anthropic's Contextual Retrieval architecture."""
    key_terms: list[str] = []
    seen: set[str] = set()
    # Capture capitalized technical nouns, numbers/units, and table headers from parent section
    for m in re.finditer(r"\b(?:[A-Z][A-Za-z0-9_.-]{1,}|[0-9]+(?:\.[0-9]+)?%?(?:ms|GB|TB|TFLOPS|x)?)\b", parent_markdown):
        tok = m.group(0)
        low = tok.lower()
        if low not in seen and len(tok) >= 2:
            seen.add(low)
            key_terms.append(tok)
        if len(key_terms) >= 8:
            break

    entities_clause = f" | Key Entities: {', '.join(key_terms)}" if key_terms else ""
    overview_clause = f" | Document Summary: {doc_overview}" if doc_overview else ""
    return f"[Document: {doc_name} | Section: {heading_path}{overview_clause}{entities_clause}]"


def _contextual_bm25_score(query_tokens: list[str], doc_text: str) -> float:
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize_terms(doc_text)
    if not doc_tokens:
        return 0.0
    freqs: dict[str, int] = {}
    for t in doc_tokens:
        freqs[t] = freqs.get(t, 0) + 1
    k1 = 1.5
    b = 0.75
    avg_dl = 65.0
    dl = float(len(doc_tokens))
    total = 0.0
    for q in set(query_tokens):
        tf = float(freqs.get(q, 0))
        if tf <= 0.0:
            continue
        # Boost rarer/longer technical tokens (acronyms, model IDs, metrics)
        idf_proxy = 1.0 + min(1.5, len(q) / 6.0)
        total += idf_proxy * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avg_dl)))
    return total


def _cross_encoder_precision_score(query: str, candidate_text: str) -> float:
    """High-precision query-document interaction scorer (phrase proximity + bigram + coverage).

    Used alongside or as zero-latency fallback when remote `NVIDIARerank` is not configured.
    """
    q_toks = [t for t in _tokenize_terms(query) if t not in _STOPWORDS] or _tokenize_terms(query)
    if not q_toks:
        return 0.0
    low_doc = candidate_text.lower()
    doc_tok_set = set(_tokenize_terms(candidate_text))

    # 1. Exact query term coverage ratio (`0..1`)
    matched = sum(1 for t in set(q_toks) if t in doc_tok_set)
    coverage = matched / max(1, len(set(q_toks)))

    # 2. Ordered adjacent phrase bonus (`q_i q_{i+1}`)
    phrase_bonus = 0.0
    for i in range(len(q_toks) - 1):
        bigram = f"{q_toks[i]} {q_toks[i + 1]}"
        if bigram in low_doc:
            phrase_bonus += 0.18

    # 3. Exact numeric/unit match bonus
    nums = [t for t in q_toks if any(ch.isdigit() for ch in t)]
    num_bonus = 0.15 if (nums and all(n in low_doc for n in nums)) else 0.0

    return round(min(1.0, 0.65 * coverage + min(0.25, phrase_bonus) + num_bonus), 5)


def rewrite_query_for_agentic_retry(original_query: str) -> str:
    """Deterministic high-recall query expansion stripping conversational fluff & expanding acronyms."""
    expansions = {
        "ttft": "time to first token latency ttft",
        "kv": "kv cache prefix memory",
        "sla": "sla latency guarantee throughput",
        "oom": "out of memory ceiling limit",
        "rag": "retrieval augmented generation chunk rerank",
        "rls": "row level security tenant isolation",
    }
    raw_tokens = _tokenize_terms(original_query)
    core = [t for t in raw_tokens if t not in _STOPWORDS]
    expanded: list[str] = []
    for tok in core:
        expanded.append(tok)
        if tok in expansions:
            expanded.append(expansions[tok])
    result = " ".join(expanded).strip()
    return result or original_query.strip()


class EnterpriseRagPipeline:
    """Production Multi-Tenant Contextual + Parent-Child + Hybrid + Reranked Agentic RAG Manager."""

    def __init__(self) -> None:
        self._db_mutex = threading.RLock()
        self._job_subscribers: dict[str, set[asyncio.Queue[dict[str, Any] | None]]] = {}
        self._inflight_tasks: set[asyncio.Task[Any]] = set()

    # ── Job Progress & Live SSE Fan-Out (`202 Accepted` Progress Stream) ───

    def _emit_job_progress(
        self,
        job_id: str,
        *,
        status: str,
        progress: int,
        detail: str,
        ctx: TenantContext,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now_iso = datetime.now(tz=UTC).isoformat()
        payload = {
            "job_id": job_id,
            "tenant_id": ctx.tenant_id,
            "workspace_id": ctx.workspace_id,
            "status": status,
            "progress": progress,
            "detail": detail,
            "updated_at": now_iso,
            **(extra or {}),
        }
        with self._db_mutex, get_db() as conn:
            conn.execute(
                """INSERT INTO background_jobs
                   (id, tenant_id, workspace_id, lane, job_type, status, progress, detail, created_at, updated_at)
                   VALUES (?, ?, ?, 'background', 'contextual_rag_ingest', ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                     status = excluded.status,
                     progress = excluded.progress,
                     detail = excluded.detail,
                     updated_at = excluded.updated_at""",
                (job_id, ctx.tenant_id, ctx.workspace_id, status, progress, detail, now_iso, now_iso),
            )
            conn.commit()

        skey = f"{ctx.tenant_id}:{ctx.workspace_id}:{job_id}"
        for q in list(self._job_subscribers.get(skey, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass
            if status in ("ready", "completed", "error"):
                try:
                    q.put_nowait(None)
                except asyncio.QueueFull:
                    pass
        return payload

    def get_job(self, job_id: str, *, ctx: TenantContext | None = None) -> dict[str, Any] | None:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            row = conn.execute(
                """SELECT * FROM background_jobs
                   WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                (job_id, active.tenant_id, active.workspace_id),
            ).fetchone()
        return dict(row) if row else None

    async def subscribe_job_sse(
        self,
        job_id: str,
        *,
        ctx: TenantContext | None = None,
    ) -> AsyncGenerator[str, None]:
        active = ctx or get_tenant_context()
        skey = f"{active.tenant_id}:{active.workspace_id}:{job_id}"
        q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=128)
        self._job_subscribers.setdefault(skey, set()).add(q)
        try:
            snap = await asyncio.to_thread(lambda: self.get_job(job_id, ctx=active))
            if snap is not None:
                yield f"event: progress\ndata: {json.dumps(snap, ensure_ascii=False)}\n\n"
                if snap["status"] in ("ready", "completed", "error"):
                    return
            while True:
                item = await q.get()
                if item is None:
                    return
                yield f"event: progress\ndata: {json.dumps(item, ensure_ascii=False)}\n\n"
                if item.get("status") in ("ready", "completed", "error"):
                    return
        finally:
            subs = self._job_subscribers.get(skey)
            if subs is not None:
                subs.discard(q)

    # ── Collection Management ─────────────────────────────────────────────

    def create_collection(
        self,
        *,
        name: str,
        embedding_model: str = "nvidia/llama-nemotron-embed-1b-v2",
        storage_bucket: str = "",
        storage_prefix: str = "",
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        cid = f"crag_{uuid.uuid4().hex[:10]}"
        now_iso = datetime.now(tz=UTC).isoformat()
        with self._db_mutex, get_db() as conn:
            conn.execute(
                """INSERT INTO local_rag_sources
                   (id, tenant_id, workspace_id, name, source_type, status,
                    doc_count, chunk_size, chunk_overlap, embedding_model,
                    storage_bucket, storage_prefix, vector_dim, created_at)
                   VALUES (?, ?, ?, ?, 'contextual_hybrid', 'ready', 0, 1400, 150, ?, ?, ?, 256, ?)""",
                (
                    cid,
                    active.tenant_id,
                    active.workspace_id,
                    name.strip(),
                    embedding_model,
                    storage_bucket,
                    storage_prefix,
                    now_iso,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM local_rag_sources WHERE id = ?", (cid,)).fetchone()
        return {**dict(row), "documents": []}

    def list_collections(self, *, ctx: TenantContext | None = None) -> list[dict[str, Any]]:
        active = ctx or get_tenant_context()
        with self._db_mutex, get_db() as conn:
            sources = conn.execute(
                """SELECT * FROM local_rag_sources
                   WHERE tenant_id = ? AND workspace_id = ?
                   ORDER BY created_at DESC""",
                (active.tenant_id, active.workspace_id),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for s in sources:
                docs = conn.execute(
                    """SELECT * FROM rag_documents
                       WHERE source_id = ? AND tenant_id = ? AND workspace_id = ?
                       ORDER BY created_at DESC""",
                    (s["id"], active.tenant_id, active.workspace_id),
                ).fetchall()
                out.append(
                    {
                        **dict(s),
                        "documents": [
                            {**dict(d), "contextual_enriched": bool(d["contextual_enriched"])}
                            for d in docs
                        ],
                    }
                )
        return out

    async def delete_collection(self, source_id: str, *, ctx: TenantContext | None = None) -> bool:
        active = ctx or get_tenant_context()

        def _tx_del() -> bool:
            with self._db_mutex, get_db() as conn:
                conn.execute(
                    "DELETE FROM rag_child_chunks WHERE source_id = ? AND tenant_id = ? AND workspace_id = ?",
                    (source_id, active.tenant_id, active.workspace_id),
                )
                conn.execute(
                    "DELETE FROM rag_parent_chunks WHERE source_id = ? AND tenant_id = ? AND workspace_id = ?",
                    (source_id, active.tenant_id, active.workspace_id),
                )
                conn.execute(
                    "DELETE FROM rag_documents WHERE source_id = ? AND tenant_id = ? AND workspace_id = ?",
                    (source_id, active.tenant_id, active.workspace_id),
                )
                cur = conn.execute(
                    "DELETE FROM local_rag_sources WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
                    (source_id, active.tenant_id, active.workspace_id),
                )
                conn.commit()
                return cur.rowcount > 0

        removed = await asyncio.to_thread(_tx_del)
        if removed:
            try:
                await shared_vector_store.delete_source(
                    source_id=source_id, namespace=RAG_VECTOR_NAMESPACE, ctx=active
                )
            except Exception:
                pass
        return removed

    # ── Core Ingestion Pipeline (SHA-256 Dedup + Parent-Child + Contextual + Dual Index) ──

    async def ingest_staged_file(
        self,
        *,
        source_id: str,
        file_path: Path,
        filename: str,
        sha256_hash: str,
        byte_size: int,
        contextual_enrich: bool = True,
        enricher_fn: Callable[[str, str], Awaitable[str]] | None = None,
        job_id: str | None = None,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Run full Parent-Child + Contextual Enrichment + Qdrant + BM25 indexing on Priority Lane `P1`."""
        active = ctx or get_tenant_context()
        jid = job_id or f"ragingest_{uuid.uuid4().hex[:10]}"

        async def _lane_p1_pipeline() -> dict[str, Any]:
            # Step 1: Check SHA-256 deduplication in `(tenant_id, workspace_id, source_id)`
            def _check_existing() -> tuple[dict[str, Any] | None, dict[str, Any]]:
                with self._db_mutex, get_db() as conn:
                    src = conn.execute(
                        """SELECT * FROM local_rag_sources
                           WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                        (source_id, active.tenant_id, active.workspace_id),
                    ).fetchone()
                    if src is None:
                        raise KeyError(f"RAG collection '{source_id}' not found")
                    dup = conn.execute(
                        """SELECT * FROM rag_documents
                           WHERE source_id = ? AND tenant_id = ? AND workspace_id = ? AND sha256_hash = ?""",
                        (source_id, active.tenant_id, active.workspace_id, sha256_hash),
                    ).fetchone()
                    return (dict(dup) if dup else None, dict(src))

            dup_row, src_row = await asyncio.to_thread(_check_existing)
            if dup_row is not None:
                await asyncio.to_thread(
                    lambda: self._emit_job_progress(
                        jid,
                        status="completed",
                        progress=100,
                        detail=f"Skipped unchanged document '{filename}' (SHA-256 match)",
                        ctx=active,
                    )
                )
                return {
                    "job_id": jid,
                    "doc_id": dup_row["id"],
                    "source_id": source_id,
                    "filename": filename,
                    "sha256_hash": sha256_hash,
                    "deduplicated": True,
                    "parent_count": dup_row["parent_count"],
                    "child_count": dup_row["child_count"],
                }

            # Optional RustFS archival of raw file under tenant object prefix
            storage_uri = f"local://{active.tenant_id}/{active.workspace_id}/{source_id}/{filename}"
            bucket_name = (src_row.get("storage_bucket") or "").strip()
            if bucket_name:
                def _archive_object_store() -> str:
                    s_client = build_object_store_client_for_bucket(bucket_name, ctx=active)
                    try:
                        prefix_clean = scoped_object_prefix(
                            src_row.get("storage_prefix") or "rag",
                            active,
                        )
                        obj_key = f"{prefix_clean}{filename}"
                        raw_bytes = file_path.read_bytes()
                        s_client.put_object(bucket_name, obj_key, io.BytesIO(raw_bytes), len(raw_bytes))
                        return f"s3://{bucket_name}/{obj_key}"
                    finally:
                        s_client.close()

                try:
                    storage_uri = await asyncio.to_thread(_archive_object_store)
                except Exception as s_exc:
                    logger.warning("RustFS archival skipped cleanly (%s)", s_exc)

            # Step 2: Load & Structure-Aware Parent-Child Chunking (`25%`)
            await asyncio.to_thread(
                lambda: self._emit_job_progress(
                    jid,
                    status="running",
                    progress=25,
                    detail=f"Parsing & table-preserving Parent-Child chunking for '{filename}'",
                    ctx=active,
                )
            )
            ext = Path(filename).suffix.lower()
            if ext in (".md", ".markdown", ".txt", ".csv", ".json"):
                full_text = await asyncio.to_thread(
                    lambda: file_path.read_text(encoding="utf-8", errors="replace")
                )
            else:
                loaded_docs = await asyncio.to_thread(load_documents, file_path)
                full_text = "\n\n".join(d.page_content for d in loaded_docs)

            structured_parents = await asyncio.to_thread(
                lambda: split_structured_parent_child(full_text, doc_name=filename)
            )
            if not structured_parents:
                raise ValueError(f"No indexable text extracted from '{filename}'")

            first_nonempty_line = next(
                (ln.strip("# ").strip() for ln in full_text.splitlines() if ln.strip()),
                filename,
            )[:120]

            # Step 3: Contextual Retrieval Enrichment (`55%`)
            await asyncio.to_thread(
                lambda: self._emit_job_progress(
                    jid,
                    status="running",
                    progress=55,
                    detail=f"Synthesizing contextual chunk headers across {len(structured_parents)} section(s)",
                    ctx=active,
                )
            )

            doc_id = f"rdoc_{uuid.uuid4().hex[:10]}"
            parent_records: list[dict[str, Any]] = []
            child_records: list[dict[str, Any]] = []

            for p_idx, p_item in enumerate(structured_parents):
                pid = f"pchk_{uuid.uuid4().hex[:12]}"
                parent_md = p_item["parent_markdown"]
                h_path = p_item["heading_path"]
                parent_records.append(
                    {
                        "id": pid,
                        "section_index": p_idx,
                        "heading_path": h_path,
                        "raw_markdown": parent_md,
                        "token_count": count_text_tokens(parent_md),
                    }
                )
                for c_idx, c_raw in enumerate(p_item["children"]):
                    cid_row = f"cchk_{uuid.uuid4().hex[:12]}"
                    if contextual_enrich:
                        if enricher_fn is not None:
                            custom_ctx = await enricher_fn(parent_md, c_raw)
                            ctx_prefix = f"[Document: {filename} | Section: {h_path} | Context: {custom_ctx.strip()}]"
                        else:
                            ctx_prefix = synthesize_contextual_prefix(
                                doc_name=filename,
                                doc_overview=first_nonempty_line,
                                heading_path=h_path,
                                parent_markdown=parent_md,
                                child_text=c_raw,
                            )
                        enriched = f"{ctx_prefix}\n{c_raw}"
                    else:
                        ctx_prefix = f"[Document: {filename} | Section: {h_path}]"
                        enriched = c_raw

                    child_records.append(
                        {
                            "id": cid_row,
                            "parent_id": pid,
                            "heading_path": h_path,
                            "child_index": c_idx,
                            "raw_text": c_raw,
                            "context_prefix": ctx_prefix,
                            "enriched_text": enriched,
                            "token_count": count_text_tokens(enriched),
                        }
                    )

            # Step 4: Dual Multi-Tenant Indexing — Batched Qdrant (`is_tenant=True` + `INT8`) + Contextual BM25 (`85%`)
            await asyncio.to_thread(
                lambda: self._emit_job_progress(
                    jid,
                    status="running",
                    progress=85,
                    detail=f"Micro-batch embedding & indexing {len(child_records)} contextual child chunks",
                    ctx=active,
                )
            )

            emb_key, embedder = _resolve_rag_embedder(
                src_row.get("embedding_model") or "nvidia/llama-nemotron-embed-1b-v2"
            )
            enriched_texts = [c["enriched_text"] for c in child_records]
            vectors = await gpu_batcher.embed_documents(
                emb_key, enriched_texts, embedder.embed_documents
            )

            await shared_vector_store.upsert_chunks(
                source_id=source_id,
                vectors=vectors,
                texts=enriched_texts,
                metadatas=[
                    {
                        "child_id": c["id"],
                        "parent_id": c["parent_id"],
                        "doc_id": doc_id,
                        "doc_name": filename,
                        "heading_path": c["heading_path"],
                    }
                    for c in child_records
                ],
                namespace=RAG_VECTOR_NAMESPACE,
                ctx=active,
            )

            now_iso = datetime.now(tz=UTC).isoformat()

            def _persist_relational() -> None:
                with self._db_mutex, get_db() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        """INSERT INTO rag_documents
                           (id, tenant_id, workspace_id, source_id, filename, sha256_hash,
                            content_type, byte_size, parent_count, child_count,
                            contextual_enriched, storage_uri, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)""",
                        (
                            doc_id,
                            active.tenant_id,
                            active.workspace_id,
                            source_id,
                            filename,
                            sha256_hash,
                            ext or "text/markdown",
                            byte_size,
                            len(parent_records),
                            len(child_records),
                            int(bool(contextual_enrich)),
                            storage_uri,
                            now_iso,
                        ),
                    )
                    for p in parent_records:
                        conn.execute(
                            """INSERT INTO rag_parent_chunks
                               (id, tenant_id, workspace_id, source_id, doc_id, doc_name,
                                section_index, heading_path, raw_markdown, token_count, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                p["id"],
                                active.tenant_id,
                                active.workspace_id,
                                source_id,
                                doc_id,
                                filename,
                                p["section_index"],
                                p["heading_path"],
                                p["raw_markdown"],
                                p["token_count"],
                                now_iso,
                            ),
                        )
                    for c in child_records:
                        conn.execute(
                            """INSERT INTO rag_child_chunks
                               (id, tenant_id, workspace_id, source_id, doc_id, parent_id,
                                doc_name, heading_path, child_index, raw_text, context_prefix,
                                enriched_text, token_count, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                c["id"],
                                active.tenant_id,
                                active.workspace_id,
                                source_id,
                                doc_id,
                                c["parent_id"],
                                filename,
                                c["heading_path"],
                                c["child_index"],
                                c["raw_text"],
                                c["context_prefix"],
                                c["enriched_text"],
                                c["token_count"],
                                now_iso,
                            ),
                        )
                    conn.execute(
                        """UPDATE local_rag_sources
                           SET status = 'ready', doc_count = doc_count + ?
                           WHERE id = ? AND tenant_id = ? AND workspace_id = ?""",
                        (len(child_records), source_id, active.tenant_id, active.workspace_id),
                    )
                    conn.commit()

            await asyncio.to_thread(_persist_relational)
            await asyncio.to_thread(
                lambda: self._emit_job_progress(
                    jid,
                    status="completed",
                    progress=100,
                    detail=f"Indexed '{filename}': {len(parent_records)} parent section(s), {len(child_records)} contextual child chunk(s)",
                    ctx=active,
                )
            )
            return {
                "job_id": jid,
                "doc_id": doc_id,
                "source_id": source_id,
                "filename": filename,
                "sha256_hash": sha256_hash,
                "deduplicated": False,
                "parent_count": len(parent_records),
                "child_count": len(child_records),
                "storage_uri": storage_uri,
            }

        return await work_pool.submit_background(
            _lane_p1_pipeline,
            job_type="contextual_rag_ingest",
            ctx=active,
        )

    def enqueue_staged_file_ingestion(
        self,
        *,
        source_id: str,
        staged_dir: Path,
        file_path: Path,
        filename: str,
        sha256_hash: str,
        byte_size: int,
        contextual_enrich: bool = True,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Return `202 Accepted` immediately while Lane P1 executes full ingestion & cleans up staging dir."""
        active = ctx or get_tenant_context()
        jid = f"ragingest_{uuid.uuid4().hex[:10]}"
        self._emit_job_progress(
            jid,
            status="queued",
            progress=5,
            detail=f"Queued '{filename}' ({byte_size:,} bytes) on Priority Lane P1",
            ctx=active,
        )

        async def _bg() -> None:
            try:
                await self.ingest_staged_file(
                    source_id=source_id,
                    file_path=file_path,
                    filename=filename,
                    sha256_hash=sha256_hash,
                    byte_size=byte_size,
                    contextual_enrich=contextual_enrich,
                    job_id=jid,
                    ctx=active,
                )
            except Exception as exc:
                logger.warning("Background RAG ingestion %s failed: %s", jid, exc)
                await asyncio.to_thread(
                    lambda: self._emit_job_progress(
                        jid,
                        status="error",
                        progress=100,
                        detail=str(exc),
                        ctx=active,
                    )
                )
            finally:
                import shutil

                await asyncio.to_thread(shutil.rmtree, staged_dir, True)

        task = asyncio.create_task(_bg())
        self._inflight_tasks.add(task)
        task.add_done_callback(self._inflight_tasks.discard)
        return {
            "job_id": jid,
            "source_id": source_id,
            "filename": filename,
            "sha256_hash": sha256_hash,
            "status": "queued",
            "progress_stream_url": f"/api/rag-pipeline/jobs/{jid}/stream",
        }

    async def flush_inflight(self) -> None:
        if self._inflight_tasks:
            await asyncio.gather(*list(self._inflight_tasks), return_exceptions=True)

    # ── 6-Stage Hybrid + Reranked + Parent-Deduplicated + Self-Grading Query Engine ──

    async def _single_pass_retrieve(
        self,
        query: str,
        *,
        source_ids: list[str] | None,
        top_k: int,
        rerank: bool,
        ctx: TenantContext,
    ) -> tuple[list[dict[str, Any]], float]:
        q_tokens = _tokenize_terms(query)
        emb_key, embedder = _resolve_rag_embedder()

        def _load_candidates() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
            sql_c = """SELECT * FROM rag_child_chunks
                       WHERE tenant_id = ? AND workspace_id = ?"""
            sql_p = """SELECT * FROM rag_parent_chunks
                       WHERE tenant_id = ? AND workspace_id = ?"""
            params: list[Any] = [ctx.tenant_id, ctx.workspace_id]
            if source_ids:
                placeholders = ",".join("?" for _ in source_ids)
                sql_c += f" AND source_id IN ({placeholders})"
                sql_p += f" AND source_id IN ({placeholders})"
                params.extend(source_ids)
            with self._db_mutex, get_db() as conn:
                c_rows = conn.execute(sql_c, params).fetchall()
                p_rows = conn.execute(sql_p, params).fetchall()
            return (
                {r["id"]: dict(r) for r in c_rows},
                {r["id"]: dict(r) for r in p_rows},
            )

        qvec_task = asyncio.create_task(
            gpu_batcher.embed_text(emb_key, query, embedder.embed_documents)
        )
        rel_task = asyncio.to_thread(_load_candidates)
        query_vec, (children_by_id, parents_by_id) = await asyncio.gather(qvec_task, rel_task)

        if not children_by_id:
            return ([], 0.0)

        # Stage 1A: Qdrant `is_tenant=True` Dense Vector Retrieval
        dense_hits: list[dict[str, Any]] = []
        if source_ids and len(source_ids) == 1:
            dense_hits = await shared_vector_store.search(
                source_id=source_ids[0],
                query_vector=query_vec,
                k=max(top_k * 5, 24),
                namespace=RAG_VECTOR_NAMESPACE,
                ctx=ctx,
            )
        else:
            raw_dense = await shared_vector_store.search(
                source_id=None,
                query_vector=query_vec,
                k=max(top_k * 6, 32),
                namespace=RAG_VECTOR_NAMESPACE,
                ctx=ctx,
            )
            allowed = set(source_ids) if source_ids else None
            dense_hits = [
                h
                for h in raw_dense
                if allowed is None or (h.get("metadata") or {}).get("child_id") in children_by_id
            ]

        dense_rank_map: dict[str, int] = {}
        dense_score_map: dict[str, float] = {}
        for idx, hit in enumerate(dense_hits):
            cid = str((hit.get("metadata") or {}).get("child_id") or "")
            if cid and cid in children_by_id and cid not in dense_rank_map:
                dense_rank_map[cid] = idx + 1
                dense_score_map[cid] = round(float(hit.get("score") or 0.0), 5)

        # Stage 1B: Contextual BM25 Search over `enriched_text` (Breadcrumb + Entities + Child Text!)
        bm25_pairs = [
            (cid, _contextual_bm25_score(q_tokens, crow["enriched_text"]))
            for cid, crow in children_by_id.items()
        ]
        bm25_pairs.sort(key=lambda pair: pair[1], reverse=True)
        bm25_rank_map = {
            cid: idx + 1 for idx, (cid, sc) in enumerate(bm25_pairs) if sc > 0.0
        }
        bm25_score_map = {cid: round(sc, 5) for cid, sc in bm25_pairs if sc > 0.0}

        # Stage 2: Weighted Reciprocal Rank Fusion ($k = 60$)
        rrf_k = 60.0
        fused_candidates: list[dict[str, Any]] = []
        candidate_ids = set(dense_rank_map) | set(bm25_rank_map)
        if not candidate_ids:
            candidate_ids = set(list(children_by_id.keys())[: max(top_k * 2, 8)])

        for cid in candidate_ids:
            d_rk = dense_rank_map.get(cid)
            b_rk = bm25_rank_map.get(cid)
            rrf_val = (1.0 / (rrf_k + d_rk) if d_rk is not None else 0.0) + (
                1.15 / (rrf_k + b_rk) if b_rk is not None else 0.0
            )
            fused_candidates.append(
                {
                    "child_id": cid,
                    "child": children_by_id[cid],
                    "dense_rank": d_rk,
                    "dense_score": dense_score_map.get(cid, 0.0),
                    "bm25_rank": b_rk,
                    "bm25_score": bm25_score_map.get(cid, 0.0),
                    "rrf_score": round(rrf_val * 100.0, 5),
                }
            )

        fused_candidates.sort(key=lambda item: item["rrf_score"], reverse=True)
        shortlist = fused_candidates[: max(top_k * 4, 16)]

        # Stage 3: Batched Cross-Encoder Reranking (`NVIDIARerank` when configured + Precision Cross-Scorer)
        nvidia_rerank_scores: dict[str, float] = {}
        rerank_url = os.getenv("NVIDIA_RERANK_BASE_URL", "").strip()
        api_key = os.getenv("NVIDIA_API_KEY", "").strip()
        if rerank and shortlist and (rerank_url or os.getenv("NIMAGENTS_RAG_REMOTE_RERANK") == "1"):
            def _remote_rerank() -> dict[str, float]:
                from langchain_nvidia_ai_endpoints import NVIDIARerank

                rk_kwargs: dict[str, Any] = {
                    "model": os.getenv("NVIDIA_RERANK_MODEL", "nvidia/llama-nemotron-rerank-1b-v2"),
                    "top_n": len(shortlist),
                    "truncate": "END",
                }
                if rerank_url:
                    rk_kwargs["base_url"] = rerank_url
                if api_key:
                    rk_kwargs["api_key"] = api_key
                reranker = NVIDIARerank(**rk_kwargs)
                docs = [
                    Document(
                        page_content=item["child"]["enriched_text"],
                        metadata={"child_id": item["child_id"]},
                    )
                    for item in shortlist
                ]
                ranked_docs = list(reranker.compress_documents(documents=docs, query=query))
                return {
                    str(d.metadata["child_id"]): round(
                        float(d.metadata.get("relevance_score", 1.0 - idx * 0.05)), 5
                    )
                    for idx, d in enumerate(ranked_docs)
                }

            try:
                nvidia_rerank_scores = await asyncio.to_thread(_remote_rerank)
            except Exception as rk_exc:
                logger.warning("Remote NVIDIARerank fallback cleanly to local cross-scorer: %s", rk_exc)

        for item in shortlist:
            cid = item["child_id"]
            if rerank:
                local_ce = _cross_encoder_precision_score(query, item["child"]["enriched_text"])
                ce_score = nvidia_rerank_scores.get(cid, local_ce)
                item["rerank_score"] = round(ce_score, 5)
                item["final_score"] = round(0.45 * item["rrf_score"] + 2.5 * ce_score, 5)
            else:
                item["rerank_score"] = 0.0
                item["final_score"] = item["rrf_score"]

        shortlist.sort(key=lambda item: item["final_score"], reverse=True)

        # Stage 4: Parent-Section Expansion & Sibling Deduplication (`1` citation per unique `parent_id`!)
        seen_parents: set[str] = set()
        deduped_citations: list[dict[str, Any]] = []
        top_relevance = 0.0

        for item in shortlist:
            crow = item["child"]
            pid = crow["parent_id"]
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            parent_row = parents_by_id.get(pid)
            parent_md = parent_row["raw_markdown"] if parent_row else crow["raw_text"]
            top_relevance = max(
                top_relevance,
                _cross_encoder_precision_score(query, parent_md),
            )
            citation_num = len(deduped_citations) + 1
            deduped_citations.append(
                {
                    "citation_id": citation_num,
                    "child_id": crow["id"],
                    "parent_id": pid,
                    "source_id": crow["source_id"],
                    "doc_id": crow["doc_id"],
                    "doc_name": crow["doc_name"],
                    "heading_path": crow["heading_path"],
                    "snippet": crow["raw_text"],
                    "context_prefix": crow["context_prefix"],
                    "full_parent_markdown": parent_md,
                    "dense_rank": item["dense_rank"],
                    "dense_score": item["dense_score"],
                    "bm25_rank": item["bm25_rank"],
                    "bm25_score": item["bm25_score"],
                    "rrf_score": item["rrf_score"],
                    "rerank_score": item["rerank_score"],
                    "final_score": item["final_score"],
                }
            )
            if len(deduped_citations) >= top_k:
                break

        return (deduped_citations, top_relevance)

    async def hybrid_retrieve(
        self,
        query: str,
        *,
        source_ids: list[str] | None = None,
        top_k: int = 5,
        rerank: bool = True,
        enable_query_rewrite: bool = True,
        ctx: TenantContext | None = None,
    ) -> dict[str, Any]:
        """Execute full 6-stage Hybrid + Reranked + Parent-Deduplicated + Self-Grading Agentic RAG."""
        t0 = time.perf_counter()
        active = ctx or get_tenant_context()

        hits_pass1, grade_pass1 = await self._single_pass_retrieve(
            query,
            source_ids=source_ids,
            top_k=top_k,
            rerank=rerank,
            ctx=active,
        )

        rewritten_query: str | None = None
        final_hits = hits_pass1
        final_grade = grade_pass1

        # Stage 5: Self-Grading Relevance Check & Automatic Query Rewrite Fallback
        if enable_query_rewrite and grade_pass1 < 0.35:
            candidate_rewrite = rewrite_query_for_agentic_retry(query)
            if candidate_rewrite and candidate_rewrite.lower() != query.strip().lower():
                rewritten_query = candidate_rewrite
                hits_pass2, grade_pass2 = await self._single_pass_retrieve(
                    candidate_rewrite,
                    source_ids=source_ids,
                    top_k=top_k,
                    rerank=rerank,
                    ctx=active,
                )
                if grade_pass2 >= grade_pass1 and hits_pass2:
                    final_hits = hits_pass2
                    final_grade = grade_pass2

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return {
            "type": "rag_citations_result",
            "query": query,
            "rewritten_query": rewritten_query,
            "relevance_grade": round(final_grade, 4),
            "reranked": rerank,
            "latency_ms": elapsed_ms,
            "count": len(final_hits),
            "citations": final_hits,
        }


rag_pipeline = EnterpriseRagPipeline()

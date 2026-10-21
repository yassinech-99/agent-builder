"""Database schema, tenant migration & connection helpers for nimAgents."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Generator

from api.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_WORKSPACE_ID, get_tenant_context

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nimagents.db"


def get_db_path() -> Path:
    custom = os.getenv("NIMAGENTS_DB_PATH")
    if custom:
        return Path(custom)
    return DEFAULT_DB_PATH


def mask_secret(value: str | None) -> str:
    """Return a non-reversible masked representation of an API key or secret."""
    if not value:
        return ""
    stripped = value.strip()
    if not stripped:
        return ""
    if len(stripped) <= 8:
        return "••••••••"
    prefix = stripped[:6] if "-" in stripped[:7] else stripped[:4]
    return f"{prefix}••••••••{stripped[-4:]}"


CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'nvidia_nim',
    base_url TEXT DEFAULT '',
    api_key TEXT DEFAULT '',
    model_name TEXT NOT NULL,
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER,
    is_default BOOLEAN DEFAULT 0,
    role TEXT DEFAULT 'primary_chat',
    fallback_model_id TEXT DEFAULT '',
    context_window INTEGER DEFAULT 131072,
    supports_thinking BOOLEAN DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_STORAGE_CONNECTIONS = """
CREATE TABLE IF NOT EXISTS storage_connections (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    endpoint_url TEXT NOT NULL DEFAULT 'http://localhost:9000',
    bucket_name TEXT NOT NULL,
    region TEXT DEFAULT 'us-east-1',
    access_key TEXT DEFAULT '',
    secret_key TEXT DEFAULT '',
    secure BOOLEAN DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_GUARDRAIL_RULES = """
CREATE TABLE IF NOT EXISTS guardrail_rules (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    pattern TEXT DEFAULT '',
    action TEXT DEFAULT 'warn',
    enabled BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_GUARDRAIL_EVENTS = """
CREATE TABLE IF NOT EXISTS guardrail_events (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    rule_id TEXT,
    rule_name TEXT,
    category TEXT,
    action TEXT,
    matched_text TEXT,
    stage TEXT DEFAULT 'input',
    detector TEXT DEFAULT 'regex',
    latency_ms REAL DEFAULT 0.0,
    risk_score REAL DEFAULT 0.0,
    thread_id TEXT DEFAULT '',
    assistant_id TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RAG_SOURCES = """
CREATE TABLE IF NOT EXISTS rag_sources (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    endpoint_url TEXT NOT NULL DEFAULT 'http://localhost:8081/v1',
    collection_name TEXT NOT NULL DEFAULT 'default',
    api_key TEXT DEFAULT '',
    ranking_model TEXT DEFAULT 'nvidia/llama-nemotron-rerank-1b-v2',
    storage_bucket TEXT DEFAULT '',
    storage_prefix TEXT DEFAULT '',
    status TEXT DEFAULT 'ready',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_LOCAL_RAG_SOURCES = """
CREATE TABLE IF NOT EXISTS local_rag_sources (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'file',
    file_path TEXT DEFAULT '',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    index_path TEXT DEFAULT '',
    vector_dim INTEGER DEFAULT 384,
    embedding_model TEXT DEFAULT 'nvidia/llama-nemotron-embed-1b-v2',
    status TEXT DEFAULT 'pending',
    doc_count INTEGER DEFAULT 0,
    storage_bucket TEXT DEFAULT '',
    storage_prefix TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_LEXICAL_RAG_SOURCES = """
CREATE TABLE IF NOT EXISTS lexical_rag_sources (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'file',
    file_path TEXT DEFAULT '',
    meili_index TEXT NOT NULL,
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    status TEXT DEFAULT 'pending',
    doc_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_BACKGROUND_JOBS = """
CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    lane TEXT NOT NULL DEFAULT 'background',
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    detail TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_USAGE_LEDGER = """
CREATE TABLE IF NOT EXISTS usage_ledger (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT DEFAULT 'anonymous',
    agent_id TEXT DEFAULT 'agent',
    thread_id TEXT DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'nvidia_nim',
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    kv_cached_tokens INTEGER DEFAULT 0,
    estimated_usd REAL DEFAULT 0.0,
    cache_tier TEXT DEFAULT 'none',
    latency_ms REAL DEFAULT 0.0,
    ttft_ms REAL DEFAULT 0.0,
    queue_wait_ms REAL DEFAULT 0.0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CUSTOM_TOOLS = """
CREATE TABLE IF NOT EXISTS custom_tools (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    namespaced_id TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'openapi',
    method TEXT NOT NULL DEFAULT 'GET',
    url_template TEXT NOT NULL,
    headers_json TEXT NOT NULL DEFAULT '{}',
    input_schema_json TEXT NOT NULL DEFAULT '{}',
    input_examples_json TEXT NOT NULL DEFAULT '[]',
    defer_loading BOOLEAN DEFAULT 1,
    requires_approval BOOLEAN DEFAULT 0,
    side_effect_level TEXT NOT NULL DEFAULT 'read',
    timeout_seconds REAL DEFAULT 20.0,
    max_output_tokens INTEGER DEFAULT 2000,
    source_spec_title TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CONTEXT_ARTIFACTS = """
CREATE TABLE IF NOT EXISTS context_artifacts (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    thread_id TEXT NOT NULL DEFAULT '',
    tool_name TEXT NOT NULL DEFAULT 'tool',
    content_type TEXT NOT NULL DEFAULT 'text/plain',
    byte_size INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    storage_uri TEXT NOT NULL DEFAULT '',
    inline_content TEXT NOT NULL DEFAULT '',
    structural_summary TEXT NOT NULL DEFAULT '',
    preview_text TEXT NOT NULL DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_AGENT_SKILLS = """
CREATE TABLE IF NOT EXISTS agent_skills (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'engineering',
    version TEXT NOT NULL DEFAULT '1.0.0',
    tags_json TEXT NOT NULL DEFAULT '[]',
    allowed_tools_json TEXT NOT NULL DEFAULT '[]',
    body_markdown TEXT NOT NULL DEFAULT '',
    resources_json TEXT NOT NULL DEFAULT '{}',
    enabled BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SESSION_COMPACTIONS = """
CREATE TABLE IF NOT EXISTS session_compactions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    thread_id TEXT NOT NULL,
    agent_name TEXT NOT NULL DEFAULT 'nimAgents Assistant',
    prefix_hash TEXT NOT NULL DEFAULT '',
    static_prefix_tokens INTEGER DEFAULT 0,
    session_prefix_tokens INTEGER DEFAULT 0,
    dynamic_tail_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    context_window INTEGER DEFAULT 131072,
    kv_cache_hit_pct REAL DEFAULT 0.0,
    deferred_tools_count INTEGER DEFAULT 0,
    active_skills_json TEXT NOT NULL DEFAULT '[]',
    offloaded_artifacts_count INTEGER DEFAULT 0,
    cleared_thinking_tokens INTEGER DEFAULT 0,
    cleared_tool_tokens INTEGER DEFAULT 0,
    compaction_count INTEGER DEFAULT 0,
    checkpoint_summary TEXT NOT NULL DEFAULT '',
    pre_compact_facts_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_DURABLE_SESSIONS = """
CREATE TABLE IF NOT EXISTS durable_sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    assistant_id TEXT NOT NULL DEFAULT 'agent',
    title TEXT NOT NULL DEFAULT 'New Session',
    pinned BOOLEAN DEFAULT 0,
    archived BOOLEAN DEFAULT 0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    forked_from_session_id TEXT DEFAULT '',
    forked_from_seq INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'idle',
    pending_prompt TEXT DEFAULT '',
    message_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SESSION_EVENTS = """
CREATE TABLE IF NOT EXISTS session_events (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    seq_id INTEGER NOT NULL,
    run_id TEXT NOT NULL DEFAULT '',
    event_type TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'system',
    payload_json TEXT NOT NULL DEFAULT '{}',
    checkpoint_id TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SESSION_LEASES = """
CREATE TABLE IF NOT EXISTS session_leases (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL DEFAULT 1,
    acquired_at REAL NOT NULL DEFAULT 0,
    expires_at REAL NOT NULL DEFAULT 0
);
"""

CREATE_SUBAGENT_RUNS = """
CREATE TABLE IF NOT EXISTS subagent_runs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    parent_session_id TEXT NOT NULL DEFAULT '',
    child_session_id TEXT NOT NULL,
    specialist_role TEXT NOT NULL DEFAULT 'researcher',
    objective TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'completed',
    tokens_used INTEGER DEFAULT 0,
    steps_count INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0.0,
    distilled_output TEXT NOT NULL DEFAULT '',
    trace_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_MEMORY_BLOCKS = """
CREATE TABLE IF NOT EXISTS memory_blocks (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    agent_id TEXT NOT NULL DEFAULT 'default',
    block_type TEXT NOT NULL DEFAULT 'agent_notes',
    title TEXT NOT NULL DEFAULT 'Agent Notes',
    content TEXT NOT NULL DEFAULT '',
    token_count INTEGER DEFAULT 0,
    pinned BOOLEAN DEFAULT 1,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_SEMANTIC_MEMORIES = """
CREATE TABLE IF NOT EXISTS semantic_memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    agent_id TEXT NOT NULL DEFAULT 'default',
    subject TEXT NOT NULL,
    subject_norm TEXT NOT NULL,
    predicate TEXT NOT NULL,
    predicate_norm TEXT NOT NULL,
    object TEXT NOT NULL,
    object_norm TEXT NOT NULL,
    statement TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'fact',
    confidence REAL DEFAULT 0.9,
    importance REAL DEFAULT 0.7,
    access_count INTEGER DEFAULT 0,
    valid_from TEXT DEFAULT CURRENT_TIMESTAMP,
    invalidated_at TEXT DEFAULT NULL,
    superseded_by_id TEXT DEFAULT '',
    source_session_id TEXT DEFAULT '',
    source_turn_id TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EPISODIC_MEMORIES = """
CREATE TABLE IF NOT EXISTS episodic_memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    agent_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    problem TEXT NOT NULL DEFAULT '',
    approach TEXT NOT NULL DEFAULT '',
    key_tool_steps_json TEXT NOT NULL DEFAULT '[]',
    user_corrections TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT 'succeeded',
    importance REAL DEFAULT 0.75,
    access_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PROCEDURAL_MEMORIES = """
CREATE TABLE IF NOT EXISTS procedural_memories (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    agent_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    rule_prompt TEXT NOT NULL,
    rationale TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'proposed',
    version INTEGER NOT NULL DEFAULT 1,
    history_json TEXT NOT NULL DEFAULT '[]',
    source_session_id TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RAG_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    source_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    sha256_hash TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text/markdown',
    byte_size INTEGER DEFAULT 0,
    parent_count INTEGER DEFAULT 0,
    child_count INTEGER DEFAULT 0,
    contextual_enriched BOOLEAN DEFAULT 1,
    storage_uri TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RAG_PARENT_CHUNKS = """
CREATE TABLE IF NOT EXISTS rag_parent_chunks (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    source_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_name TEXT NOT NULL DEFAULT '',
    section_index INTEGER DEFAULT 0,
    heading_path TEXT NOT NULL DEFAULT '',
    raw_markdown TEXT NOT NULL DEFAULT '',
    token_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_RAG_CHILD_CHUNKS = """
CREATE TABLE IF NOT EXISTS rag_child_chunks (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    source_id TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    doc_name TEXT NOT NULL DEFAULT '',
    heading_path TEXT NOT NULL DEFAULT '',
    child_index INTEGER DEFAULT 0,
    raw_text TEXT NOT NULL DEFAULT '',
    context_prefix TEXT NOT NULL DEFAULT '',
    enriched_text TEXT NOT NULL DEFAULT '',
    token_count INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_NEMOGUARD_POLICIES = """
CREATE TABLE IF NOT EXISTS nemoguard_policies (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    nemoguard_enabled BOOLEAN DEFAULT 1,
    safety_model TEXT NOT NULL DEFAULT 'nvidia/llama-3.1-nemotron-safety-guard-8b-v3',
    topic_model TEXT NOT NULL DEFAULT 'nvidia/llama-3.1-nemoguard-8b-topic-control',
    jailbreak_model TEXT NOT NULL DEFAULT 'nvidia/nemoguard-jailbreak-detect',
    endpoint_url TEXT NOT NULL DEFAULT '',
    enforce_input BOOLEAN DEFAULT 1,
    enforce_tool_output BOOLEAN DEFAULT 1,
    enforce_model_output BOOLEAN DEFAULT 1,
    allowed_topics_json TEXT NOT NULL DEFAULT '[]',
    blocked_topics_json TEXT NOT NULL DEFAULT '["malware development", "credential theft", "insider trading"]',
    jailbreak_threshold REAL DEFAULT 0.65,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_VOICE_SESSIONS = """
CREATE TABLE IF NOT EXISTS voice_sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    thread_id TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    asr_engine TEXT NOT NULL DEFAULT 'nvidia/parakeet-ctc-1.1b-asr',
    tts_engine TEXT NOT NULL DEFAULT 'nvidia/magpie-tts-multilingual',
    voice_preset TEXT NOT NULL DEFAULT 'English-US.Female-1',
    sample_rate_hz INTEGER DEFAULT 16000,
    utterances_count INTEGER DEFAULT 0,
    barge_ins_count INTEGER DEFAULT 0,
    dropped_frames_count INTEGER DEFAULT 0,
    avg_asr_ms REAL DEFAULT 0.0,
    avg_ttft_ms REAL DEFAULT 0.0,
    avg_tts_ms REAL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EVAL_DATASETS = """
CREATE TABLE IF NOT EXISTS eval_datasets (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT 'regression',
    cases_json TEXT NOT NULL DEFAULT '[]',
    pass_threshold REAL DEFAULT 0.85,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EVAL_RUNS = """
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    dataset_id TEXT NOT NULL DEFAULT '',
    dataset_name TEXT NOT NULL DEFAULT '',
    run_mode TEXT NOT NULL DEFAULT 'regression',
    model_a TEXT NOT NULL DEFAULT 'nvidia/nemotron-3-super-120b-a12b',
    model_b TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'completed',
    pass_rate REAL DEFAULT 1.0,
    avg_score_a REAL DEFAULT 1.0,
    avg_score_b REAL DEFAULT 0.0,
    p50_latency_ms REAL DEFAULT 0.0,
    p95_latency_ms REAL DEFAULT 0.0,
    total_tokens INTEGER DEFAULT 0,
    estimated_usd REAL DEFAULT 0.0,
    results_json TEXT NOT NULL DEFAULT '[]',
    gate_passed BOOLEAN DEFAULT 1,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

TENANT_SCOPED_TABLES = (
    "models",
    "rag_sources",
    "local_rag_sources",
    "lexical_rag_sources",
    "storage_connections",
    "guardrail_rules",
    "guardrail_events",
    "background_jobs",
    "usage_ledger",
    "custom_tools",
    "context_artifacts",
    "agent_skills",
    "session_compactions",
    "durable_sessions",
    "session_events",
    "session_leases",
    "subagent_runs",
    "memory_blocks",
    "semantic_memories",
    "episodic_memories",
    "procedural_memories",
    "rag_documents",
    "rag_parent_chunks",
    "rag_child_chunks",
    "nemoguard_policies",
    "voice_sessions",
    "eval_datasets",
    "eval_runs",
)

_MIGRATE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    table: [
        ("tenant_id", "TEXT NOT NULL DEFAULT 'default'"),
        ("workspace_id", "TEXT NOT NULL DEFAULT 'default'"),
    ]
    for table in TENANT_SCOPED_TABLES
}
_MIGRATE_COLUMNS["models"].extend(
    [
        ("role", "TEXT DEFAULT 'primary_chat'"),
        ("fallback_model_id", "TEXT DEFAULT ''"),
        ("context_window", "INTEGER DEFAULT 131072"),
        ("supports_thinking", "BOOLEAN DEFAULT 0"),
    ]
)
_MIGRATE_COLUMNS["local_rag_sources"].append(("vector_dim", "INTEGER DEFAULT 384"))
_MIGRATE_COLUMNS["guardrail_events"].extend(
    [
        ("stage", "TEXT DEFAULT 'input'"),
        ("detector", "TEXT DEFAULT 'regex'"),
        ("latency_ms", "REAL DEFAULT 0.0"),
        ("risk_score", "REAL DEFAULT 0.0"),
    ]
)
_MIGRATE_COLUMNS["usage_ledger"].extend(
    [
        ("agent_id", "TEXT DEFAULT 'agent'"),
        ("thread_id", "TEXT DEFAULT ''"),
        ("kv_cached_tokens", "INTEGER DEFAULT 0"),
        ("ttft_ms", "REAL DEFAULT 0.0"),
        ("queue_wait_ms", "REAL DEFAULT 0.0"),
    ]
)


def _open_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection and guarantee it is closed on exit."""
    conn = _open_connection()
    try:
        yield conn
    finally:
        conn.close()


def get_db_raw() -> sqlite3.Connection:
    """Return a raw SQLite connection for callers managing lifecycle directly."""
    return _open_connection()


def ensure_tenant_guardrails_seeded(conn: sqlite3.Connection, tenant_id: str | None = None) -> None:
    """Ensure every tenant gets its baseline guardrail rule set on first access."""
    tid = tenant_id or get_tenant_context().tenant_id or DEFAULT_TENANT_ID
    existing = conn.execute(
        "SELECT COUNT(*) FROM guardrail_rules WHERE tenant_id = ?",
        (tid,),
    ).fetchone()[0]
    if existing > 0:
        return
    prefix = "" if tid == DEFAULT_TENANT_ID else f"{tid}:"
    for rule_id, name, category, pattern, action in _DEFAULT_GUARDRAIL_RULES:
        conn.execute(
            """INSERT OR IGNORE INTO guardrail_rules
               (id, tenant_id, workspace_id, name, category, pattern, action, enabled)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)""",
            (f"{prefix}{rule_id}", tid, DEFAULT_WORKSPACE_ID, name, category, pattern, action),
        )
    conn.commit()


def init_db() -> None:
    with get_db() as conn:
        for ddl in (
            CREATE_MODELS,
            CREATE_RAG_SOURCES,
            CREATE_LOCAL_RAG_SOURCES,
            CREATE_LEXICAL_RAG_SOURCES,
            CREATE_STORAGE_CONNECTIONS,
            CREATE_GUARDRAIL_RULES,
            CREATE_GUARDRAIL_EVENTS,
            CREATE_BACKGROUND_JOBS,
            CREATE_USAGE_LEDGER,
            CREATE_CUSTOM_TOOLS,
            CREATE_CONTEXT_ARTIFACTS,
            CREATE_AGENT_SKILLS,
            CREATE_SESSION_COMPACTIONS,
            CREATE_DURABLE_SESSIONS,
            CREATE_SESSION_EVENTS,
            CREATE_SESSION_LEASES,
            CREATE_SUBAGENT_RUNS,
            CREATE_MEMORY_BLOCKS,
            CREATE_SEMANTIC_MEMORIES,
            CREATE_EPISODIC_MEMORIES,
            CREATE_PROCEDURAL_MEMORIES,
            CREATE_RAG_DOCUMENTS,
            CREATE_RAG_PARENT_CHUNKS,
            CREATE_RAG_CHILD_CHUNKS,
            CREATE_NEMOGUARD_POLICIES,
            CREATE_VOICE_SESSIONS,
            CREATE_EVAL_DATASETS,
            CREATE_EVAL_RUNS,
        ):
            conn.execute(ddl)

        for table, cols in _MIGRATE_COLUMNS.items():
            for col_name, col_type in cols:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table}_tenant_ws ON {table}(tenant_id, workspace_id)"
            )

        ensure_tenant_guardrails_seeded(conn, DEFAULT_TENANT_ID)
        conn.commit()


_DEFAULT_GUARDRAIL_RULES = [
    ("pii-email", "Email Addresses", "pii", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "redact"),
    ("pii-phone", "Phone Numbers", "pii", r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "warn"),
    ("pii-ssn", "Social Security Numbers", "pii", r"\b\d{3}-\d{2}-\d{4}\b", "block"),
    ("secrets-nim-key", "NVIDIA NIM API Keys", "secrets", r"nvapi-[A-Za-z0-9_-]{20,}", "block"),
    ("secrets-api-key", "Generic API & Bearer Keys", "secrets", r"(?i)(api[_-]?key|apikey|secret[_-]?key|bearer)\s*[:=]\s*\S+", "warn"),
    ("harmful-prompt-injection", "Prompt Injection", "harmful", r"(?i)(ignore\s+(all\s+)?previous|disregard\s+(all\s+)?instructions)", "block"),
]

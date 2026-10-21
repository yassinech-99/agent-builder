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
    provider TEXT NOT NULL DEFAULT 'nvidia_nim',
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    estimated_usd REAL DEFAULT 0.0,
    cache_tier TEXT DEFAULT 'none',
    latency_ms REAL DEFAULT 0.0,
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


def _open_connection() -> sqlite3.Connection:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
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

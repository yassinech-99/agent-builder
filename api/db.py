"""SQLite database setup with context-managed connections."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "gridagents.db"

CREATE_MODELS = """
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'openai',
    base_url TEXT DEFAULT '',
    api_key TEXT DEFAULT '',
    model_name TEXT NOT NULL,
    temperature REAL DEFAULT 0.7,
    max_tokens INTEGER,
    is_default BOOLEAN DEFAULT 0,
    aws_region TEXT DEFAULT '',
    aws_secret_key TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_MIGRATE_MODELS_COLUMNS = [
    ("aws_region", "TEXT DEFAULT ''"),
    ("aws_secret_key", "TEXT DEFAULT ''"),
]

_MIGRATE_RAG_COLUMNS = [
    ("knowledge_base_id", "TEXT DEFAULT ''"),
    ("description", "TEXT DEFAULT ''"),
    ("region", "TEXT DEFAULT ''"),
    ("s3_bucket", "TEXT DEFAULT ''"),
    ("s3_prefix", "TEXT DEFAULT ''"),
]

CREATE_S3_CONNECTIONS = """
CREATE TABLE IF NOT EXISTS s3_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    bucket_name TEXT NOT NULL,
    region TEXT DEFAULT '',
    access_key_id TEXT DEFAULT '',
    secret_access_key TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_GUARDRAIL_RULES = """
CREATE TABLE IF NOT EXISTS guardrail_rules (
    id TEXT PRIMARY KEY,
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
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    knowledge_base_id TEXT NOT NULL DEFAULT '',
    region TEXT DEFAULT '',
    status TEXT DEFAULT 'ready',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_LOCAL_RAG_SOURCES = """
CREATE TABLE IF NOT EXISTS local_rag_sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'file',
    file_path TEXT DEFAULT '',
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    index_path TEXT DEFAULT '',
    embedding_model TEXT DEFAULT 'nova-multimodal',
    status TEXT DEFAULT 'pending',
    doc_count INTEGER DEFAULT 0,
    s3_bucket TEXT DEFAULT '',
    s3_prefix TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_LEXICAL_RAG_SOURCES = """
CREATE TABLE IF NOT EXISTS lexical_rag_sources (
    id TEXT PRIMARY KEY,
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


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection and guarantee it is closed on exit."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def get_db_raw() -> sqlite3.Connection:
    """Return a raw connection (for code that cannot use context managers, e.g. agent/model.py)."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.execute(CREATE_MODELS)
        conn.execute(CREATE_RAG_SOURCES)
        conn.execute(CREATE_LOCAL_RAG_SOURCES)
        conn.execute(CREATE_LEXICAL_RAG_SOURCES)
        conn.execute(CREATE_S3_CONNECTIONS)
        conn.execute(CREATE_GUARDRAIL_RULES)
        conn.execute(CREATE_GUARDRAIL_EVENTS)
        _seed_default_guardrail_rules(conn)
        for col_name, col_type in _MIGRATE_MODELS_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE models ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
        for col_name, col_type in _MIGRATE_RAG_COLUMNS:
            try:
                conn.execute(f"ALTER TABLE rag_sources ADD COLUMN {col_name} {col_type}")
            except Exception:
                pass
        conn.commit()


_DEFAULT_GUARDRAIL_RULES = [
    ("pii-email", "Email Addresses", "pii", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "redact"),
    ("pii-phone", "Phone Numbers", "pii", r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "warn"),
    ("pii-ssn", "Social Security Numbers", "pii", r"\b\d{3}-\d{2}-\d{4}\b", "block"),
    ("secrets-aws-key", "AWS Access Keys", "secrets", r"AKIA[0-9A-Z]{16}", "block"),
    ("secrets-api-key", "Generic API Keys", "secrets", r"(?i)(api[_-]?key|apikey)\s*[:=]\s*\S+", "warn"),
    ("harmful-prompt-injection", "Prompt Injection", "harmful", r"(?i)(ignore\s+(all\s+)?previous|disregard\s+(all\s+)?instructions)", "block"),
]


def _seed_default_guardrail_rules(conn: sqlite3.Connection) -> None:
    existing = conn.execute("SELECT COUNT(*) FROM guardrail_rules").fetchone()[0]
    if existing > 0:
        return
    for rule_id, name, category, pattern, action in _DEFAULT_GUARDRAIL_RULES:
        conn.execute(
            """INSERT OR IGNORE INTO guardrail_rules (id, name, category, pattern, action, enabled)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (rule_id, name, category, pattern, action),
        )

-- nimAgents Multi-Tenant PostgreSQL 17 Schema + Mandatory Row-Level Security (RLS)
-- Enforces strict tenant isolation even if an application query omits a WHERE clause:
-- Every transaction binds: SET LOCAL app.current_tenant_id = '<tenant_id>';

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'nvidia_nim',
    base_url TEXT DEFAULT '',
    api_key TEXT DEFAULT '',
    model_name TEXT NOT NULL,
    temperature DOUBLE PRECISION DEFAULT 0.7,
    max_tokens INTEGER,
    is_default BOOLEAN DEFAULT FALSE,
    role TEXT DEFAULT 'primary_chat',
    fallback_model_id TEXT DEFAULT '',
    context_window INTEGER DEFAULT 131072,
    supports_thinking BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    secure BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS guardrail_rules (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    pattern TEXT DEFAULT '',
    action TEXT DEFAULT 'warn',
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS background_jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    lane TEXT NOT NULL DEFAULT 'background',
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    detail TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS usage_ledger (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT DEFAULT 'anonymous',
    provider TEXT NOT NULL DEFAULT 'nvidia_nim',
    model_name TEXT NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    estimated_usd DOUBLE PRECISION DEFAULT 0.0,
    cache_tier TEXT DEFAULT 'none',
    latency_ms DOUBLE PRECISION DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable & FORCE Row-Level Security across every multi-tenant table
DO $$
DECLARE
    tbl TEXT;
BEGIN
    FOREACH tbl IN ARRAY ARRAY[
        'models',
        'rag_sources',
        'local_rag_sources',
        'lexical_rag_sources',
        'storage_connections',
        'guardrail_rules',
        'guardrail_events',
        'background_jobs',
        'usage_ledger'
    ]
    LOOP
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_%I_tenant_ws ON %I (tenant_id, workspace_id)', tbl, tbl);
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', tbl);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation_policy ON %I', tbl);
        EXECUTE format(
            'CREATE POLICY tenant_isolation_policy ON %I '
            'USING (tenant_id = COALESCE(NULLIF(current_setting(''app.current_tenant_id'', true), ''''), ''default'')) '
            'WITH CHECK (tenant_id = COALESCE(NULLIF(current_setting(''app.current_tenant_id'', true), ''''), ''default''))',
            tbl
        );
    END LOOP;
END
$$;

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
    defer_loading BOOLEAN DEFAULT TRUE,
    requires_approval BOOLEAN DEFAULT FALSE,
    side_effect_level TEXT NOT NULL DEFAULT 'read',
    timeout_seconds DOUBLE PRECISION DEFAULT 20.0,
    max_output_tokens INTEGER DEFAULT 2000,
    source_spec_title TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    kv_cache_hit_pct DOUBLE PRECISION DEFAULT 0.0,
    deferred_tools_count INTEGER DEFAULT 0,
    active_skills_json TEXT NOT NULL DEFAULT '[]',
    offloaded_artifacts_count INTEGER DEFAULT 0,
    cleared_thinking_tokens INTEGER DEFAULT 0,
    cleared_tool_tokens INTEGER DEFAULT 0,
    compaction_count INTEGER DEFAULT 0,
    checkpoint_summary TEXT NOT NULL DEFAULT '',
    pre_compact_facts_json TEXT NOT NULL DEFAULT '[]',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS durable_sessions (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    assistant_id TEXT NOT NULL DEFAULT 'agent',
    title TEXT NOT NULL DEFAULT 'New Session',
    pinned BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE,
    tags_json TEXT NOT NULL DEFAULT '[]',
    forked_from_session_id TEXT DEFAULT '',
    forked_from_seq INTEGER DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'idle',
    pending_prompt TEXT DEFAULT '',
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_leases (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    workspace_id TEXT NOT NULL DEFAULT 'default',
    session_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    fencing_token INTEGER NOT NULL DEFAULT 1,
    acquired_at DOUBLE PRECISION NOT NULL DEFAULT 0,
    expires_at DOUBLE PRECISION NOT NULL DEFAULT 0
);

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
    duration_ms DOUBLE PRECISION DEFAULT 0.0,
    distilled_output TEXT NOT NULL DEFAULT '',
    trace_json TEXT NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    pinned BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
    confidence DOUBLE PRECISION DEFAULT 0.9,
    importance DOUBLE PRECISION DEFAULT 0.7,
    access_count INTEGER DEFAULT 0,
    valid_from TIMESTAMPTZ DEFAULT NOW(),
    invalidated_at TIMESTAMPTZ DEFAULT NULL,
    superseded_by_id TEXT DEFAULT '',
    source_session_id TEXT DEFAULT '',
    source_turn_id TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
    importance DOUBLE PRECISION DEFAULT 0.75,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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
    contextual_enriched BOOLEAN DEFAULT TRUE,
    storage_uri TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
    created_at TIMESTAMPTZ DEFAULT NOW()
);

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
        'usage_ledger',
        'custom_tools',
        'context_artifacts',
        'agent_skills',
        'session_compactions',
        'durable_sessions',
        'session_events',
        'session_leases',
        'subagent_runs',
        'memory_blocks',
        'semantic_memories',
        'episodic_memories',
        'procedural_memories',
        'rag_documents',
        'rag_parent_chunks',
        'rag_child_chunks'
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

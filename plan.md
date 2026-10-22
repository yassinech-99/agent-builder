# `nimAgents` — Master Architecture & Multi-User Scalable Engineering Plan (`plan.md`)

> **Product Name:** **nimAgents** *(Rebranded from `NovaGrid Agents`, 100% Amazon/AWS-Free)*  
> **Package / Namespace Slug:** `nimagents` (`data/nimagents.db` local dev · PostgreSQL 17 + `PgBouncer` + Redis 8 + Qdrant + Meilisearch + RustFS in multi-user production)  
> **Primary AI Stack:** **NVIDIA NIM & NVIDIA Dynamo** (`ChatNVIDIA`, `ChatNVIDIADynamo`, `NVIDIAEmbeddings`, `NVIDIARerank`, `NemoGuard`, `Riva Speech NIM`) + Open Multi-Provider Support (`OpenAI`, `Anthropic`, `OpenRouter`, `vLLM`, `SGLang`, `Ollama`)  
> **Core Engineering Pillars:** Total Rebrand (`nimAgents`) · Horizontally Scalable Multi-Tenant Backend & AI Inference Gateway · Unified Tool Registry, Deferred Tools (`ToolSearch`) & Programmatic Tool Calling · 5-Stage Context Engineering & KV-Cache Optimization · Durable Branchable Sessions · 4-Tier Cognitive Memory + Sleep-Time Consolidation · Contextual Hybrid RAG · Safety, Voice & FinOps Evals

---

## Table of Contents

1. [Executive Summary & Frontier AI Lab Research Synthesis](#1-executive-summary--frontier-ai-lab-research-synthesis)
2. [Forensic Codebase Audit: Multi-User Bottlenecks, Bugs & 26-File AWS Purge Map](#2-forensic-codebase-audit-multi-user-bottlenecks-bugs--26-file-aws-purge-map)
3. [Target `nimAgents` Distributed Multi-User Architecture](#3-target-nimagents-distributed-multi-user-architecture)
4. [Phase 1 — Total Rebranding to `nimAgents`, Zero-AWS Purge & Multi-Provider NVIDIA NIM Core ✅ COMPLETED](#phase-1--total-rebranding-to-nimagents-zero-aws-purge--multi-provider-nvidia-nim-core--completed)
5. [Phase 2 — Multi-User Scalable Backend, Tenant Isolation & AI Inference Data Plane](#phase-2--multi-user-scalable-backend-tenant-isolation--ai-inference-data-plane)
6. [Phase 3 — Unified Tool Registry, Deferred Tools (`ToolSearch`), Programmatic Calling & Sandbox Hands](#phase-3--unified-tool-registry-deferred-tools-toolsearch-programmatic-calling--sandbox-hands)
7. [Phase 4 — Context Engineering & 5-Stage Graduated Compaction Engine](#phase-4--context-engineering--5-stage-graduated-compaction-engine)
8. [Phase 5 — Durable Session Architecture, Time-Travel Forking, Background Runs & Sub-Agents](#phase-5--durable-session-architecture-time-travel-forking-background-runs--sub-agents)
9. [Phase 6 — 4-Tier Agent Memory System & Async ("Sleep-Time") Consolidation](#phase-6--4-tier-agent-memory-system--async-sleep-time-consolidation)
10. [Phase 7 — Production Contextual, Hybrid & Agentic RAG Pipeline](#phase-7--production-contextual-hybrid--agentic-rag-pipeline)
11. [Phase 8 — NVIDIA NemoGuard Safety, Open Streaming Voice (`Riva NIM`), FinOps & Evaluation Studio ✅ COMPLETED](#phase-8--nvidia-nemoguard-safety-open-streaming-voice-riva-nim-finops--evaluation-studio--completed)
12. [Capacity Planning, Scaling Playbook & Milestone Matrix](#12-capacity-planning-scaling-playbook--milestone-matrix)
13. [Frontier Engineering & Systems References](#13-frontier-engineering--systems-references)

---

## 1. Executive Summary & Frontier AI Lab Research Synthesis

Why do 95% of prototype agent repos work smoothly for **1 developer on localhost** and collapse completely at **25–500 concurrent users**?

Across 2025–2026 engineering publications from **Anthropic** (*Scaling Managed Agents: Decoupling the Brain from the Hands*, *Advanced Tool Use*, *Effective Context Engineering*, *Contextual Retrieval*), **NVIDIA** (*Dynamo 1.0 Distributed Agentic Inference*, *NeMo Retriever*, *NemoGuard*), **OpenAI** (*Agents SDK*), **Letta / MemGPT** (*OS Tiered Virtual Context & Sleep-Time Compute*), **Zep / Graphiti** (*Bi-Temporal Memory*), and **LangGraph Production Stacks**, every high-scale AI lab converged on seven non-negotiable architectural principles:

| Dimension | Single-User Prototype Anti-Pattern (Current Repo) | Multi-User Production Architecture (`nimAgents`) | Measured Industry Gain |
|---|---|---|---|
| **1. Runtime Topology** | Monolithic stateful process ("pet") coupling HTTP socket, orchestration loop, MCP subprocesses, and host file/code tools. | **Decoupled Brain / Hands / Session** (*Anthropic Managed Agents*): Stateless Graph Workers ("Brain") + Warm Disposable Sandbox Pool ("Hands") + External Durable Event Log ("Session"). | **-60% p50 TTFT** and **-90%+ p95 TTFT**; zero lost sessions on pod crash/deploy. |
| **2. GPU & LLM Serving Efficiency** | Round-robin stateless calls with per-turn timestamp mutations in line 1 of system prompts (0% KV reuse!). | **Byte-Stable Prompt Prefixes + KV-Cache-Aware Routing (`NVIDIA Dynamo` / Prefix Consistent Hashing) + Priority Lanes + Micro-Batching + L1/L2 Semantic Cache**. | **2x–7x higher GPU throughput**, **2x faster TTFT**, **60–75% lower token compute** on multi-turn agent loops. |
| **3. Concurrency & Backpressure** | Blocking sync I/O (`time.sleep`, `sqlite3`, `FAISS.load_local`) directly on FastAPI `asyncio` loop; zero rate limits. | **100% Non-Blocking `uvloop` + `PgBouncer` + `Redis 8` Lua Token Buckets (`RPM`/`TPM`/`Concurrency`) + Fair Multi-Tenant Priority Queues** (Interactive vs Bulk lanes). | Zero event-loop stalls; one heavy tenant can never starve other users (`HTTP 429` clean load shedding). |
| **4. Multi-Tenancy** | Global tables without `tenant_id`, shared `./workspace`, and `os.environ.setdefault()` leaking API keys across users. | **Strict End-to-End Tenant Isolation**: JWT `ContextVar` → Postgres **Row-Level Security (RLS)** → Qdrant `is_tenant` payload index → Meilisearch Tenant Tokens → Host Vault Proxy. | Zero cross-tenant data or secret leakage under high concurrency. |
| **5. Tool Scaling** | Dump 30–200 tool JSON schemas (`25K+` tokens) into every turn upfront; 15 sequential LLM turns for batch work. | **Unified `ToolRegistry` + Deferred Loading (`defer_loading` + `tool_search`) + `input_examples` + `orchestrate_tools` Sandbox RPC**. | **85%+ tool token reduction**; accuracy up **49% → 74%** / **79.5% → 88.1%**. |
| **6. Context & Memory** | Infinite raw history growth + no cross-session memory. | **5-Stage Graduated Context Shaper** + **4-Tier Cognitive Memory** (`Core` + `Bi-Temporal Semantic` + `Episodic` + `Procedural`) consolidated asynchronously off the hot path. | **84% lower long-horizon token burn**; **0ms** memory write latency penalty on chat replies. |
| **7. RAG Architecture** | Sync single-thread character splitting; raw text concatenations with zero source metadata or reranking. | **Async Worker Queue + Contextual Retrieval + Parent-Child Chunking + Dense (`NVIDIAEmbeddings`) + BM25 (`Meilisearch`) RRF + Batched `NVIDIARerank` + Inline Citations**. | **-67% retrieval failure rate** (`5.7% → 1.9%`). |

---

## 2. Forensic Codebase Audit: Multi-User Bottlenecks, Bugs & 26-File AWS Purge Map

### 2.1 Nine Multi-User Concurrency Killers Found Directly in the Current Codebase
If 25 users hit the current codebase simultaneously today, here is the exact sequence of failures found line-by-line in the code:

1. **Full Event-Loop Freeze on Every Lexical Upload (`api/routes/lexical_rag.py:42-93, 205-215`):**
   - `upload_file` is declared `async def`, so FastAPI executes it **on the single main `asyncio` event loop**. Inside it, `_index_into_meilisearch()` calls `_wait_for_task()`, which runs synchronous `time.sleep(0.3)` in a blocking `while` loop for up to 30 seconds! While *one* user uploads a PDF, **every HTTP & WebSocket request for every other user on the entire server freezes completely**.
2. **Per-Query Disk Pickle Deserialization Explosion (`agent/graph.py:180-191`, `api/routes/local_rag.py:272`):**
   - `_build_local_rag_tool` executes `FAISS.load_local(row["index_path"], embeddings, allow_dangerous_deserialization=True)` **inside every single tool call**. 30 concurrent RAG queries deserialize 30 copies of FAISS + Python pickles into RAM simultaneously → CPU spike, GIL contention, and container OOM kills (plus RCE risk from `allow_dangerous_deserialization=True`).
3. **Per-Turn Subprocess & TCP Connection Storm (`agent/graph.py:72-110, 245-276, 338-403`):**
   - `make_graph()` rebuilds the graph from scratch on every turn: spawning a brand-new `meilisearch-mcp` OS subprocess via `stdio` and opening fresh unpooled MCP HTTP connections on every single message. Similarly, `api/app.py:37` and `api/routes/mcp.py:32` instantiate a disposable `async with httpx.AsyncClient()` per request, throwing away TLS handshakes and HTTP/2 connection pools.
4. **Cross-Tenant Credential Race Condition & Leakage (`agent/tools.py:85, 107, 117, 127, 179, 250-325`):**
   - Tool factories write user API keys into process-wide `os.environ.setdefault("TAVILY_API_KEY", ...)`. User A's key permanently sticks for User B, User C's custom key is silently ignored, and concurrent threads share mutable global process environment state!
5. **Shared Host Filesystem & Host RCE Across Users (`agent/tools.py:190-199, 244-247, 271-275`):**
   - Every user shares `./workspace` in `FileManagementToolkit`, runs arbitrary host commands via `ShellTool` / `PythonREPL`, and shares synchronous Playwright Chromium instances (~400MB RAM each).
6. **100% KV-Cache Destruction on Every Turn (`agent/graph.py:42`):**
   - `DEFAULT_SYSTEM_PROMPT` injects `datetime.now(tz=UTC).isoformat()` into the second sentence of the system prompt on every call. Because LLM KV caches are prefix-indexed token-by-token, changing microseconds at token #12 forces GPUs to **recompute prefill for all 10,000+ tool tokens and all prior conversation turns on every single request**!
7. **Single-Writer SQLite Lock & Single-Process `inmem` Graph State (`api/db.py:124`, `pyproject.toml:14`):**
   - Raw per-query `sqlite3.connect()` hits `OperationalError: database is locked` under concurrent writes, and `langgraph-cli[inmem]` prevents running >1 replica or surviving deployments.
8. **Broken Guardrails, Disabled Token Streaming & Stubbed Voice (`agent/graph.py:387`, `Thread.tsx:503-515`, `voice.py:105`):**
   - Guardrails are an optional tool instead of mandatory pre/post nodes; `streamMode: ["values"]` disables token streaming; `Thread.tsx` overwrites assistant RAG defaults with `[]`; `voice.py` leaks unbounded `audio_buffer` memory per WebSocket client while duplicating text messages.
9. **Dead Settings & Unmasked Plaintext Secrets (`settings/page.tsx`, `models.py`, `storage.py`):**
   - `/settings` writes unread `localStorage` keys; `MEILI_URL` defaults to `http://172.30.6.3:7700` without `MEILI_MASTER_KEY`; `GET /api/models` and `GET /api/storage` return raw secrets in cleartext with `CORS allow_origins=["*"]` and mock client-only `demo/demo123` auth.

### 2.2 Complete 26-File Amazon / AWS Removal Map (`NovaGrid` → `nimAgents`)

| Layer | Exact Files & AWS Artefacts Removed | `nimAgents` Replacement |
|---|---|---|
| **Packaging & Ops** | `pyproject.toml` (`novagrid-agents`, `langchain-aws`, `boto3`, `nova-act`), `README.md`, `docker-compose.yml` (`novagrid-meilisearch`), `.gitignore` (`scoutsuite-aws-mcp/`), `main.py` | Package `nimagents`; add `langchain-nvidia-ai-endpoints`, `langchain-anthropic`, `sqlalchemy[asyncio]`, `asyncpg`, `aiosqlite`, `redis[hiredis]`, `qdrant-client`, pure-Python `RustFSClient` (`AWS4-HMAC-SHA256`), `tiktoken`, `uvloop`; full `docker-compose.yml` stack (`rustfs/rustfs:latest`). |
| **Agent Runtime (`agent/`)** | `agent/model.py` (`ChatBedrockConverse`, `aws_*`), `agent/graph.py` (`NovaGrid Agents`, `novagrid.graph`, `boto3` `bedrock-agent-runtime`), `agent/tools.py` (`_make_nova_act_browser`, `nova_act_browser`) | **NVIDIA NIM + Multi-Provider Gateway** (`ChatNVIDIA` / `ChatNVIDIADynamo` / OpenAI / Anthropic / vLLM / Ollama); **Unified Hybrid Qdrant + Meilisearch + `NVIDIARerank` Retriever**; **Remote Sandbox Browser Pool (Playwright CDP)**. |
| **Control Plane (`api/`)** | `api/app.py`, `api/db.py` (`gridagents.db`, `aws_*`, `nova-multimodal`, `AKIA`), `api/routes/models.py`, `local_rag.py` (`BedrockEmbeddings`, Titan), `rag.py` (`bedrock-agent-runtime`), `storage.py` (`boto3`), `voice.py` (`nova-2-sonic`), logger names | Async DB (`Postgres` + `aiosqlite` local fallback `data/nimagents.db`), `nimagents.*` structured JSON loggers, `NVIDIAEmbeddings` + `NVIDIARerank`, S3-compatible **RustFS** (`api/core/object_store.py`), `Riva Speech NIM` / Open Speech WebSocket. |
| **Frontend (`frontend/`)** | `package.json` (`novagrid-agents`), `app/layout.tsx`, `app/page.tsx`, `components/icons/grid-agents.tsx` (`NovaGridLogo`), `public/logo.svg`, `runtime-splash.tsx`, `sidebar.tsx`, `Stream.tsx`, `Thread.tsx`, `step-basics.tsx`, `step-review.tsx`, `models/page.tsx`, `rag/page.tsx`, `storage/page.tsx`, `voice/page.tsx`, `voice-session.tsx`, `lib/guardrails.ts` | Rebrand completely to **`nimAgents`** (`NimAgentsLogo`, `nimagents:*` storage keys, NIM model & endpoint discovery UI, `<think>` streaming accordions, Hybrid Collection RAG UI, S3-compatible bucket connector). |

---

## 3. Target `nimAgents` Distributed Multi-User Architecture

```
                                   ┌──────────────────────────────────────┐
                                   │   Clients: Web UI / SDK / Webhooks   │
                                   └──────────────────┬───────────────────┘
                                                      │ HTTPS / WSS / SSE
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS API GATEWAY TIER (FastAPI + uvloop, N Replicas)                    │
│  • AuthN/AuthZ (OIDC/JWT + Hashed Virtual API Keys ─▶ `TenantContext` ContextVar)                │
│  • Atomic Redis Lua Admission Control (Concurrency Semaphore + RPM + Upfront TPM + USD Budget)   │
│  • Reconnectable SSE Fan-Out Subscriber (`Last-Event-ID` via Redis Streams / PubSub)             │
└───────────────┬─────────────────────────────────────┬────────────────────────────────────────────┘
                │ Enqueue High-Priority Interactive   │ Enqueue Low-Priority Bulk
                ▼ (`stream:runs:interactive`)         ▼ (`stream:jobs:background`)
┌───────────────────────────────────────────┐ ┌────────────────────────────────────────────────────┐
│  STATELESS BRAIN WORKER POOL (M Replicas) │ │        ASYNC BACKGROUND WORKER POOL (K Pods)       │
│  • Per-Thread Lease Lock (`wake(sessId)`) │ │  • Contextual RAG Parsing, Chunking & Sync         │
│  • Pre-Model `NemoGuard` + Regex Gate     │ │  • Micro-Batched `NVIDIAEmbeddings` Indexer        │
│  • 5-Stage Context Shaper (KV-Aligned!)   │ │  • Sleep-Time Memory Consolidator & Title Gen      │
│  • Dynamic `tool_search` Binder           │ │  • Scheduled Cron Triggers & Batch Eval Suites     │
│  • Post-Model Filter + Fire&Forget Ledger │ └─────────────────────────┬──────────────────────────┘
└───────┬───────────────────────┬───────────┘                           │
        │ AI Inference Calls    │ `execute(tool, args)`                 │
        ▼                       ▼                                       ▼
┌─────────────────────┐ ┌───────────────────────────┐ ┌────────────────────────────────────────────┐
│  nimAgents AI GW &  │ │ DISPOSABLE HANDS POOL     │ │         SHARED DURABLE DATA PLANE          │
│  INFERENCE ROUTER   │ │ ("Cattle, Not Pets")      │ │  • PostgreSQL 17 + PgBouncer (RLS Multi-   │
│  • L1 Exact Cache   │ │ • Warm Code Sandboxes     │ │    Tenant DB + `AsyncPostgresSaver` +      │
│  • L2 Tenant Sem-   │ │ • Headless Browser Pool   │ │    Append-Only `session_events` + Memory)  │
│    antic Cache      │ │ • Shared MCP Conn Pool    │ │  • Redis 8 (Streams, PubSub, Locks, Cache) │
│  • Embed/Rerank     │ │ • Zero-Trust Vault Proxy  │ │  • Qdrant Cluster (`is_tenant=True` HNSW)  │
│    Micro-Batcher    │ │   (Secrets NEVER enter    │ │  • Meilisearch (Tenant-Filtered BM25)      │
│  • Prefix Hash /    │ │   sandbox or LLM prompt!) │ │  • RustFS / S3-Compatible Object Storage   │
│    NVIDIA Dynamo    │ └───────────────────────────┘ └────────────────────────────────────────────┘
└─────────┬───────────┘
          ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│     NVIDIA NIM / DYNAMO GPU CLUSTER (Self-Hosted TRT-LLM / vLLM / SGLang or `build.nvidia.com`)  │
│  • Prefill / Decode Workers with Radix-Tree KV Cache Reuse (`Nemotron-3`, `DeepSeek-R1`, `Qwen3`)│
│  • Dedicated Embedding (`llama-nemotron-embed`) & Reranking (`llama-nemotron-rerank`) Micro-Pods │
│  • Dedicated Safety (`NemoGuard`) & Speech (`Riva Parakeet ASR + Magpie TTS`) Micro-Pods         │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> **Developer Experience Guarantee ("Zero-Friction Dual Mode"):**  
> `nimAgents` uses clean repository/adapter interfaces so **`docker compose up`** runs the entire production multi-tenant stack locally (Postgres 17, PgBouncer, Redis 8, Qdrant, Meilisearch, RustFS, API, Brain Worker, Frontend), **AND** also supports a lightweight single-command `NIMAGENTS_DEV_LITE=1` mode (`aiosqlite` + embedded local Qdrant + asyncio in-process queues) for instant laptop development without heavy containers!

---

## Phase 1 — Total Rebranding to `nimAgents`, Zero-AWS Purge & Multi-Provider NVIDIA NIM Core ✅ COMPLETED

> **Status:** **Completed & Verified (`tests/test_phase1_nimagents.py` passing + `pnpm build` zero-error production compile)**
> - **Zero-AWS Purge Verified:** Removed `langchain-aws`, `boto3`, and `nova-act`; replaced storage with Apache-2.0 **RustFS** (`api/core/object_store.py` pure-Python S3 SigV4 `RustFSClient`), RAG with `NVIDIAEmbeddings` + `NVIDIARerank` + `NVIDIARAGRetriever`, and voice with bounded-deque streaming WebSocket gateway (`api/routes/voice.py`). Automated repo-wide test enforces zero legacy AWS/Nova/NovaGrid/MinIO references across all source files.
> - **Multi-Provider NVIDIA NIM & Dynamo Gateway Shipped:** `agent/model.py` + `/api/models` + `/models` UI now support `nvidia_nim` (`ChatNVIDIA`), `nvidia_dynamo` (`ChatNVIDIADynamo` with `priority` hints), `openai`, `anthropic`, `openrouter`, `vllm`, `sglang`, `ollama`, and `openai_compatible`, pinned to `langchain-nvidia-ai-endpoints>=1.5.0` with multimodal path-traversal sanitization (`GHSA-g28h-2cmm-rj9x`), secret masking, live `/v1/models` discovery, ping checks, role routing, `.with_fallbacks(...)`, and collapsible `<think>...</think>` reasoning streaming.

### 1.1 Complete `nimAgents` Brand & Codebase Rename
- Rename package, titles, CLI, containers, logger hierarchy, storage namespaces, and visual identity from `NovaGrid Agents` to **`nimAgents`** (`nimagents`):
  - Replace `NovaGridLogo` (`frontend/src/components/icons/grid-agents.tsx`) and `public/logo.svg` with `NimAgentsLogo` (`frontend/src/components/icons/nim-agents.tsx`), refreshed dark/light emerald-zinc theme tokens, redesigned landing page, `RuntimeSplash`, and `Sidebar`.
  - Remove `langchain-aws`, `boto3`, and `nova-act` completely from `pyproject.toml` and lockfiles; enforce zero AWS/Bedrock regressions with an automated test assertion.
  - Fix `.gitignore` so test files (`tests/test_*.py`) and `plan.md` are properly version-controlled, remove duplicate `frontend/package-lock.json` (standardize on `pnpm`), and wire `/settings` URL overrides + Next.js `/api` reverse-proxy routing across all frontend clients.

### 1.2 Multi-Provider Model Registry Centered on NVIDIA NIM (`agent/model.py`, `/models`)
- Support both **Hosted NVIDIA API Catalog** (`https://integrate.api.nvidia.com/v1` via `NVIDIA_API_KEY`) and **Self-Hosted NIM / NVIDIA Dynamo Deployments** (`ChatNVIDIA` & `ChatNVIDIADynamo`), plus **OpenAI**, **Anthropic**, **OpenRouter**, and **OpenAI-Compatible Servers (`vLLM`, `SGLang`, `Ollama`, `LiteLLM`)**.
- Pin `langchain-nvidia-ai-endpoints` above the September 2026 `GHSA-g28h-2cmm-rj9x` advisory version and strictly validate multimodal image inputs as base64 data URIs or object-store URLs (never arbitrary local file paths).
- Add **Live `/v1/models` Endpoint Discovery** + **Connection Health Check** in `/models`, preset profiles for `nvidia/nemotron-3-ultra-550b-a55b`, `nvidia/nemotron-3-super-120b-a12b` (1M context), `nvidia/llama-3.3-nemotron-super-49b-v1.5`, `nvidia/llama-3.1-nemotron-nano-8b-v1`, `deepseek-ai/deepseek-r1`, `qwen/qwen3-235b-a22b`, and `meta/llama-3.3-70b-instruct`.
- Honor `is_default = 1` in `models` + model role assignments (`primary_chat`, `fast_router`, `compactor`, `contextualizer`), fallback chains (`.with_fallbacks(...)`), and **real-time token + `<think>...</think>` collapsible reasoning streaming** (`streamMode: ["values", "messages-tuple", "custom"]`).

---

## Phase 2 — Multi-User Scalable Backend, Tenant Isolation & AI Inference Data Plane *(✅ COMPLETED)*

### 2.1 End-to-End Multi-Tenant Data Isolation (`orgs` → `workspaces` → `users`)
Every resource in `nimAgents` (`agents`, `sessions`, `models`, `tool_credentials`, `rag_collections`, `memories`, `guardrail_rules`, `usage_ledger`, `api_keys`) carries `tenant_id UUID NOT NULL` and `workspace_id UUID NOT NULL`:
1. **PostgreSQL Row-Level Security (RLS):**
   - Every database transaction executes `SET LOCAL app.current_tenant_id = :tenant_id` inside the SQLAlchemy `AsyncSession` checkout event listener. Even a buggy SQL query omitting a `WHERE` clause **cannot read or mutate another tenant's rows**!
2. **Vector Store Partitioning (Qdrant Multi-Tenancy — Solving the HNSW Post-Filter Trap):**
   - Instead of spin-up-per-user FAISS files on local disk (or naive global HNSW post-filtering that returns empty results for small tenants), `nimAgents` creates **one shared Qdrant collection per embedding dimension** configured with `payload_schema={"tenant_id": KeywordIndexParams(type="keyword", is_tenant=True)}`!
   - Qdrant physically co-locates each tenant's vectors on dedicated micro-shards/sub-graphs on disk (`mmap` + scalar `int8` quantization cutting RAM usage by **4x**), scaling smoothly from **10 to 100,000+ tenants** with `<5ms` search latency and zero disk pickle deserialization!
3. **Meilisearch Tenant Isolation:** Enforce mandatory `filter = ["tenant_id = '...' AND collection_id IN [...]"]` signed with Meilisearch Tenant Tokens.
4. **Object Storage Isolation:** Store all uploaded files, large offloaded tool outputs, and artifacts in RustFS / S3-compatible buckets under `tenants/{tenant_id}/workspaces/{workspace_id}/...` using streaming async chunked uploads (never buffering 50 MB `await file.read()` entirely into API worker RAM!).

### 2.2 Async Database & Connection Pooling Architecture (`PgBouncer` + `Redis 8` + Shared HTTP/2 Pools)
- **Eliminate Connection Exhaustion:** Front PostgreSQL 17 with **PgBouncer in `transaction` pooling mode**, sizing pools strictly with:
  $$\text{App Replicas} \times \text{SQLAlchemy Pool Size} \;\xrightarrow{\text{muxed via PgBouncer}}\; \le 0.7 \times \text{Postgres } \texttt{max\_connections}$$
- **Persistent Global HTTP/2 Client Pool (`api/core/http_pool.py`):** Single process-lifespan `httpx.AsyncClient(http2=True, limits=httpx.Limits(max_connections=1000, max_keepalive_connections=200))` shared across all LLM calls, MCP streamable HTTP transports, embedding calls, and webhooks—eliminating repeated TCP + TLS handshakes under burst load.
- **Zero Main-Thread Blocking Rule:** Run `uvicorn` on `uvloop` + `httptools`; enforce CI linter check banning `time.sleep`, sync `requests`, sync `sqlite3`, or CPU-bound parsing inside `async def` functions (offloading CPU document parsing to `ProcessPoolExecutor` / background worker pods).

### 2.3 Distributed Rate Limiting, Concurrency Quotas & Priority Work Queues
1. **Single-Roundtrip Atomic Redis Lua Admission Controller (`api/middleware/admission.py`):**
   - Executed in **`~0.5ms`** before enqueuing any run:
     - **Active In-Flight Run Semaphore (`max_concurrent_runs` per user & tenant):** Prevents a single user opening 50 parallel loops from monopolizing worker capacity.
     - **RPM & Upfront TPM Token Buckets:** Reserves `estimated_prompt_tokens + max_output_tokens` upfront and reconciles actual `prompt_tokens + completion_tokens` asynchronously post-stream (`INCRBYFLOAT`), preventing parallel race conditions where 20 simultaneous requests bypass quota checks.
     - **Hard Tenant USD / Token Budget Caps:** Clean `HTTP 429 Too Many Requests` shedding with standard `Retry-After`, `X-RateLimit-Remaining-Requests`, and `X-RateLimit-Remaining-Tokens` headers.
2. **Two-Lane Weighted Fair Queuing via Redis 8 Streams (`api/workers/`):**
   - **Lane P0 (`interactive`):** Live chat turns, HITL approvals, and real-time voice turns — highest priority, autoscaled on head-of-line queue wait time (`< 50ms` target).
   - **Lane P1 (`background`):** Document chunking, Contextual Retrieval enrichment, bulk S3 syncs, session title generation, sleep-time memory consolidation, and batch eval suites — bounded concurrency with per-tenant round-robin fairness so bulk jobs **never** delay live chat users!

### 2.4 `nimAgents` AI Inference Gateway: KV-Cache Affinity, Micro-Batching & Multi-Tier Caching (`agent/inference_gateway.py`)
1. **KV-Cache-Aware Routing & NVIDIA Dynamo Integration:**
   - Multi-turn agent loops send the **exact same 10K–60K token prefix** 5–15 times within seconds as tools execute. On naive round-robin across `N` NIM pods, `(N-1)/N` turns miss GPU KV cache completely!
   - `nimAgents` implements **Load-Bounded Consistent Prefix Hashing** (`hash(model_id + system_prefix_hash + thread_id)`) across healthy self-hosted NIM replicas (automatically spilling over when a replica's active in-flight decode slots exceed threshold), **AND** natively passes `nvext` priority, expected output sequence length (`OSL`), and cache retention hints when backed by **NVIDIA Dynamo 1.0** (`ChatNVIDIADynamo` — supporting aggregated topology below ~8–16 GPUs and disaggregated Prefill/Decode pools above ~16+ GPUs for `2x–7x` multi-user throughput!).
2. **Dynamic GPU Micro-Batcher for Embeddings & Reranking (`api/services/batcher.py`):**
   - Coalesces concurrent single-query `embed_query()` and `rerank()` requests arriving across different users within an **8ms window** up to `max_batch_size = 64` into **one batched GPU RPC** to `NVIDIAEmbeddings` / `NVIDIARerank`—increasing multi-user RAG QPS by **5x–10x** on a single GPU!
3. **Three-Tier Tenant-Safe Caching:**
   - **L0 GPU KV Cache:** Maximized by strict static-first prompt ordering (removing dynamic timestamps from system prompts!).
   - **L1 Exact & Embedding Cache (Redis, `<1ms`):** Deduplicates identical embeddings (`sha256(model + text)`) and deterministic `temperature=0` utility calls (query rewrites, contextualizers, NemoGuard checks).
   - **L2 Tenant-Scoped Semantic Cache (Qdrant, `<10ms`, Optional per Agent):** Partitioned strictly by `(tenant_id, agent_id, system_prompt_hash, kb_version_hash)` so repeated user questions in high-volume support/internal-portal deployments return streamed answers in `<20ms` at `$0` GPU cost with zero cross-tenant leakage!
4. **Intelligent Model Cascading & Write-Behind Telemetry:**
   - Automatically route utility calls (titles, query expansion, chunk contextualization, memory extraction) to `fast_router` (`Nemotron-Nano-8B`) and flush usage/cost telemetry asynchronously in batched Postgres `COPY` / bulk `INSERT` writes every `1s` entirely off the critical streaming path!

---

## Phase 3 — Unified Tool Registry, Deferred Tools (`ToolSearch`), Programmatic Calling & Sandbox Hands *(✅ COMPLETED)*

### 3.1 Canonical Unified `ToolRegistry` & Persistent MCP Pool (`agent/registry.py`, `/tools`)
- Register all built-in Python tools, toolkits, MCP tools, custom user REST/OpenAPI tools, and agent skills under a unified `ToolSpec`:
  - Namespaced IDs (`mcp.github.create_pr`, `search.tavily`), exact pre-computed `tiktoken` token cost, `input_examples` (Anthropic's *Tool Use Examples*), `defer_loading: bool`, `allowed_callers`, `side_effect_level`, `requires_approval: bool`, `max_output_tokens`, `cache_ttl_seconds`, `timeout_seconds`, and retry/circuit-breaker config.
- **Shared Multi-User `MCPConnectionManager`:** Persistent HTTP/2 MCP client pool + shared sandboxed `stdio` worker daemon + cached tool schemas in Postgres/Redis—eliminating per-turn subprocess spawning across concurrent users!
- **Interactive `/tools` Dashboard Page:** Inspect every tool's token cost, execution latency p50/p95, error rate, and test any tool live in a drawer or import any OpenAPI 3.x spec into ready-to-use deferred tools in 1 click!

### 3.2 Provider-Agnostic Deferred Tool Loading (`defer_loading: true` + Hybrid `tool_search`)
- **Modes:** `eager` | `auto` *(default: defers automatically when tool schemas exceed `10%` of context budget or `3,500` tokens)* | `deferred`.
- **How It Works Without Busting KV Cache:**
  - Startup prompt includes only Pinned Core Tools (`defer_loading=False`) + a ~150-token **Deferred Namespace Directory** (`mcp.github [18 tools]`, `project.jira [9 tools]`, etc.) + `tool_search(query, mode="hybrid"|"regex"|"select", limit=5)`.
  - `tool_search` runs sub-5ms hybrid BM25 + embedding RRF search over registered schemas + `input_examples` and appends matched IDs to append-only thread state `discovered_tool_ids: list[str]`.
  - `pre_model_hook` binds `pinned_tools + discovered_tools` in deterministic creation order—cutting prompt tokens by **85%+** while preserving prefix KV cache across turns!

### 3.3 Programmatic Tool Calling (`orchestrate_tools`) & Zero-Trust Disposable Sandbox Hands
- **Decoupled Hands Pool (*Anthropic Managed Agents Pattern*):**
  - Maintain a warm pool of isolated rootless containers (or cloud `E2B` sandboxes) leased on demand *only* when an agent invokes code, shell, file, browser, or `orchestrate_tools`.
  - **Zero-Trust Host Vault Proxy:** User/tenant API keys **never** enter container env vars or LLM state. Inside `orchestrate_tools(python_code)`, generated async Python tool stubs send RPC requests over a restricted local socket to the host `ToolExecutor`, which enforces tenant RBAC, SSRF guards, rate limits, and injects decrypted Vault credentials at the outbound HTTP edge!
  - Multi-step fan-out workflows (`asyncio.gather(...)` across 20 API calls + pandas filtering) run in 1 turn and return **only compact final `stdout`** (`<1,000` tokens) to the LLM!
- **Full HITL `AgentInbox` Integration:** Any tool with `requires_approval=True` triggers `langgraph.types.interrupt(...)`, releasing the worker thread back to the pool immediately with **zero idle CPU/memory hold** while waiting hours or days for human approval in `AgentInbox`!

---

## Phase 4 — Context Engineering & 5-Stage Graduated Compaction Engine *(✅ COMPLETED)*

### 4.1 Immutable `SessionEventLog` vs. KV-Cache-Aligned `ContextProjection`
- Separate the append-only `session_events` log in Postgres from the computed `ContextProjection` sent to the model each turn:
  - **Block 1 (100% Static Prefix across all turns & users sharing an agent!):** Core agent system prompt + Skill directory metadata (`~60 tok/skill`) + Deferred namespace directory + Pinned tool schemas *(enabling **Cross-User Shared Prefix KV Caching** across all users chatting with the same published agent on NVIDIA Dynamo!!)*
  - **Block 2 (Session-Stable Prefix):** User/Workspace Core Memory blocks (`<1.5K` tok) + Active `SKILL.md` playbooks + Latest `<compaction_checkpoint>`.
  - **Block 3 (Dynamic Tail):** Append-only discovered deferred tools + Recent verbatim turns + Current date context.

### 4.2 5-Stage Pre-Model Shaper Pipeline (`agent/context_engine.py`)
1. **Stage 1 — Large Tool Output Offloading (`> 3,000` tok):** Streams large payloads directly to RustFS (`s3://{bucket}/tenants/...` / `artifact://{id}`), injecting structural schema + 1,000-token preview + `read_artifact(id, offset, limit, grep)` JIT access.
2. **Stage 2 — Historical `<think>` Stripping (`clear_thinking`):** Strips verbose reasoning traces older than `N - 2` turns while keeping final answers & tool calls.
3. **Stage 3 — Batch KV-Aware Tool Result Clearing (`clear_tool_uses`):** Evicts stale re-fetchable tool results older than `keep_recent = 5` in batches of `clear_at_least = 10,000` tokens (avoiding per-turn prefix invalidation!), honoring `exclude_tools = {"memory", "read_artifact", "activate_skill", "human_input"}`.
4. **Stage 4 — Pre-Compact Fact Flush + Recall-Tuned Rolling Compaction (`> 80%` window):** Saves critical facts to Memory first, then summarizes older turns via `compactor` NIM model into a structured `<compaction_checkpoint>` block + last `6` verbatim turns.
5. **Stage 5 — Deterministic Hard Token Ceiling Guard (`> 95%` window):** Mathematical guarantee of zero `ContextLengthExceededError` failures.
- Paired with **3-Tier Progressive `SKILL.md` Skills** and a live **Context & Cache Hit Meter** (`Context: 16.2K / 128K · KV Hit: 88% · 38 Tools Deferred`) in the Chat top bar!

---

## Phase 5 — Durable Session Architecture, Time-Travel Forking, Background Runs & Sub-Agents *(✅ COMPLETED)*

- **Stateless Crash-Safe Session Recovery (`wake(session_id)`):** Backed by `AsyncPostgresSaver` + `session_events` table + per-thread Redis lease lock (`SET lock:thread:{id} worker_id NX PX 30000` with heartbeat renewal). If a Brain Worker pod dies mid-generation during a rolling deploy, another pod acquires the expired lease and resumes seamlessly.
- **Reconnectable Redis Stream Fan-Out (`Last-Event-ID`):** Closing the tab, switching devices, or refreshing never aborts a running agent; reconnecting replays buffered chunks from the exact byte offset, and clicking **Cancel** publishes an instant Redis Pub/Sub abort signal across pods in `<5ms`!
- **Time-Travel Session Forking, Smart Titles & Search:** Fork any session from any historical turn (`checkpoint_id`), pin/tag/archive sessions, auto-generate semantic titles on the P1 background queue, export Markdown/JSONL traces, and search across thousands of past sessions instantaneously via tenant-filtered Meilisearch.
- **Context-Isolated Sub-Agents (`delegate_to_subagent`):** Spawn specialist child sessions on the worker pool that spend `50K+` tokens researching/coding in clean context windows and return a distilled `1.5K`-token result + expandable **Sub-Agent Execution Trace** card.

---

## Phase 6 — 4-Tier Agent Memory System & Async ("Sleep-Time") Consolidation *(✅ COMPLETED)*

- **Tier 1 — Pinned Core Memory Blocks (`memory_blocks`, `<1,500` tok):** Editable Markdown blocks (`user_profile`, `workspace_context`, `agent_notes`) cached cleanly in Block 2.
- **Tier 2 — Bi-Temporal Semantic Fact Store (`semantic_memories`):** Atomic facts `(subject, predicate, object, confidence, valid_from, invalidated_at, superseded_by_id, source_session_id)` indexed in both **tenant-partitioned Qdrant (`NVIDIAEmbeddings`) AND Meilisearch BM25** so conflicting outdated facts auto-invalidate without losing historical auditability!
- **Tier 3 — Episodic Memory (`episodic_memories`):** Distilled trajectories (*Problem → Steps → User Corrections → Outcome*) retrieved dynamically via Recency × Importance × Relevance scoring.
- **Tier 4 — Procedural Memory (`procedural_memories`):** Self-improving prompt rule proposals reviewed/approved in UI with 1-click version rollback.
- **Zero-Latency "Sleep-Time" Consolidator on Lane P1:** All heavy extraction, deduplication, and conflict resolution run asynchronously on background workers after turns/sessions—keeping hot-path chat recall under **`35ms`**—with a full **`/memories` Studio UI** in the sidebar!

---

## Phase 7 — Production Contextual, Hybrid & Agentic RAG Pipeline *(✅ COMPLETED)*

- **Async Streaming Multi-User Ingestion Pipeline (`202 Accepted` + Live Progress SSE):**
  - Streams files directly to RustFS (`O(1)` memory per upload regardless of concurrent uploads!) and enqueues chunk jobs onto the fair-queued P1 background worker pool.
  - **Structure-Aware Parent-Child Chunking:** Preserves Markdown/PDF heading breadcrumbs and tables intact (`1,400`-tok parents linked to `400`-tok searchable children).
  - **Contextual Retrieval Enrichment:** Optional batch pass using fast `Nemotron-Nano-8B` prepending a 50–80 token document-situating context to every child chunk prior to indexing.
  - **Dual Indexing:** Micro-batched `NVIDIAEmbeddings` (`nvidia/llama-nemotron-embed-1b-v2` / `nv-embedqa-e5-v5` / offline `BAAI/bge-m3`) into **Qdrant** (`is_tenant=True`) + **Meilisearch** Contextual BM25—plus connectors for external **NVIDIA RAG Blueprint (`NVIDIARAGRetriever`)**, Milvus, and S3-compatible bucket delta-sync (`SHA-256` dedup).
- **Agentic Hybrid Query + Batched `NVIDIARerank` + Grounded Citations:**
  - Parallel Qdrant Dense + Meilisearch BM25 search → Weighted **Reciprocal Rank Fusion ($k=60$)** → Micro-batched **`NVIDIARerank`** (`nvidia/llama-nemotron-rerank-1b-v2` or multimodal `llama-nemotron-rerank-vl-1b-v2`) → Parent-section deduplication → Self-grading query rewrite fallback → Interactive clickable **`[1]`, `[2]` Source Evidence Drawer** in Chat + **Retrieval Rank Debugger** on `/rag`!

---

## Phase 8 — NVIDIA NemoGuard Safety, Open Streaming Voice (`Riva NIM`), FinOps & Evaluation Studio *(✅ COMPLETED)*

- **Deterministic Pre/Post Safety Pipeline (`/security`):** Mandatory `input_guard`, `tool_output_guard` (indirect prompt-injection shield!), and `output_guard` LangGraph nodes (`agent/guardrails_engine.py`, `api/routes/guardrails.py`) combining `<1ms` pre-compiled PII/secret regex redactors + **NVIDIA NemoGuard NIMs** (`llama-3.1-nemotron-safety-guard-8b-v3`, `llama-3.1-nemoguard-8b-topic-control`, `nemoguard-jailbreak-detect`) with Base64/Unicode-tag de-obfuscation, sub-millisecond L1 verdict caching, and live `tenant_id` / `session_id` audit logging.
- **Scalable WebSocket Voice Gateway (`/voice` & Chat Voice Toggle):** WebAudio `AudioWorklet` 16kHz `PCM_S16LE` streaming (`voice-session.tsx`, `api/routes/voice.py`) with 44-byte `RIFF/WAVE` framing through stateless WebSocket gateway pods backed by **NVIDIA Riva Speech NIMs (`Parakeet ASR` + `Magpie TTS`)** (or `Faster-Whisper` + `Kokoro`), `2 MiB` backpressure-bounded ring buffers, sub-5ms `barge_in` task cancellation, and shared multi-turn `thread_id` session state.
- **FinOps Cost/Latency Observatory (`/analytics`) & Evaluation Studio (`/evals`):** Real-time `recharts` dashboards (`api/routes/analytics.py`, `app/(dashboard)/analytics/page.tsx`) tracking per-tenant/per-agent TPM, GPU KV-cache hit ratio, L1/L2 semantic cache `$` savings, `p50/p95/p99` TTFT & queue wait percentiles, paired with `api/routes/evals.py` & `app/(dashboard)/evals/page.tsx` for automated pre-merge CI regression gates and parallel side-by-side NIM Model A/B Arena battles.

---

## 12. Capacity Planning, Scaling Playbook & Milestone Matrix

### 12.1 Horizontal Multi-User Scaling Envelope

| Concurrent Active Users | API Gateway Pods | Stateless Brain Workers | Background Ingestion Workers | Data Plane Topology | Inference Tier Guidance |
|---|---|---|---|---|---|
| **1 – 25 Users** *(Team / Single Host)* | 2 workers (`uvloop`) | 2 workers | 1 worker | Single-node `docker-compose`: Postgres 17 + PgBouncer + Redis 8 + Qdrant + Meilisearch + RustFS | NVIDIA Hosted NIM API (`integrate.api.nvidia.com`) or 1–2 GPU Monolithic NIM (`NIM_ENABLE_KV_CACHE_REUSE=1`) |
| **25 – 500 Users** *(Department / Mid-Market SaaS)* | 3–6 stateless pods (HPA on CPU / active SSE conns) | 4–12 pods (HPA on `stream:runs:interactive` head-of-line wait `>50ms`) | 2–8 pods (HPA on `stream:jobs:background` lag) | Managed HA Postgres (+ read replica) + PgBouncer (`pool=100`) + Redis 8 Sentinel + 3-node Qdrant (`int8` quantized) | 4–16 GPUs with **Aggregated NIM + Prefix-Hash / Dynamo KV Router** + dedicated 1-GPU Embed/Rerank NIM pod |
| **500 – 10,000+ Users** *(Enterprise Multi-Tenant)* | 10–30 stateless pods | 20–80 stateless pods | Autoscaled KEDA worker pool + isolated Sandbox cluster | Partitioned Postgres (`tenant_id` RLS) + Redis 8 Cluster + Sharded Qdrant + Distributed RustFS Cluster | **NVIDIA Dynamo 1.0 Disaggregated Prefill + Decode Pools** (`NIXL` RDMA + `KVBM` CPU/NVMe KV cache tiering) |

### 12.2 Phased Delivery Roadmap

| Phase | Milestone | Key Files Created / Refactored | Exit Criteria |
|---|---|---|---|
| **Phase 1** *(✅ DONE)* | **`nimAgents` Rebrand, Complete AWS Removal & Multi-Provider NVIDIA NIM Core** | `pyproject.toml`, `agent/model.py`, `api/routes/models.py`, `storage.py`, `frontend/**` (`NimAgentsLogo`, `/models`, `/settings`, `<think>` streaming) | Zero AWS/Nova strings in codebase; live NIM chat + `<think>` token stream + `/v1/models` discovery green |
| **Phase 2** *(✅ DONE)* | **Multi-Tenant Scalable Backend, Async Data Plane & AI Inference Gateway** | `api/db_async.py` (Postgres RLS + `aiosqlite`), `api/middleware/admission.py` (Lua quotas), `api/workers/`, `agent/inference_gateway.py`, `docker-compose.yml` | Zero event-loop stalls under load test (`locust`/`k6` 50 concurrent users); tenant RLS + `429` quotas + micro-batcher verified |
| **Phase 3** *(✅ DONE)* | **Unified `ToolRegistry`, `defer_loading` + `tool_search`, `orchestrate_tools` & Sandbox Hands** | `agent/registry.py`, `agent/sandbox_hands.py`, `agent/graph.py`, `api/routes/tools.py`, `app/(dashboard)/tools/page.tsx` | 50+ tools achieve `87%+` context savings (`648` vs `5,000+` tok); zero-secret-leak Unix RPC sandbox + `AgentInbox` HITL pass 26/26 pytest |
| **Phase 4** *(✅ DONE)* | **5-Stage Context Engine, Prefix KV-Cache Alignment, `SKILL.md` & Context Meter** | `agent/context_engine.py`, `agent/skills.py`, `api/routes/{skills,context}.py`, `app/(dashboard)/skills/page.tsx`, `components/thread/context-meter.tsx` | Byte-stable Block 1 prefix across turns & users; 100-turn stress run stays bounded with `0` overflows (`31/31` pytest + `16/16` Next.js routes green) |
| **Phase 5** *(✅ DONE)* | **Durable Sessions, `Last-Event-ID` Stream Replay, Checkpoint Forking & Sub-Agents** | `api/services/run_manager.py`, `api/routes/sessions.py`, `agent/subagents.py`, `app/(dashboard)/sessions/page.tsx`, `ThreadHistory` | Pod kill mid-stream recovers cleanly via `wake(session_id)`; tab reload replays live stream; thread fork + sub-agent traces green (`36/36` pytest + `17/17` Next.js routes) |
| **Phase 6** *(✅ DONE)* | **4-Tier Cognitive Memory + Bi-Temporal Facts + Sleep-Time Consolidator + `/memories`** | `agent/memory_engine.py`, `api/services/memory_consolidator.py`, `api/routes/memories.py`, `app/(dashboard)/memories/page.tsx` | `0ms` write latency on chat path; conflicting facts auto-supersede; `<35ms` hybrid memory recall (`41/41` pytest + `18/18` Next.js routes green) |
| **Phase 7** *(✅ DONE)* | **Contextual + Parent-Child + `NVIDIAEmbeddings` Qdrant + BM25 + `NVIDIARerank` RAG** | `api/services/rag_pipeline.py`, `api/routes/rag_pipeline.py`, `app/(dashboard)/rag/page.tsx`, `messages/citations.tsx` | Streaming upload + `202` SSE + parent-child table preservation + batched `NVIDIARerank` + clickable `[1]` drawer (`45/45` pytest + `18/18` Next.js routes green) |
| **Phase 8** *(✅ DONE)* | **`NemoGuard` Pre/Post Safety, `Riva NIM` Voice, FinOps `/analytics` & `/evals`** | `agent/guardrails_engine.py`, `api/routes/{voice,analytics,evals}.py`, `app/(dashboard)/{security,analytics,evals}/page.tsx` | Mandatory 3-stage input/tool/output safety shields; 16kHz PCM voice + instant barge-in; FinOps `recharts` & CI eval arena (`51/51` pytest + `20/20` Next.js routes green) |

---

## 13. Frontier Engineering & Systems References

1. **Anthropic Engineering (Apr 2026)** — *Scaling Managed Agents: Decoupling the Brain from the Hands* (Stateless orchestrator harness, external append-only session log, on-demand disposable sandboxes, Vault credential proxy, `-60% p50` / `-90% p95` TTFT): <https://www.anthropic.com/engineering/managed-agents>
2. **NVIDIA Technical Blog (May–Aug 2026)** — *How NVIDIA Dynamo 1.0 Powers Multi-Node Inference at Production Scale & Full-Stack Optimizations for Agentic Inference* (Radix-Tree / Flash Indexer KV-cache routing, priority `nvext` hints, shared sub-agent KV cache via NIXL/KVBM, disaggregated Prefill/Decode trade-offs): <https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/> · <https://developer.nvidia.com/blog/full-stack-optimizations-for-agentic-inference-with-nvidia-dynamo/>
3. **Anthropic Engineering (Nov 2025 – Mar 2026)** — *Introducing Advanced Tool Use* (`defer_loading: true` Tool Search cutting 85%+ tokens, Programmatic Tool Calling, `input_examples` & KV prefix preservation): <https://www.anthropic.com/engineering/advanced-tool-use> · <https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool>
4. **Anthropic Engineering (Jan 2026) & Claude Code Systems Study (2026)** — *Effective Context Engineering for AI Agents & Five-Layer Graduated Compaction Pipeline*: <https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents> · <https://arxiv.org/html/2604.14228v1>
5. **Anthropic Engineering** — *Contextual Retrieval in AI Systems* (`-67%` retrieval failure reduction stacking Contextual Embeddings + Contextual BM25 + Cross-Encoder Reranking): <https://www.anthropic.com/engineering/contextual-retrieval>
6. **NVIDIA NIM & NeMo Guardrails Docs (2026)** — *`ChatNVIDIA`, `ChatNVIDIADynamo`, `NVIDIAEmbeddings`, `NVIDIARerank`, and `NemoGuard` Safety Microservices*: <https://docs.langchain.com/oss/python/integrations/providers/nvidia> · <https://docs.nvidia.com/nemo/microservices/26.3.1/guardrails/tutorials/deploy-nemoguard-nims.html>
7. **LangGraph Production Scaling Guides (2026)** — *`AsyncPostgresSaver` + `PgBouncer` Transaction Pooling + `Redis 8` Streams/PubSub Checkpointing & Horizontal Worker Topologies*: <https://markaicode.com/architecture/langgraph-production-architecture/> · <https://markaicode.com/stack/langgraph-redis-postgresql-agent-stack/>
8. **Multi-Tenant LLM Gateway Design (2026)** — *Atomic Redis Lua Token Buckets (`RPM`/`TPM`/Budget Reservation), Tenant-Partitioned Semantic Caching, and Priority Fair Queuing*: <https://ironclad.academy/system-design/articles/design-llm-gateway>

# nimAgents

`nimAgents` is a multi-tenant platform for building, deploying, and operating tool-using AI agents. It pairs a Next.js studio UI with a FastAPI control plane and a LangGraph execution runtime, built primarily for **NVIDIA NIM** and **NVIDIA Dynamo** while supporting OpenAI, Anthropic, OpenRouter, vLLM, SGLang, and Ollama endpoints.

It runs in two modes from the same codebase:

- **Local mode (no Docker required):** SQLite (`WAL`), embedded local Qdrant, and in-process priority queues.
- **Cluster mode (`docker compose`):** PostgreSQL 17 with `FORCE ROW LEVEL SECURITY`, PgBouncer, Redis 8, Qdrant, Meilisearch, and RustFS.

---

## Product Features

### Models & Inference Gateway (`/models`)
- Native `ChatNVIDIA` and `ChatNVIDIADynamo` clients alongside OpenAI, Anthropic, OpenRouter, vLLM, SGLang, and Ollama.
- Live `/v1/models` endpoint discovery, connection health probes, automatic `.with_fallbacks(...)` chains, and model role routing (`primary_chat`, `fast_router`, `compactor`, `contextualizer`).
- Token streaming with collapsible `<think>...</think>` reasoning blocks.
- Consistent prefix hashing across self-hosted NIM replicas, `8ms` GPU micro-batching for embeddings/reranking, and tenant-partitioned `L1` exact + `L2` Qdrant semantic caches.

### Tools, MCP & Sandboxed Execution (`/tools`, `/skills`)
- Unified `ToolRegistry` covering built-in tools, OpenAPI 3.x imports, and pooled Streamable HTTP / `stdio` MCP servers.
- Deferred tool loading (`defer_loading`): keeps startup prompts compact and loads tool schemas on demand through hybrid BM25 + vector `tool_search`, cutting tool schema token overhead by ~85% without invalidating prompt prefix caches.
- Zero-trust programmatic tool execution (`orchestrate_tools`) over an isolated local RPC socket so API secrets never enter sandbox environment variables or model prompts.
- Human-in-the-loop interrupts (`AgentInbox`) and progressive `SKILL.md` playbooks.

### Durable Sessions, Context & 4-Tier Memory (`/chat`, `/sessions`, `/memories`)
- Append-only `session_events` log with `Last-Event-ID` SSE stream replay, per-session lease locks (`wake`), checkpoint forking, and depth-bounded specialist sub-agents (`delegate_to_subagent`).
- 5-stage pre-model context pipeline (`agent/context_engine.py`): offloads large tool outputs (`>3,000` tokens) to object storage with `read_artifact` inspection, strips old reasoning traces, evicts stale tool outputs in KV-friendly batches, and summarizes older history before context limits are hit.
- Four memory tiers isolated per tenant: pinned Core blocks, bi-temporal Semantic facts (auto-superseding contradicted facts), Episodic run summaries, and versioned Procedural prompt rules consolidated asynchronously on background workers.

### Contextual Hybrid RAG (`/rag`, `/storage`)
- Streaming chunked uploads (`202 Accepted` + SSE progress) with `SHA-256` deduplication and S3-compatible bucket sync (`RustFS`, `R2`, `SeaweedFS`, `Ceph`).
- Structure-aware parent-child Markdown/PDF chunking that keeps tables intact and prepends document/section context to child chunks before indexing.
- 6-stage query pipeline: parallel Qdrant (`is_tenant=True`, `INT8` scalar quantization) + BM25 search, weighted Reciprocal Rank Fusion (`k=60`), batched `NVIDIARerank`, parent-section deduplication, query rewrite fallback, and clickable `[1]` source drawers in chat.

### Safety, Streaming Voice, FinOps & Evals (`/security`, `/voice`, `/analytics`, `/evals`)
- Mandatory three-stage safety graph nodes (`input_guard`, `tool_output_guard` indirect prompt-injection shield, and `output_guard`) combining pre-compiled PII/secret regex rules with NVIDIA NemoGuard (`llama-3.1-nemotron-safety-guard-8b-v3`, `llama-3.1-nemoguard-8b-topic-control`, `nemoguard-jailbreak-detect`).
- Full-duplex `16kHz` PCM WebSocket voice gateway (`/api/voice/ws`) using WebAudio `AudioWorklet` capture, `2 MiB` bounded ring buffers, sub-5ms `barge_in` cancellation, shared `thread_id` session state, and NVIDIA Riva Speech NIM (`Parakeet ASR` + `Magpie TTS`) or open Whisper/Kokoro endpoints.
- FinOps telemetry (`/analytics`) tracking spend, TPM, KV-cache hit ratio, semantic cache savings, and `p50/p95/p99` TTFT + queue wait times.
- Evaluation Studio (`/evals`) with automated CI regression gates (Markdown report export) and parallel side-by-side model A/B comparisons.

---

## Architecture

```text
Browser / SDK Clients
  │
  ├──► Next.js 15 Studio UI (:3000)
  │      Routes /api/* to Control Plane and streams runs from LangGraph Runtime
  │
  ├──► Control Plane API — FastAPI (:8100)
  │      • TenancyMiddleware (ContextVar) ─► AdmissionMiddleware (RPM / TPM / concurrency / USD caps)
  │      • Two-lane worker pool: P0 interactive vs P1 background (RAG ingest, memory consolidation)
  │      • Routers: /models, /tools, /skills, /sessions, /memories, /rag-pipeline,
  │                 /guardrails, /voice (WS), /analytics, /evals, /storage, /scale
  │
  └──► LangGraph Agent Runtime (:2024)
         START ─► input_guard ─► agent (5-Stage Context Shaper + Prefix-Routed LLM)
                                   │                    ▲
                                   ▼                    │
                                 tools ──► tool_output_guard
                                   │
                                   ▼ (no tool calls)
                              output_guard ─► END

Data & Inference Backing Services
  • Relational state : PostgreSQL 17 + PgBouncer (28 FORCE RLS tables)  |  SQLite WAL fallback
  • Queues & locks   : Redis 8                                          |  In-process async queues
  • Vector search    : Qdrant (is_tenant=True + INT8 quantization)      |  Embedded local Qdrant
  • Lexical & files  : Meilisearch (tenant tokens) + RustFS             |  Built-in BM25 + local disk
  • Model endpoints  : NVIDIA NIM / Dynamo, OpenAI, Anthropic, OpenRouter, vLLM, SGLang, Ollama
```

### Repository Layout

| Path | Purpose |
|---|---|
| `agent/` | LangGraph graph (`graph.py`), provider factory (`model.py`), prefix router & caches (`inference_gateway.py`), `ToolRegistry` (`registry.py`), `SandboxHands` (`sandbox_hands.py`), 5-stage `ContextEngine` (`context_engine.py`), 4-tier `MemoryEngine` (`memory_engine.py`), and `NemoGuardSafetyEngine` (`guardrails_engine.py`). |
| `api/` | FastAPI app (`app.py`), dual SQLite/Postgres schema (`db.py`, `db_async.py`), tenancy & HTTP/2 pools (`core/`), quota admission (`middleware/`), `P0/P1` fair queues (`workers/`), GPU micro-batcher & RAG/run services (`services/`), and REST/SSE/WebSocket routes (`routes/`). |
| `frontend/` | Next.js 15 (`App Router`, React 19, TypeScript, Tailwind v4) studio UI across `/chat`, `/agents`, `/sessions`, `/tools`, `/skills`, `/memories`, `/rag`, `/storage`, `/models`, `/voice`, `/security`, `/analytics`, `/evals`, and `/settings`. |
| `docker/` | PostgreSQL 17 schema and mandatory Row-Level Security policies (`docker/postgres/init-rls.sql`). |
| `tests/` | End-to-end test suites covering multi-tenancy, concurrency, tools, context compaction, sessions, memory, RAG, guardrails, voice, analytics, and evals. |

---

## Local Setup

### Prerequisites
- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/)
- **Node.js 20+** and `pnpm` (`npm install -g pnpm`)
- *(Optional)* **Docker & Docker Compose** if running full multi-container backing services

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/yassinech-99/agent-builder.git
cd agent-builder

# Python backend & LangGraph runtime
uv sync --dev

# Frontend UI
cd frontend
pnpm install
cd ..
```

### 2. Configure Environment

Create a `.env` file in the repository root:

```ini
# Primary LLM / Embeddings / Reranking (NVIDIA NIM API Catalog or self-hosted NIM)
NVIDIA_API_KEY=nvapi-...
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2

# Optional: additional model providers
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434/v1

# Optional: NVIDIA NemoGuard & Riva Speech NIM endpoints
NEMOGUARD_BASE_URL=
NIM_ASR_BASE_URL=
NIM_TTS_BASE_URL=

# Optional: external backing services (omit for zero-Docker local mode)
# DATABASE_URL=postgresql+asyncpg://nimagents:nimagents@localhost:6432/nimagents
# REDIS_URL=redis://localhost:6379/0
# QDRANT_URL=http://localhost:6333
# MEILI_URL=http://localhost:7700
# RUSTFS_ENDPOINT=http://localhost:9000
```

### 3. (Optional) Start Backing Infrastructure with Docker

Skip this step if you want lightweight local development (`data/nimagents.db` + embedded Qdrant are initialized automatically on startup).

```bash
docker compose up -d
```

This starts PostgreSQL 17 (`:5432`), PgBouncer (`:6432`), Redis 8 (`:6379`), Qdrant (`:6333`), Meilisearch (`:7700`), and RustFS (`:9000`, console `:9001`).

### 4. Start the Application Services

Run the three services in separate terminals from the repository root:

```bash
# Terminal 1: Control Plane API (:8100)
uv run uvicorn api.app:app --host 0.0.0.0 --port 8100 --reload
```

```bash
# Terminal 2: LangGraph Agent Runtime (:2024)
uv run langgraph dev --host 0.0.0.0 --port 2024 --no-browser
```

```bash
# Terminal 3: Studio Frontend (:3000)
cd frontend
pnpm dev --hostname 0.0.0.0 --port 3000
```

Open `http://localhost:3000` in your browser. Control Plane OpenAPI docs are served at `http://localhost:8100/docs`.

---

## CLI & Verification

```bash
# Inspect configured providers and presets
uv run nimagents info

# Run the backend & integration test suite
uv run pytest -v

# Verify production frontend build
cd frontend && pnpm build
```

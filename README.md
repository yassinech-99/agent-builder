# nimAgents

**nimAgents** is an open, multi-provider enterprise AI agent studio centered on **NVIDIA NIM & NVIDIA Dynamo** with first-class support for OpenAI, Anthropic, OpenRouter, vLLM, SGLang, and Ollama.

## Key Capabilities

- **NVIDIA NIM & Dynamo Core** — First-class `ChatNVIDIA` & `ChatNVIDIADynamo` integration supporting Hosted NVIDIA API Catalog (`integrate.api.nvidia.com`) and self-hosted NIM clusters (`Nemotron-3`, `Llama-3.3-Nemotron`, `DeepSeek-R1`, `Qwen3`), live `/v1/models` endpoint discovery, health checks, role routing, and fallback chains.
- **Open Multi-Provider Freedom** — Mix and match NVIDIA NIM, OpenAI, Anthropic, OpenRouter, vLLM, SGLang, and Ollama per agent or role (`primary_chat`, `fast_router`, `compactor`, `contextualizer`).
- **Real-Time Streaming & `<think>` Reasoning Traces** — Token-level SSE streaming with collapsible live reasoning trace accordions and KV-cache-aligned static system prefixes.
- **Hybrid Knowledge & Reranking** — External NIM RAG endpoints, Local/Object-Store Vector RAG (`NVIDIAEmbeddings` + `NVIDIARerank` + FAISS/Qdrant), and Lexical BM25 (`Meilisearch`).
- **Open S3-Compatible Storage** — Native `minio` connector supporting MinIO, Cloudflare R2, SeaweedFS, Ceph, and standard S3-compatible buckets with secret masking.
- **Visual Agent Builder, Guardrails & Streaming Voice** — Multi-step agent wizard with 30+ prebuilt tools, streamable HTTP MCP discovery, regex/PII/secret guardrails monitor, and WebSocket voice sessions.

## Environment Variables

Create a local `.env` file (not committed):

```ini
# Primary NVIDIA NIM Configuration
NVIDIA_API_KEY=nvapi-...
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
NVIDIA_EMBED_MODEL=nvidia/llama-nemotron-embed-1b-v2
NVIDIA_RERANK_MODEL=nvidia/llama-nemotron-rerank-1b-v2

# Optional Additional Model Providers
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434/v1

# Data Services (Meilisearch / Qdrant / MinIO)
MEILI_URL=http://localhost:7700
MEILI_MASTER_KEY=
QDRANT_URL=http://localhost:6333
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Optional Tool API Keys
TAVILY_API_KEY=...
```

## Quickstart

### 1. Start Data Services (Optional Docker Stack)

```bash
docker compose up -d
```

### 2. Backend & Runtime

```bash
# Terminal 1 — LangGraph Agent Runtime (port 2024)
uv run langgraph dev --host 0.0.0.0 --port 2024

# Terminal 2 — nimAgents Control Plane API (port 8100)
uv run uvicorn api.app:app --host 0.0.0.0 --port 8100
```

### 3. Frontend Studio (port 3000)

```bash
cd frontend
pnpm install
pnpm dev --hostname 0.0.0.0 --port 3000
```

Open **nimAgents** at `http://localhost:3000`.

## Running Tests

```bash
uv run pytest -v
```

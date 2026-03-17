# NovaGrid Agents

A Nova-native enterprise agent workspace built on Amazon Nova for reasoning, retrieval, browser automation, and optional voice interaction.

## Features

- **Nova 2 Lite Runtime** -- Amazon Nova 2 Lite powers agent reasoning via `ChatBedrockConverse`
- **Nova Multimodal Embeddings** -- Vector RAG with Amazon Nova embeddings + FAISS
- **Nova Act Browser Tool** -- UI workflow automation via the Nova Act SDK
- **S3 Knowledge Ingestion** -- Import documents directly from S3 buckets for RAG
- **Agent Builder** -- Visual wizard with tools, MCP servers, and model selection
- **Hybrid Knowledge** -- Bedrock Knowledge Bases, Local RAG (FAISS), and Lexical RAG (Meilisearch)
- **Dashboard** -- Chat, Agents, Models, RAG, and Settings in one workspace

## Architecture (high level)

NovaGrid Agents is split into:

- **LangGraph runtime** (port **2024**) — builds a per-agent ReAct graph (`agent/graph.py`)
- **Builder API** (FastAPI, port **8100**) — models, RAG sources, tool catalog, storage connections (`api/app.py`)
- **Frontend** (Next.js, port **3000**) — dashboard UI (`frontend/`)
- **Meilisearch** (port **7700**) — required for Lexical RAG (runs via Docker)
- **SQLite** — local persistence at `data/gridagents.db` (created at runtime; do not commit)

## Prerequisites

- **Python 3.11+** and [`uv`](https://github.com/astral-sh/uv)
- **Node.js 18+** and `pnpm`
- **Docker** (for Meilisearch)
- **AWS credentials** with access to Amazon Bedrock models (Nova 2 Lite + embeddings)

## Environment variables

Create a local `.env` (do not commit it):

```text
# AWS / Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Default chat model
BEDROCK_MODEL=amazon.nova-2-lite-v1:0

# Optional tools
NOVA_ACT_API_KEY=...
TAVILY_API_KEY=...

# Lexical RAG (Meilisearch)
MEILI_URL=http://localhost:7700
# MEILI_MASTER_KEY=...   # only if you set one in docker-compose
```

## Quickstart (Docker prerequisites)

### Start Meilisearch (required for Lexical RAG)

PowerShell:

```powershell
docker compose up -d
```

bash:

```bash
docker compose up -d
```

Meilisearch should be available at `http://localhost:7700`.

## Quickstart (local dev)

### Terminal 1 — LangGraph runtime (port 2024)

PowerShell:

```powershell
uv run langgraph dev
```

bash:

```bash
uv run langgraph dev
```

### Terminal 2 — Builder API (port 8100)

PowerShell:

```powershell
uv run uvicorn api.app:app --host 0.0.0.0 --port 8100
```

bash:

```bash
uv run uvicorn api.app:app --host 0.0.0.0 --port 8100
```

### Terminal 3 — Frontend (port 3000)

PowerShell:

```powershell
cd frontend
pnpm install
pnpm dev
```

bash:

```bash
cd frontend
pnpm install
pnpm dev
```

Open the UI at `http://localhost:3000`.

## Smoke test checklist (2 minutes)

1. **Builder API**: open `http://localhost:8100/health` → should return `ok`
2. **Tool catalog**: open `http://localhost:8100/api/tools/catalog` → should return JSON
3. **Frontend**: open `http://localhost:3000`
4. Go to **RAG**:
   - Bedrock Knowledge Bases: register a KB ID and test query
   - Local RAG: upload a small document and test query
   - Lexical RAG: upload a doc and test keyword query
5. Go to **Chat**:
   - open **Add Internal Knowledge** and select sources
   - ask a question that requires retrieval

## Notes

- `.env`, `data/*.db`, `.langgraph_api/*`, and `frontend/.next/` are intentionally ignored by git.
- If you don’t have Meilisearch running, **Lexical RAG** will be unavailable, but the rest of the app should still start.

## Architecture (components)

| Layer | Tech |
|-------|------|
| LLM Runtime | `ChatBedrockConverse` (Nova 2 Lite) via `langchain-aws` |
| Embeddings | `BedrockEmbeddings` via `langchain-aws` |
| Managed RAG | Bedrock Knowledge Bases (`bedrock-agent-runtime.retrieve()`) |
| Browser | `nova-act` SDK |
| Orchestration | LangGraph ReAct agent |
| Local Vector Store | FAISS |
| Lexical Search | Meilisearch |
| Backend | FastAPI + SQLite |
| Frontend | Next.js 15 / React 19 / Tailwind v4 |

## Submission

- **Category:** Agentic AI
- **Hashtag:** #AmazonNova

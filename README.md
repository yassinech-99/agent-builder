# NovaGrid Agents

A Nova-native enterprise agent workspace built on Amazon Nova 

## Features

- **Nova 2 Lite Runtime** -- Amazon Nova 2 Lite powers agent reasoning via `ChatBedrockConverse`
- **Nova Multimodal Embeddings** -- Vector RAG with Amazon Nova embeddings + FAISS
- **Nova Act Browser Tool** -- UI workflow automation via the Nova Act SDK
- **S3 Knowledge Ingestion** -- Import documents directly from S3 buckets for RAG
- **Agent Builder** -- Visual wizard with tools, MCP servers, and model selection
- **Hybrid Knowledge** -- Bedrock Knowledge Bases, Local RAG (FAISS), and  BM25 (Meilisearch)
- **Dashboard** -- Chat, Agents, Models, RAG, and Settings in one workspace


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


```bash
docker compose up -d
```

Meilisearch should be available at `http://localhost:7700`.

## Quickstart (local dev)

### Terminal 1 — LangGraph runtime (port 2024)



```bash
uv run langgraph dev
```

### Terminal 2 — Builder API (port 8100)



```bash
uv run uvicorn api.app:app --host 0.0.0.0 --port 8100
```

### Terminal 3 — Frontend (port 3000)


```bash
cd frontend
npm run build
npm run dev
```

Open the UI at `http://localhost:3000`.

"""Comprehensive Phase 1 Verification Suite for nimAgents."""

from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA, ChatNVIDIADynamo

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_nimagents.db"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-placeholder-key-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-placeholder-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-placeholder-key")
    from api import db

    db.init_db()
    return db_file


@pytest.fixture()
def client(isolated_db):
    from api.app import app

    with TestClient(app) as c:
        yield c


# ─── 1. Full Repo Zero-AWS / Complete Rebranding Verification ──────────


def test_zero_aws_and_complete_rebranding_across_codebase():
    forbidden_patterns = [
        r"\bboto3\b",
        r"\bbotocore\b",
        r"\bminio\b",
        r"\bMinIO\b",
        r"\bMINIO_",
        r"langchain[-_]aws",
        r"nova[-_]act",
        r"\bNovaGrid\b",
        r"\bnovagrid\b",
        r"\bGridAgentsLogo\b",
        r"\bChatBedrock",
        r"\bBedrockEmbeddings\b",
        r"\bAmazonKnowledgeBasesRetriever\b",
        r"s3-import",
        r"bedrock_kb",
    ]
    combined = re.compile("|".join(forbidden_patterns))

    skip_dirs = {
        ".git",
        ".venv",
        "node_modules",
        ".next",
        "__pycache__",
        ".pytest_cache",
        "data",
        "tests",
    }
    skip_files = {"plan.md", "uv.lock", "pnpm-lock.yaml"}

    offenders: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.name in skip_files:
            continue
        if path.suffix in {".png", ".jpg", ".ico", ".woff", ".woff2", ".db"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        match = combined.search(text)
        if match:
            offenders.append(
                f"{path.relative_to(REPO_ROOT)} -> matched '{match.group(0)}'"
            )

    assert offenders == [], f"Found forbidden legacy references: {offenders}"
    assert not (REPO_ROOT / "frontend" / "package-lock.json").exists()
    assert not (
        REPO_ROOT / "frontend" / "src" / "components" / "icons" / "grid-agents.tsx"
    ).exists()
    assert (
        REPO_ROOT / "frontend" / "src" / "components" / "icons" / "nim-agents.tsx"
    ).exists()


def test_nvidia_sdk_version_patched_past_cve():
    ver = importlib.metadata.version("langchain-nvidia-ai-endpoints")
    major, minor, *_ = [int(x) for x in ver.split(".")[:2]]
    assert (major, minor) >= (1, 2), f"Expected patched >=1.2.x version, got {ver}"


# ─── 2. Multi-Provider NIM / Dynamo Factory & Security Sanitizer ───────


def test_multi_provider_model_instantiation(isolated_db):
    from agent.model import create_model_from_config

    nim_llm = create_model_from_config(
        {
            "provider": "nvidia_nim",
            "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "nvapi-unit-test-secret-key-999999",
            "temperature": 0.4,
            "max_tokens": 4096,
        }
    )
    assert isinstance(nim_llm, ChatNVIDIA)

    dynamo_llm = create_model_from_config(
        {
            "provider": "nvidia_dynamo",
            "model_name": "nvidia/nemotron-3-super-120b-a12b",
            "base_url": "http://localhost:8000/v1",
            "api_key": "",
            "temperature": 0.2,
            "max_tokens": 2048,
        },
        priority=8,
    )
    assert isinstance(dynamo_llm, ChatNVIDIADynamo)

    for prov, model_name, base_url in [
        ("openai", "gpt-4o", ""),
        ("anthropic", "claude-sonnet-4-5", ""),
        ("openrouter", "deepseek/deepseek-r1", "https://openrouter.ai/api/v1"),
        ("vllm", "Qwen/Qwen3-32B", "http://localhost:8000/v1"),
        ("ollama", "qwen3:8b", "http://localhost:11434/v1"),
    ]:
        inst = create_model_from_config(
            {
                "provider": prov,
                "model_name": model_name,
                "base_url": base_url,
                "api_key": "test-key",
                "temperature": 0.5,
                "max_tokens": 1024,
            }
        )
        assert inst is not None


def test_multimodal_uri_sanitizer_blocks_path_traversal():
    from agent.model import sanitize_multimodal_messages

    safe_png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    msgs = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Inspect these images"},
                {"type": "image_url", "image_url": {"url": safe_png}},
                {"type": "image_url", "image_url": {"url": "https://nvidia.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "../../../etc/passwd"}},
                {"type": "image_url", "image_url": {"url": "file:///etc/shadow"}},
            ]
        )
    ]

    cleaned = sanitize_multimodal_messages(msgs)
    blocks = cleaned[0].content
    assert isinstance(blocks, list)
    image_blocks = [b for b in blocks if b.get("type") == "image_url"]
    assert len(image_blocks) == 2  # ONLY safe base64 PNG + HTTPS image survive!
    assert "passwd" not in str(blocks)
    assert "shadow" not in str(blocks)


# ─── 3. Control Plane API CRUD, Masking, Discovery, Fallbacks & WS ─────


def test_models_crud_secret_masking_and_fallback_resolution(client):
    presets_res = client.get("/api/models/presets")
    assert presets_res.status_code == 200
    presets = presets_res.json()
    assert len(presets["presets"]) >= 7
    assert "nvidia_nim" in presets["providers"]

    # Create fallback model
    fb_res = client.post(
        "/api/models",
        json={
            "name": "Local Ollama Fallback",
            "provider": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model_name": "qwen3:8b",
            "role": "fast_router",
        },
    )
    assert fb_res.status_code == 201
    fb_id = fb_res.json()["id"]

    # Create primary default NVIDIA NIM model with secret API key
    raw_secret = "nvapi-SUPER-SECRET-PRODUCTION-KEY-9876543210"
    primary_res = client.post(
        "/api/models",
        json={
            "name": "Nemotron Primary",
            "provider": "nvidia_nim",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": raw_secret,
            "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "is_default": True,
            "role": "primary_chat",
            "fallback_model_id": fb_id,
            "supports_thinking": True,
        },
    )
    assert primary_res.status_code == 201
    body = primary_res.json()
    # Critical security check: raw key must NEVER appear in API JSON responses
    assert raw_secret not in str(body)
    assert body["api_key"] == ""
    assert body["has_api_key"] is True
    assert "••••" in body["api_key_masked"]

    # Updating name with blank api_key preserves saved secret key
    upd = client.put(
        f"/api/models/{body['id']}",
        json={
            "name": "Nemotron Primary v2",
            "provider": "nvidia_nim",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "",
            "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "is_default": True,
            "role": "primary_chat",
            "fallback_model_id": fb_id,
            "supports_thinking": True,
        },
    )
    assert upd.status_code == 200
    assert upd.json()["has_api_key"] is True

    # Verify resolve_model resolves default model + attaches fallback runnable
    from agent.model import resolve_model

    resolved = resolve_model()
    assert hasattr(resolved, "fallbacks")
    assert len(resolved.fallbacks) == 1


def test_live_model_discovery_endpoint(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "data": [
            {"id": "nvidia/llama-3.3-nemotron-super-49b-v1.5", "owned_by": "nvidia"},
            {"id": "deepseek-ai/deepseek-r1", "owned_by": "deepseek-ai"},
        ]
    }
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    with patch("api.app.get_http_client", return_value=mock_client):
        res = client.post(
            "/api/models/discover",
            json={
                "provider": "nvidia_nim",
                "base_url": "https://integrate.api.nvidia.com/v1",
                "api_key": "nvapi-test",
            },
        )
    assert res.status_code == 200
    ids = [m["id"] for m in res.json()]
    assert "deepseek-ai/deepseek-r1" in ids
    assert "nvidia/llama-3.3-nemotron-super-49b-v1.5" in ids


def test_storage_rag_and_guardrails_endpoints(client):
    # Storage bucket registration with masked secrets
    st = client.post(
        "/api/storage",
        json={
            "name": "Local RustFS",
            "endpoint_url": "http://localhost:9000",
            "bucket_name": "nimagents-corpus",
            "access_key": "nimagentsadmin",
            "secret_key": "supersecretrustfskey",
            "skip_validation": True,
        },
    )
    assert st.status_code == 201
    assert "supersecretrustfskey" not in str(st.json())

    # NIM RAG registration
    rag = client.post(
        "/api/rag/register",
        json={
            "name": "Engineering KB",
            "endpoint_url": "http://localhost:8081/v1",
            "collection_name": "eng_docs",
            "ranking_model": "nvidia/llama-nemotron-rerank-1b-v2",
        },
    )
    assert rag.status_code == 201
    assert rag.json()["collection_name"] == "eng_docs"

    # Guardrails detect NVIDIA NIM nvapi-* secret leaks out of the box
    gr = client.post(
        "/api/guardrails/check",
        json={"text": "Here is my key nvapi-abcdefghijklmnopqrstuvwxyz123456"},
    )
    assert gr.status_code == 200
    data = gr.json()
    assert data["passed"] is False
    assert "[REDACTED]" in data["redacted_text"]


def test_voice_websocket_streams_reply_cleanly(client):
    class FakeStreamingLLM:
        async def astream(self, _messages):
            yield AIMessageChunk(content="<think> planning </think>Hello from ")
            yield AIMessageChunk(content="nimAgents Voice!")

    with patch("api.routes.voice.get_model", return_value=FakeStreamingLLM()):
        with client.websocket_connect("/api/voice/ws") as ws:
            initial = ws.receive_json()
            assert initial == {"type": "status", "state": "listening"}
            ws.send_json({"type": "text", "data": "Hi there"})
            thinking = ws.receive_json()
            assert thinking["state"] == "thinking"
            transcript = ws.receive_json()
            assert transcript["type"] == "transcript"
            assert transcript["text"] == "Hello from nimAgents Voice!"
            back_to_listen = ws.receive_json()
            assert back_to_listen["state"] == "listening"


@pytest.mark.asyncio
async def test_langgraph_agent_compiles_with_kv_cache_stable_prompt(isolated_db):
    from agent.graph import DEFAULT_SYSTEM_PROMPT, make_graph

    assert "datetime" not in DEFAULT_SYSTEM_PROMPT
    graph = await make_graph(
        {
            "configurable": {
                "agent_name": "TestAgent",
                "enabled_prebuilt_tools": ["calculator"],
            }
        }
    )
    assert graph is not None

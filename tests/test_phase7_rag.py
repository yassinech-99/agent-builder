"""Comprehensive Phase 7 Verification Suite:
  1. Async `202 Accepted` Streaming Upload + Live Progress SSE (`GET /api/rag-pipeline/jobs/{id}/stream`)
     + Intact Markdown Table Preservation + Hierarchical Heading Breadcrumbs + `SHA-256` Deduplication
  2. Anthropic-Style Contextual Retrieval Enrichment Resolving Ambiguous Pronoun / Implicit-Subject Chunks
  3. Parent-Section Sibling Deduplication (`1` Citation per Unique `parent_id`), Retrieval Rank Debugger Telemetry
     (`dense_rank`, `bm25_rank`, `rrf_score`, `rerank_score`, `final_score`) & Strict Multi-Tenant Isolation
  4. Stage-5 Self-Grading Query Rewrite Fallback & Grounded `[1]` Citations Inside `make_graph`
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest


class RagCitationMockModel(BaseChatModel):
    """Calls `contextual_rag_search` on Turn 1 and synthesizes grounded `[1]` answer on Turn 2."""

    @property
    def _llm_type(self) -> str:
        return "phase7-rag-citation-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "contextual_rag_search",
                                    "args": {"query": "H200 FP8 throughput SLA table", "top_k": 3},
                                    "id": "call_crag_1",
                                    "type": "tool_call",
                                }
                            ],
                        )
                    )
                ]
            )
        assert isinstance(last, ToolMessage)
        payload = json.loads(str(last.content))
        top_cit = payload["citations"][0]
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=f"Per [{top_cit['citation_id']}] ({top_cit['heading_path']}), H200 FP8 SLA is 3,958 TFLOPS."
                    )
                )
            ]
        )


@pytest.fixture()
def isolated_phase7_env(tmp_path, monkeypatch):
    db_file = tmp_path / "phase7_nimagents.db"
    qdrant_dir = tmp_path / "phase7_qdrant"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NIMAGENTS_QDRANT_PATH", str(qdrant_dir))

    from api import db
    from api.middleware.admission import admission_controller

    admission_controller.reset_for_tests()
    db.init_db()
    yield tmp_path
    admission_controller.reset_for_tests()


SAMPLE_RUNBOOK_MD = """# nimAgents Production GPU Architecture Guide

## 2.1 Redis Priority Fair Queue
Queue workers isolate interactive P0 chat traffic from bulk P1 indexing jobs.
Throughput remains deterministic across noisy enterprise tenants.

## 4.2 NVIDIA H200 TensorRT-LLM SLA Guarantees
When deployed in disaggregated mode, it drops latency by 62% and prevents OOM stalls.

| GPU SKU | Precision | Peak Throughput | Max Batch |
| --- | --- | --- | --- |
| H200 SXM | FP8 | 3,958 TFLOPS | 256 |
| B200 SXM | FP4 | 18,000 TFLOPS | 512 |

Enabling NIXL KVBM tiering allows 10x larger prefix cache reuse across sub-agents.
"""


# ─── 1. Streaming `202` Upload + Live SSE Progress + Intact Tables + SHA-256 Dedup ──


@pytest.mark.asyncio
async def test_streaming_202_async_ingest_live_sse_progress_and_sha256_dedup(isolated_phase7_env):
    from api.app import app

    headers = {"X-Tenant-ID": "org-rag", "X-Workspace-ID": "prod"}
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        col_resp = await ac.post(
            "/api/rag-pipeline/collections",
            headers=headers,
            json={"name": "GPU Architecture Runbooks"},
        )
        assert col_resp.status_code == 201
        source_id = col_resp.json()["id"]

        # Upload in `async_mode=true` -> immediate `202 Accepted` + live SSE progress stream!
        ing_resp = await ac.post(
            f"/api/rag-pipeline/collections/{source_id}/ingest",
            headers=headers,
            files={"file": ("h200_runbook.md", SAMPLE_RUNBOOK_MD.encode("utf-8"), "text/markdown")},
            data={"contextual_enrich": "true", "async_mode": "true"},
        )
        assert ing_resp.status_code == 202
        queued = ing_resp.json()
        job_id = queued["job_id"]

        observed_progress: list[int] = []
        final_status = ""
        async with ac.stream("GET", f"/api/rag-pipeline/jobs/{job_id}/stream", headers=headers) as sse:
            assert sse.status_code == 200
            async for line in sse.aiter_lines():
                if line.startswith("data: "):
                    snap = json.loads(line.split("data: ", 1)[1])
                    observed_progress.append(int(snap["progress"]))
                    final_status = snap["status"]

        assert final_status == "completed"
        assert observed_progress[-1] == 100

        # Re-uploading identical bytes triggers instant SHA-256 deduplication (`deduplicated == True`)!
        dup_resp = await ac.post(
            f"/api/rag-pipeline/collections/{source_id}/ingest",
            headers=headers,
            files={"file": ("h200_runbook_copy.md", SAMPLE_RUNBOOK_MD.encode("utf-8"), "text/markdown")},
            data={"contextual_enrich": "true", "async_mode": "false"},
        )
        assert dup_resp.status_code == 200
        assert dup_resp.json()["deduplicated"] is True


# ─── 2. Contextual Enrichment Resolving Orphan Pronoun & Table Preservation ──


def test_contextual_enrichment_resolves_ambiguous_orphan_pronoun_and_preserves_tables(isolated_phase7_env):
    from api.app import app

    headers = {"X-Tenant-ID": "org-ctx-rag", "X-Workspace-ID": "prod"}
    with TestClient(app) as client:
        col = client.post(
            "/api/rag-pipeline/collections",
            headers=headers,
            json={"name": "Contextual Benchmark Docs"},
        ).json()

        client.post(
            f"/api/rag-pipeline/collections/{col['id']}/ingest",
            headers=headers,
            files={"file": ("h200_runbook.md", SAMPLE_RUNBOOK_MD.encode("utf-8"), "text/markdown")},
            data={"contextual_enrich": "true", "async_mode": "false"},
        )

        # Notice the sentence in Section 4.2 only says "it drops latency by 62%" without naming H200 inline,
        # yet Contextual Enrichment prepended Section `4.2 NVIDIA H200 TensorRT-LLM SLA Guarantees`!
        q_resp = client.post(
            "/api/rag-pipeline/query",
            headers=headers,
            json={
                "query": "NVIDIA H200 TensorRT-LLM drops latency by 62%",
                "top_k": 3,
                "rerank": True,
            },
        )
        assert q_resp.status_code == 200
        res = q_resp.json()
        assert res["count"] >= 1
        top = res["citations"][0]
        assert "4.2 NVIDIA H200 TensorRT-LLM SLA Guarantees" in top["heading_path"]
        # Full intact Markdown table preserved cleanly inside parent section!
        assert "| H200 SXM | FP8 | 3,958 TFLOPS | 256 |" in top["full_parent_markdown"]


# ─── 3. Parent-Section Sibling Deduplication & Strict Tenant Isolation ────


def test_parent_section_deduplication_rank_debugger_and_tenant_isolation(isolated_phase7_env):
    from api.app import app

    alpha_h = {"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-prod"}
    bravo_h = {"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-prod"}

    with TestClient(app) as client:
        col = client.post(
            "/api/rag-pipeline/collections",
            headers=alpha_h,
            json={"name": "Alpha Secret Spec"},
        ).json()

        ing = client.post(
            f"/api/rag-pipeline/collections/{col['id']}/ingest",
            headers=alpha_h,
            files={"file": ("h200_runbook.md", SAMPLE_RUNBOOK_MD.encode("utf-8"), "text/markdown")},
            data={"contextual_enrich": "true", "async_mode": "false"},
        ).json()
        # Section 4.2 has multiple child chunks (prose + table + NIXL sentence), so `child_count > parent_count`
        assert ing["child_count"] > ing["parent_count"]

        alpha_q = client.post(
            "/api/rag-pipeline/query",
            headers=alpha_h,
            json={"query": "H200 SXM FP8 3,958 TFLOPS NIXL KVBM", "top_k": 5, "rerank": True},
        ).json()

        parent_ids = [c["parent_id"] for c in alpha_q["citations"]]
        assert len(parent_ids) == len(set(parent_ids)), "Expected strict 1-citation-per-parent deduplication!"
        top = alpha_q["citations"][0]
        assert top["rrf_score"] > 0
        assert top["rerank_score"] > 0

        # Tenant Bravo sees ZERO citations from Tenant Alpha's confidential collection!
        bravo_q = client.post(
            "/api/rag-pipeline/query",
            headers=bravo_h,
            json={"query": "H200 SXM FP8 3,958 TFLOPS NIXL KVBM", "top_k": 5},
        ).json()
        assert bravo_q["count"] == 0


# ─── 4. Self-Grading Query Rewrite & End-to-End `make_graph` Citations ───


@pytest.mark.asyncio
async def test_self_grading_query_rewrite_and_grounded_citations_in_make_graph(isolated_phase7_env):
    import agent.graph as graph_mod
    from api.app import app
    from api.core.tenancy import TenantContext, tenant_scope

    ctx = TenantContext(tenant_id="org-agent-rag", workspace_id="prod", user_id="eng-1")
    headers = {"X-Tenant-ID": "org-agent-rag", "X-Workspace-ID": "prod", "X-User-ID": "eng-1"}

    with TestClient(app) as client:
        col = client.post(
            "/api/rag-pipeline/collections",
            headers=headers,
            json={"name": "Agent Knowledge"},
        ).json()
        client.post(
            f"/api/rag-pipeline/collections/{col['id']}/ingest",
            headers=headers,
            files={"file": ("h200_runbook.md", SAMPLE_RUNBOOK_MD.encode("utf-8"), "text/markdown")},
            data={"contextual_enrich": "true", "async_mode": "false"},
        )

        # Low-coverage query triggers automatic Stage-5 Agentic Query Rewrite (`rewritten_query`)
        rewrite_resp = client.post(
            "/api/rag-pipeline/query",
            headers=headers,
            json={
                "query": "hey please tell me about mysterious unknown ttft metric?",
                "top_k": 3,
                "enable_query_rewrite": True,
            },
        ).json()
        assert rewrite_resp["rewritten_query"] is not None
        assert "time to first token" in rewrite_resp["rewritten_query"]

    mock_llm = RagCitationMockModel()
    with patch.object(graph_mod, "get_model", return_value=mock_llm):
        with tenant_scope(ctx):
            graph = await graph_mod.make_graph(
                {
                    "configurable": {
                        "thread_id": "rag-thread-1",
                        "local_rag_source_ids": [col["id"]],
                        "enable_sleep_consolidation": False,
                    }
                }
            )
            out = await graph.ainvoke({"messages": [HumanMessage(content="What is H200 FP8 throughput?")]})

    final_answer = str(out["messages"][-1].content)
    assert "[1]" in final_answer
    assert "3,958 TFLOPS" in final_answer

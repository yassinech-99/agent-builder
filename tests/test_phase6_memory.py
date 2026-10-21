"""Comprehensive Phase 6 Verification Suite:
  1. Tier-1 Pinned Core Memory Blocks (`<= 1,500` tok enforcement) + Block 1 Static KV Prefix Preservation
  2. Tier-2 Bi-Temporal Semantic Fact Store (`semantic_memories`):
     - Exact triple deduplication / confidence reinforcement
     - Atomic contradiction invalidation (`invalidated_at` + `superseded_by_id` lineage)
     - Point-in-time historical truth reconstruction (`?as_of=...`)
  3. `< 35ms` Multi-Tenant Hybrid Recall (`Qdrant is_tenant=True Dense + BM25 Lexical RRF`) & Zero Cross-Tenant Leakage
  4. Tier-4 Self-Improving Procedural Rules (`proposed` -> `active`) & 1-Click Version Rollback
  5. Zero-Latency "Sleep-Time" Consolidator on Lane P1 Inside `make_graph` (`0ms` Hot Chat Stall!)
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest


class CapturingMemoryMockModel(BaseChatModel):
    """Captures exact `[Block 1 SystemMessage, Block 2 SessionPrefix, ...]` calls sent by `make_graph`."""

    captured_calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "phase6-memory-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.captured_calls.append(list(messages))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content="Recorded your H200 Dynamo cluster preferences.")
                )
            ]
        )


@pytest.fixture()
def isolated_phase6_env(tmp_path, monkeypatch):
    db_file = tmp_path / "phase6_nimagents.db"
    qdrant_dir = tmp_path / "phase6_qdrant"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NIMAGENTS_QDRANT_PATH", str(qdrant_dir))

    from api import db
    from api.middleware.admission import admission_controller

    admission_controller.reset_for_tests()
    db.init_db()
    yield tmp_path
    admission_controller.reset_for_tests()


# ─── 1. Tier-1 Core Memory Ceiling & Cross-User Block 1 KV Cache Identity ─


@pytest.mark.asyncio
async def test_tier1_core_blocks_token_budget_and_block1_kv_prefix_preservation(isolated_phase6_env):
    import agent.graph as graph_mod
    from api.app import app

    user_a_h = {"X-Tenant-ID": "org-nim", "X-Workspace-ID": "prod", "X-User-ID": "alice"}
    user_b_h = {"X-Tenant-ID": "org-nim", "X-Workspace-ID": "prod", "X-User-ID": "bob"}

    with TestClient(app) as client:
        # Oversized core block (> 1,500 tokens) is rejected deterministically with HTTP 422
        too_big = client.post(
            "/api/memories/blocks",
            headers=user_a_h,
            json={
                "block_type": "user_profile",
                "title": "Oversized Profile",
                "content": "token " * 1900,
            },
        )
        assert too_big.status_code == 422

        # Valid user-specific core blocks for Alice vs Bob
        ok_a = client.post(
            "/api/memories/blocks",
            headers=user_a_h,
            json={
                "block_type": "user_profile",
                "title": "Alice Principal SRE",
                "content": "Manages 64x NVIDIA H200 GPUs running Dynamo 1.0 in Algiers.",
            },
        )
        ok_b = client.post(
            "/api/memories/blocks",
            headers=user_b_h,
            json={
                "block_type": "user_profile",
                "title": "Bob FinOps Lead",
                "content": "Tracks per-tenant TPM budgets and FP8 cost efficiency.",
            },
        )
        assert ok_a.status_code == 201 and ok_b.status_code == 201

    # Verify BOTH users share 100% BYTE-IDENTICAL `Block 1` (`messages[0]`) while `Block 2` personalizes cleanly!
    from api.core.tenancy import TenantContext, tenant_scope

    mock_model = CapturingMemoryMockModel(captured_calls=[])
    with patch.object(graph_mod, "get_model", return_value=mock_model):
        with tenant_scope(TenantContext("org-nim", "prod", "alice")):
            g_alice = await graph_mod.make_graph({"configurable": {"thread_id": "t-alice", "enable_sleep_consolidation": False}})
            await g_alice.ainvoke({"messages": [HumanMessage(content="Hello")]})

        with tenant_scope(TenantContext("org-nim", "prod", "bob")):
            g_bob = await graph_mod.make_graph({"configurable": {"thread_id": "t-bob", "enable_sleep_consolidation": False}})
            await g_bob.ainvoke({"messages": [HumanMessage(content="Hello")]})

    block1_alice = str(mock_model.captured_calls[0][0].content)
    block1_bob = str(mock_model.captured_calls[1][0].content)
    assert block1_alice == block1_bob  # 100% shared GPU KV prefix across users!
    assert "64x NVIDIA H200 GPUs" in str(mock_model.captured_calls[0][1].content)
    assert "per-tenant TPM budgets" in str(mock_model.captured_calls[1][1].content)


# ─── 2. Bi-Temporal Fact Contradiction Superseding & Point-in-Time Audit ──


def test_tier2_bitemporal_contradiction_superseding_dedup_and_as_of_time_travel(isolated_phase6_env):
    from api.app import app

    headers = {"X-Tenant-ID": "org-bitemporal", "X-Workspace-ID": "prod", "X-User-ID": "eng-1"}

    with TestClient(app) as client:
        # Monday (`t1`): Initial architecture fact
        r1 = client.post(
            "/api/memories/facts",
            headers=headers,
            json={
                "subject": "workspace",
                "predicate": "primary inference cluster",
                "object": "vLLM 0.5 on 4x A100 GPUs",
                "valid_from": "2026-10-19T09:00:00+00:00",
            },
        )
        assert r1.status_code == 201
        fact_v1 = r1.json()
        assert fact_v1["deduplicated"] is False

        # Tuesday: Re-stating exact same triple deduplicates & reinforces confidence instead of duplicating!
        r_dup = client.post(
            "/api/memories/facts",
            headers=headers,
            json={
                "subject": "workspace",
                "predicate": "primary inference cluster",
                "object": "vLLM 0.5 on 4x A100 GPUs",
                "valid_from": "2026-10-20T09:00:00+00:00",
            },
        )
        assert r_dup.json()["deduplicated"] is True
        assert r_dup.json()["id"] == fact_v1["id"]

        # Wednesday (`t2`): Migration contradicts & atomically supersedes `fact_v1`!
        r2 = client.post(
            "/api/memories/facts",
            headers=headers,
            json={
                "subject": "workspace",
                "predicate": "primary inference cluster",
                "object": "NVIDIA Dynamo 1.0 on 16x H200 GPUs",
                "valid_from": "2026-10-21T12:00:00+00:00",
            },
        )
        assert r2.status_code == 201
        fact_v2 = r2.json()
        assert fact_v2["superseded_ids"] == [fact_v1["id"]]

        # Default hot query returns ONLY the active truth (`fact_v2`)
        active_now = client.get("/api/memories/facts", headers=headers).json()
        assert [f["id"] for f in active_now] == [fact_v2["id"]]

        # Point-in-time query on Tuesday (`2026-10-20T15:00:00+00:00`) reconstructs Monday's exact truth (`fact_v1`)!
        tuesday_truth = client.get(
            "/api/memories/facts",
            params={"as_of": "2026-10-20T15:00:00+00:00"},
            headers=headers,
        ).json()
        assert [f["id"] for f in tuesday_truth] == [fact_v1["id"]]
        assert "4x A100" in tuesday_truth[0]["object"]

        # Full historical lineage returns both active + invalidated record with `superseded_by_id` pointer
        lineage = client.get("/api/memories/facts?include_historical=true", headers=headers).json()
        assert len(lineage) == 2
        old_rec = next(f for f in lineage if f["id"] == fact_v1["id"])
        assert old_rec["superseded_by_id"] == fact_v2["id"]
        assert old_rec["invalidated_at"] == "2026-10-21T12:00:00+00:00"


# ─── 3. `< 35ms` Hybrid Qdrant + BM25 Recall & Strict Tenant Isolation ───


def test_sub_35ms_multitenant_hybrid_qdrant_bm25_recall_and_zero_cross_tenant_leakage(isolated_phase6_env):
    from api.app import app

    alpha_h = {"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-1", "X-User-ID": "u1"}
    bravo_h = {"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-1", "X-User-ID": "u2"}

    with TestClient(app) as client:
        for i in range(12):
            client.post(
                "/api/memories/facts",
                headers=alpha_h,
                json={
                    "subject": f"service-{i}",
                    "predicate": "uses cache tier",
                    "object": (
                        "Qdrant INT8 co-located HNSW shard with NIXL KVBM"
                        if i == 7
                        else f"standard worker pool {i}"
                    ),
                },
            )
        client.post(
            "/api/memories/episodes",
            headers=alpha_h,
            json={
                "summary": "Tuned NIXL KVBM cache transfer latency down by 62%",
                "problem": "Sub-agent cold KV cache misses",
                "approach": "Enabled NIXL shared prefix blocks across Dynamo workers",
                "key_tool_steps": ["delegate_to_subagent", "calculator"],
                "outcome": "p95 TTFT dropped from 940ms to 350ms",
            },
        )

        # Warm Qdrant shard once, then measure steady-state hot-path hybrid recall (< 35ms target!)
        client.post("/api/memories/recall", headers=alpha_h, json={"query": "warmup", "limit": 4})

        resp_alpha = client.post(
            "/api/memories/recall",
            headers=alpha_h,
            json={"query": "NIXL KVBM shared HNSW cache latency", "limit": 5},
        )
        assert resp_alpha.status_code == 200
        body_alpha = resp_alpha.json()
        assert body_alpha["count"] >= 2
        assert body_alpha["latency_ms"] < 35.0, f"Expected <35ms recall, got {body_alpha['latency_ms']}ms"
        top_texts = " ".join(r["text"] for r in body_alpha["results"][:2])
        assert "NIXL KVBM" in top_texts

        # Tenant Bravo sees ZERO memories from Tenant Alpha!
        resp_bravo = client.post(
            "/api/memories/recall",
            headers=bravo_h,
            json={"query": "NIXL KVBM shared HNSW cache latency", "limit": 5},
        ).json()
        assert resp_bravo["count"] == 0


# ─── 4. Tier-4 Procedural Approval Workflow & 1-Click Version Rollback ────


def test_tier4_procedural_approval_workflow_and_one_click_version_rollback(isolated_phase6_env):
    from api.app import app

    headers = {"X-Tenant-ID": "org-proc", "X-Workspace-ID": "prod"}
    with TestClient(app) as client:
        # 1. Propose v1 (`proposed` -> NOT injected into prompt until human approval!)
        v1 = client.post(
            "/api/memories/procedural",
            headers=headers,
            json={
                "title": "Structured Executive Tables",
                "rule_prompt": "Always format multi-GPU comparisons as Markdown tables with FP8 TFLOPS.",
                "status": "proposed",
            },
        ).json()
        assert v1["version"] == 1 and v1["status"] == "proposed"

        # 2. Approve (`v2`, `status == 'active'`)
        v2 = client.patch(
            f"/api/memories/procedural/{v1['id']}",
            headers=headers,
            json={"status": "active"},
        ).json()
        assert v2["version"] == 2 and v2["status"] == "active"

        # 3. Bad edit (`v3`) introduced later
        v3 = client.patch(
            f"/api/memories/procedural/{v1['id']}",
            headers=headers,
            json={"rule_prompt": "BROKEN EXPERIMENTAL PROMPT RULE"},
        ).json()
        assert v3["version"] == 3

        # 4. Instant 1-Click Rollback back to `target_version = 2`!
        rb = client.post(
            f"/api/memories/procedural/{v1['id']}/rollback",
            headers=headers,
            json={"target_version": 2},
        )
        assert rb.status_code == 200
        restored = rb.json()
        assert restored["version"] == 4
        assert restored["status"] == "active"
        assert "FP8 TFLOPS" in restored["rule_prompt"]


# ─── 5. Zero-Latency "Sleep-Time" Consolidator on Lane P1 in `make_graph` ──


@pytest.mark.asyncio
async def test_zero_latency_sleep_time_consolidator_on_lane_p1_inside_make_graph(isolated_phase6_env):
    import agent.graph as graph_mod
    from api.app import app
    from api.core.tenancy import TenantContext, tenant_scope
    from api.services.memory_consolidator import memory_consolidator

    ctx = TenantContext(tenant_id="org-sleep", workspace_id="prod", user_id="architect-1")
    mock_model = CapturingMemoryMockModel(captured_calls=[])

    with patch.object(graph_mod, "get_model", return_value=mock_model):
        with tenant_scope(ctx):
            graph = await graph_mod.make_graph({"configurable": {"thread_id": "sleep-sess-1"}})
            t0 = time.perf_counter()
            await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                "We migrated our primary inference cluster to NVIDIA H200 Dynamo in Algiers. "
                                "Always include citation IDs in every architecture review."
                            )
                        )
                    ]
                }
            )
            turn_wall_ms = (time.perf_counter() - t0) * 1000.0
            assert turn_wall_ms < 150.0  # Zero blocking LLM extraction on hot path!

            # Drain Lane-P1 background consolidation job cleanly
            await memory_consolidator.flush_inflight()

    headers = {"X-Tenant-ID": "org-sleep", "X-Workspace-ID": "prod", "X-User-ID": "architect-1"}
    with TestClient(app) as client:
        summary = client.get("/api/memories/summary", headers=headers).json()
        assert summary["active_semantic_facts"] >= 1
        assert summary["episodic_memories_count"] == 1
        assert summary["procedural_proposed_count"] >= 1

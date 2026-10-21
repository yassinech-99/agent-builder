"""Comprehensive Phase 4 Verification Suite:
  1. Cross-User & Cross-Turn Byte-Identical Block 1 Prefix KV-Cache Alignment (`prefix_hash`)
  2. Stage 1 Large Tool Output Offloading (`context_artifacts`), Structural Schema Inspection & JIT `read_artifact(..., grep=...)`
  3. Stage 2 Historical `<think>` & `reasoning_content` Stripping Preserving Latest 2 Turns
  4. Stage 3 KV-Aware Batch Stale Tool Clearing (`clear_at_least_tokens`) & `PROTECTED_TOOL_NAMES` Exemption
  5. Stage 4 & Stage 5 100-Turn Long-Horizon Run: Pre-Compact Fact Extraction, Rolling `<compaction_checkpoint>`,
     Atomic `[AI+ToolCalls, ToolMessage]` Integrity & Hard 95% Token Ceiling Guarantee (`0` Overflows!)
  6. 3-Tier Progressive `SKILL.md` YAML Frontmatter Import, Multi-Tenant Isolation & Live `make_graph` `activate_skill` Execution
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class InspectingChatModel(BaseChatModel):
    """Captures exact `list[BaseMessage]` sent to the model on every turn."""

    seen_calls: list[list[BaseMessage]] = []
    step_fn: Any = None

    @property
    def _llm_type(self) -> str:
        return "inspecting-phase4-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_calls.append(list(messages))
        reply: AIMessage = self.step_fn(messages, self.seen_calls)
        return ChatResult(generations=[ChatGeneration(message=reply)])


@pytest.fixture()
def isolated_phase4_env(tmp_path, monkeypatch):
    db_file = tmp_path / "phase4_nimagents.db"
    qdrant_dir = tmp_path / "phase4_qdrant"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NIMAGENTS_QDRANT_PATH", str(qdrant_dir))

    from api import db
    from api.middleware.admission import admission_controller

    admission_controller.reset_for_tests()
    db.init_db()
    yield tmp_path
    admission_controller.reset_for_tests()


# ─── 1. Cross-User & Cross-Turn Byte-Identical Block 1 KV Prefix ──────────


@pytest.mark.asyncio
async def test_cross_user_and_cross_turn_block1_kv_prefix_identity(isolated_phase4_env):
    from unittest.mock import patch
    import agent.graph as graph_mod

    llm_alice = InspectingChatModel(step_fn=lambda _m, _c: AIMessage(content="Hi Alice"))
    llm_bob = InspectingChatModel(step_fn=lambda _m, _c: AIMessage(content="Hi Bob"))

    with patch.object(graph_mod, "get_model", side_effect=[llm_alice, llm_bob]):
        g_alice = await graph_mod.make_graph(
            {
                "configurable": {
                    "x-tenant-id": "acme-ai",
                    "x-workspace-id": "prod",
                    "x-user-id": "alice",
                    "thread_id": "thread-alice-1",
                    "agent_name": "Shared Dynamo Agent",
                    "system_prompt": "You are {agent_name}. Ref date: {system_time}.",
                    "session_memory": "User Alice prefers concise tables.",
                }
            }
        )
        g_bob = await graph_mod.make_graph(
            {
                "configurable": {
                    "x-tenant-id": "acme-ai",
                    "x-workspace-id": "prod",
                    "x-user-id": "bob",
                    "thread_id": "thread-bob-1",
                    "agent_name": "Shared Dynamo Agent",
                    "system_prompt": "You are {agent_name}. Ref date: {system_time}.",
                    "session_memory": "User Bob prefers Python code blocks.",
                }
            }
        )
        await g_alice.ainvoke({"messages": [HumanMessage(content="Hello from Alice")]})
        await g_bob.ainvoke({"messages": [HumanMessage(content="Hello from Bob")]})

    alice_msgs = llm_alice.seen_calls[0]
    bob_msgs = llm_bob.seen_calls[0]

    # Block 1 (`SystemMessage[0]`) is 100.0% BYTE-IDENTICAL between Alice and Bob -> Shared Dynamo GPU KV Cache!
    assert isinstance(alice_msgs[0], SystemMessage)
    assert alice_msgs[0].content == bob_msgs[0].content
    # Per-user memory lives cleanly in Block 2 (`SystemMessage[1]`) without polluting Block 1!
    assert "Alice prefers concise tables" in alice_msgs[1].content
    assert "Bob prefers Python code blocks" in bob_msgs[1].content


# ─── 2. Stage 1: Large Tool Output Offloading & JIT `read_artifact` Grep ──


@pytest.mark.asyncio
async def test_stage1_large_tool_artifact_offload_and_jit_grep_isolation(isolated_phase4_env):
    from agent.context_engine import ContextPolicy, context_engine
    from api.app import app
    from api.core.tenancy import TenantContext

    alpha = TenantContext(tenant_id="tenant-alpha", workspace_id="ws-1")
    huge_payload = json.dumps(
        [
            {
                "pod_id": f"nim-worker-{i:03d}",
                "status": "CRITICAL_OOM_SPIKE" if i == 137 else "healthy",
                "kv_blocks_free": 4096 - i,
                "notes": "Telemetry window sample " * 8,
            }
            for i in range(220)
        ],
        indent=2,
    )

    proj = await context_engine.shape_context(
        static_system_prompt="You are nimAgents SRE.",
        messages=[
            HumanMessage(content="Check cluster pods"),
            AIMessage(content="", tool_calls=[{"name": "list_pods", "args": {}, "id": "call_1"}]),
            ToolMessage(content=huge_payload, tool_call_id="call_1", name="list_pods"),
        ],
        policy=ContextPolicy(offload_threshold_tokens=1500, preview_tokens=300),
        ctx=alpha,
        thread_id="sre-thread-1",
    )

    assert len(proj.offloaded_artifact_ids) == 1
    art_id = proj.offloaded_artifact_ids[0]
    shaped_tool_msg = proj.block3_messages[-1]
    assert "JSON array with 220 `dict` records" in shaped_tool_msg.content
    assert proj.dynamic_tail_tokens < 1200  # Down from >10,000 raw tokens!

    # Verify HTTP `/api/context/artifacts/{id}?grep=...` works for Alpha and blocks Bravo with 404!
    with TestClient(app) as client:
        grep_resp = client.get(
            f"/api/context/artifacts/{art_id}",
            params={"grep": "CRITICAL_OOM_SPIKE"},
            headers={"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-1"},
        )
        assert grep_resp.status_code == 200
        assert "CRITICAL_OOM_SPIKE" in grep_resp.json()["output"]

        denied = client.get(
            f"/api/context/artifacts/{art_id}",
            headers={"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-1"},
        )
        assert denied.status_code == 404


# ─── 3. Stage 2 & Stage 3: `<think>` Stripping & Batch Tool Eviction ──────


@pytest.mark.asyncio
async def test_stage2_thinking_stripping_and_stage3_batch_eviction(isolated_phase4_env):
    from agent.context_engine import ContextPolicy, context_engine

    history: list[BaseMessage] = []
    for turn in range(6):
        history.append(HumanMessage(content=f"Question {turn}"))
        history.append(
            AIMessage(
                content=f"<think>{'Deep reasoning trace step. ' * 90}</think>Final answer {turn}",
                tool_calls=[{"name": "read_artifact" if turn == 0 else "web_scrape", "args": {}, "id": f"c_{turn}"}],
            )
        )
        history.append(
            ToolMessage(
                content=(f"Result {turn}: " + ("scraped payload words " * 250)),
                tool_call_id=f"c_{turn}",
                name="read_artifact" if turn == 0 else "web_scrape",
            )
        )

    # First check WITH high `clear_at_least_tokens=50_000`: Stage 3 holds off to preserve KV cache!
    proj_no_evict = await context_engine.shape_context(
        static_system_prompt="Static Prefix",
        messages=history,
        policy=ContextPolicy(
            offload_threshold_tokens=50_000,
            keep_recent_thinking_turns=2,
            keep_recent_tool_messages=2,
            clear_at_least_tokens=50_000,
        ),
    )
    assert proj_no_evict.cleared_thinking_tokens > 400
    assert proj_no_evict.cleared_tool_tokens == 0
    # Oldest AI message had `<think>` stripped, latest AI message kept `<think>` intact!
    ai_msgs = [m for m in proj_no_evict.block3_messages if isinstance(m, AIMessage)]
    assert "<think>" not in ai_msgs[0].content
    assert "Final answer 0" in ai_msgs[0].content
    assert "<think>" in ai_msgs[-1].content

    # Now with `clear_at_least_tokens=800`: batch eviction fires AND spares protected `read_artifact`!
    proj_evicted = await context_engine.shape_context(
        static_system_prompt="Static Prefix",
        messages=history,
        policy=ContextPolicy(
            offload_threshold_tokens=50_000,
            keep_recent_thinking_turns=2,
            keep_recent_tool_messages=2,
            clear_at_least_tokens=800,
        ),
    )
    assert proj_evicted.cleared_tool_tokens > 800
    tool_msgs = [m for m in proj_evicted.block3_messages if isinstance(m, ToolMessage)]
    # Turn 0 was `read_artifact` (PROTECTED!) -> preserved verbatim!
    assert "scraped payload words" in tool_msgs[0].content
    # Turn 1..3 were `web_scrape` -> cleared in batch!
    assert tool_msgs[1].content.startswith("[Cleared stale tool result from `web_scrape`")


# ─── 4. Stage 4 & Stage 5: 100-Turn Stress Test (`0` Context Overflows!) ──


@pytest.mark.asyncio
async def test_100_turn_long_horizon_agent_stays_bounded_with_zero_overflows(isolated_phase4_env):
    from agent.context_engine import ContextPolicy, context_engine

    window = 4096
    hard_cap = int(window * 0.95)
    policy = ContextPolicy(
        context_window=window,
        offload_threshold_tokens=900,
        preview_tokens=180,
        clear_at_least_tokens=1000,
        soft_compact_ratio=0.80,
        hard_ceiling_ratio=0.95,
        keep_verbatim_turn_groups=4,
    )

    rolling_checkpoint = ""
    raw_log: list[BaseMessage] = []
    compactions_triggered = 0

    for turn in range(1, 101):
        raw_log.append(
            HumanMessage(
                content=(
                    f"Turn {turn}: Remember architecture budget = $18,500 USD and target latency = 35ms. "
                    + ("Context load filler tokens. " * 25)
                )
            )
        )
        raw_log.append(
            AIMessage(
                content=f"<think>{'Reasoning chain ' * 30}</think>",
                tool_calls=[{"name": "metrics_probe", "args": {"step": turn}, "id": f"tc_{turn}"}],
            )
        )
        raw_log.append(
            ToolMessage(
                content=f"Probe step {turn} telemetry: " + ("kv_cache_hit=0.91 " * 45),
                tool_call_id=f"tc_{turn}",
                name="metrics_probe",
            )
        )
        raw_log.append(AIMessage(content=f"Completed analysis step {turn} cleanly."))

        proj = await context_engine.shape_context(
            static_system_prompt="You are Long-Horizon NIM Architect.",
            prior_compaction_checkpoint=rolling_checkpoint,
            messages=raw_log,
            policy=policy,
            thread_id="stress-100-turns",
        )
        assert proj.total_tokens <= hard_cap, f"Turn {turn} exceeded hard cap: {proj.total_tokens} > {hard_cap}"

        # Verify atomic tool-call invariant: NO orphan `ToolMessage` ever appears without preceding `AIMessage(tool_calls)`!
        b3 = proj.block3_messages
        for idx, msg in enumerate(b3):
            if isinstance(msg, ToolMessage):
                assert idx > 0 and isinstance(b3[idx - 1], (AIMessage, ToolMessage))

        if proj.compacted_this_turn:
            compactions_triggered += 1
            rolling_checkpoint = proj.updated_checkpoint
            # Simulate checkpoint trimming older compacted messages while keeping recent verbatim tail
            raw_log = list(proj.block3_messages)

    assert compactions_triggered >= 5
    assert "<compaction_checkpoint>" in rolling_checkpoint
    assert "$18,500" in rolling_checkpoint


# ─── 5. 3-Tier Progressive `SKILL.md` API & Live `make_graph` Activation ──


@pytest.mark.asyncio
async def test_progressive_skill_md_crud_and_live_langgraph_activation(isolated_phase4_env):
    from unittest.mock import patch
    import agent.graph as graph_mod
    from api.app import app

    custom_skill_md = """---
name: Dynamo Prefill Disaggregation Tuner
slug: dynamo-prefill-tuner
description: Tune NIXL RDMA prefill/decode worker ratios and queue thresholds on H100/B200 clusters.
category: infrastructure
version: 2.1.0
tags: [dynamo, nixl, h100]
allowed_tools: [calculator]
---

## Secret Disaggregation Formula
1. Set `DYNAMO_PREFILL_RATIO = 3:1` whenever `ISL > 16384` and `OSL < 512` across multi-node H100/B200 clusters.
2. Enable `NIXL_RDMA_ZEROCOPY = 1` and allocate 32 GB pinned host DRAM bounce buffers for KVBM block transfers.
3. Gate prefill admission queue depth at `max_prefill_queue = 8` per worker to keep p99 TTFT under 75ms.
"""

    with TestClient(app) as client:
        created = client.post("/api/skills", json={"raw_skill_md": custom_skill_md})
        assert created.status_code == 201
        created_data = created.json()
        assert created_data["slug"] == "dynamo-prefill-tuner"
        assert created_data["tier1_index_tokens"] < created_data["tier2_playbook_tokens"]

        # Cross-tenant isolation: another tenant does NOT see default tenant's custom skill
        other_tenant_slugs = {
            s["slug"]
            for s in client.get("/api/skills", headers={"X-Tenant-ID": "other-org"}).json()
        }
        assert "dynamo-prefill-tuner" not in other_tenant_slugs

    def _step(messages: list[BaseMessage], calls: list[list[BaseMessage]]) -> AIMessage:
        if len(calls) == 1:
            # Turn 1: Playbook formula is NOT in startup prompt (Tier-1 progressive disclosure!)
            joined = "\n".join(str(m.content) for m in messages)
            assert "DYNAMO_PREFILL_RATIO = 3:1" not in joined
            assert "dynamo-prefill-tuner" in messages[0].content
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "activate_skill",
                        "args": {"skill_slug": "dynamo-prefill-tuner"},
                        "id": "call_skill_1",
                        "type": "tool_call",
                    }
                ],
            )
        # Turn 2: Block 1 (`messages[0]`) is 100% UNCHANGED while Block 2 (`messages[1]`) now has the full playbook!
        assert calls[1][0].content == calls[0][0].content
        assert "DYNAMO_PREFILL_RATIO = 3:1" in messages[1].content
        return AIMessage(content="Applied DYNAMO_PREFILL_RATIO = 3:1!")

    fake_llm = InspectingChatModel(step_fn=_step)
    with patch.object(graph_mod, "get_model", return_value=fake_llm):
        graph = await graph_mod.make_graph({"configurable": {"thread_id": "skill-run-1"}})
        result = await graph.ainvoke({"messages": [HumanMessage(content="Tune my 32K ISL Dynamo cluster")]})

    assert len(fake_llm.seen_calls) == 2
    assert "DYNAMO_PREFILL_RATIO = 3:1" in result["messages"][-1].content

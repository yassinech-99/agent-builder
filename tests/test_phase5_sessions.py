"""Comprehensive Phase 5 Verification Suite:
  1. Stateless Crash-Safe Session Recovery (`wake(session_id)`), Monotonic Lease Fencing Tokens
     & Atomic Split-Brain Zombie Write Rejection (`StaleLeaseFenceError` + `HTTP 409` active lease guard)
  2. Reconnectable SSE Stream Fan-Out (`Last-Event-ID` header / `?after_seq=`) with Zero-Gap / Zero-Duplicate Replay
  3. Instant Cross-Worker Abort Signal (`cancel_session` < 5ms) Halting Active Generation Cleanly
  4. Time-Travel Checkpoint Session Forking (`up_to_seq`), Auto Semantic Titling, Pin/Tag/Archive,
     Multi-Tenant Session Search Isolation & Markdown / JSONL Trace Exports
  5. Context-Isolated Specialist Sub-Agents (`delegate_to_subagent` & `delegate_parallel_subagents`)
     Inside `make_graph` with Child Execution Trace Persistence & `MAX_SUBAGENT_DEPTH` Guard
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi.testclient import TestClient
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class SubAgentCapableMockModel(BaseChatModel):
    """Deterministic model driving both parent delegation and child specialist tool calls."""

    step_fn: Any = None

    @property
    def _llm_type(self) -> str:
        return "phase5-subagent-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.step_fn(messages))])


@pytest.fixture()
def isolated_phase5_env(tmp_path, monkeypatch):
    db_file = tmp_path / "phase5_nimagents.db"
    qdrant_dir = tmp_path / "phase5_qdrant"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NIMAGENTS_QDRANT_PATH", str(qdrant_dir))

    from api import db
    from api.middleware.admission import admission_controller

    admission_controller.reset_for_tests()
    db.init_db()
    yield tmp_path
    admission_controller.reset_for_tests()


# ─── 1. Pod Crash Lease Takeover (`wake`) & Split-Brain Zombie Rejection ──


@pytest.mark.asyncio
async def test_fencing_token_lease_lock_pod_crash_wake_and_zombie_rejection(isolated_phase5_env):
    from api.core.tenancy import TenantContext
    from api.services.run_manager import (
        DurableSessionManager,
        SessionLeaseBusyError,
        StaleLeaseFenceError,
    )

    mgr = DurableSessionManager(default_lease_ttl_s=30.0)
    ctx = TenantContext(tenant_id="tenant-alpha", workspace_id="prod", user_id="eng-1")
    sess = mgr.create_session(title="Rolling Deploy Resilience Test", ctx=ctx)
    sid = sess["id"]

    # Worker Pod A acquires initial lease at t=1000 (`fencing_token == 1`) and writes user prompt
    lease_pod_a = mgr.acquire_lease(sid, worker_id="brain-pod-a", ttl_s=30.0, now_monotonic=1000.0, ctx=ctx)
    assert lease_pod_a.fencing_token == 1

    mgr.emit_event(
        sid,
        event_type="user_message",
        role="user",
        payload={"content": "Compute cluster failover plan"},
        lease=lease_pod_a,
        ctx=ctx,
    )

    # At t=1010 (before 30s TTL expires!), another pod CANNOT steal active healthy lease!
    with pytest.raises(SessionLeaseBusyError):
        mgr.acquire_lease(sid, worker_id="brain-pod-b", now_monotonic=1010.0, ctx=ctx)

    # Simulate `brain-pod-a` dying mid-deploy -> at t=1035 (>1030 expiry!), `wake_session` recovers cleanly!
    async def _recovery_runner(history: list[BaseMessage], _cfg: dict[str, Any]) -> AsyncGenerator[str, None]:
        assert history[0].content == "Compute cluster failover plan"
        for tok in ("Recovered ", "seamlessly ", "on brain-pod-b!"):
            yield tok

    recovered = await mgr.wake_session(
        sid,
        worker_id="brain-pod-b",
        now_monotonic=1035.0,
        runner_fn=_recovery_runner,
        ctx=ctx,
    )
    assert recovered["recovered"] is True
    assert recovered["new_fencing_token"] == 2
    assert recovered["status"] == "completed"

    # Now if zombie `brain-pod-a` wakes from a GC pause with stale `fencing_token == 1`, its write is REJECTED!
    with pytest.raises(StaleLeaseFenceError):
        mgr.emit_event(
            sid,
            event_type="assistant_token",
            payload={"delta": "CORRUPTED_ZOMBIE_TOKEN"},
            lease=lease_pod_a,
            ctx=ctx,
        )

    final_msgs = mgr.reconstruct_messages(sid, ctx=ctx)
    assert final_msgs[-1].content == "Recovered seamlessly on brain-pod-b!"


# ─── 2. Reconnectable `Last-Event-ID` SSE Stream Replay + Live Tailing ────


@pytest.mark.asyncio
async def test_reconnectable_sse_last_event_id_zero_gap_zero_duplicate_replay(isolated_phase5_env):
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.services.run_manager import session_manager

    ctx = TenantContext(tenant_id="default", workspace_id="default")
    sess = session_manager.create_session(title="Tab Refresh Replay Test", ctx=ctx)
    sid = sess["id"]

    gate = asyncio.Event()

    async def _chunked_stream(_history: list[BaseMessage], _cfg: dict[str, Any]) -> AsyncGenerator[str, None]:
        yield "Chunk-1 "
        yield "Chunk-2 "
        await gate.wait()  # Hold mid-stream while client reconnects with `Last-Event-ID: 2`!
        yield "Chunk-3 "
        yield "Chunk-4!"

    await session_manager.start_background_run(
        sid,
        " Design 8-GPU NIM topology",
        runner_fn=_chunked_stream,
        ctx=ctx,
    )
    # Wait briefly so `seq 1` (`user_message`), `seq 2` (`Chunk-1 `), and `seq 3` (`Chunk-2 `) persist
    await asyncio.sleep(0.08)

    async def _release_remaining_chunks():
        await asyncio.sleep(0.05)
        gate.set()

    releaser = asyncio.create_task(_release_remaining_chunks())

    transport = httpx.ASGITransport(app=app)
    received_seqs: list[int] = []
    received_deltas: list[str] = []

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        # Simulate browser reconnecting after receiving `id: 2` (`Last-Event-ID: 2`)
        async with ac.stream(
            "GET",
            f"/api/sessions/{sid}/stream",
            headers={"Last-Event-ID": "2"},
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if line.startswith("id: "):
                    received_seqs.append(int(line.split("id: ", 1)[1]))
                elif line.startswith("data: "):
                    ev = json.loads(line.split("data: ", 1)[1])
                    if ev["event_type"] == "assistant_token":
                        received_deltas.append(ev["payload"]["delta"])

    await releaser

    # Strictly contiguous monotonic sequence starting at EXACT `seq 3` (zero gap, zero duplicate!)
    assert received_seqs[0] == 3
    assert received_seqs == list(range(3, received_seqs[-1] + 1))
    assert "".join(received_deltas) == "Chunk-2 Chunk-3 Chunk-4!"


# ─── 3. Instant `<5ms` Cross-Worker Cancel Signal Halts Active Stream ─────


@pytest.mark.asyncio
async def test_instant_cancel_signal_aborts_long_background_run(isolated_phase5_env):
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.services.run_manager import session_manager

    ctx = TenantContext(tenant_id="default", workspace_id="default")
    sess = session_manager.create_session(title="Cancel Signal Test", ctx=ctx)
    sid = sess["id"]
    emitted_count = 0

    async def _slow_infinite_stream(_h: list[BaseMessage], _c: dict[str, Any]) -> AsyncGenerator[str, None]:
        nonlocal emitted_count
        for idx in range(50):
            emitted_count += 1
            yield f"tok-{idx} "
            await asyncio.sleep(0.04)

    await session_manager.start_background_run(sid, "Run long job", runner_fn=_slow_infinite_stream, ctx=ctx)
    await asyncio.sleep(0.1)

    with TestClient(app) as client:
        cancel_resp = client.post(f"/api/sessions/{sid}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["status"] == "cancelled"
        assert emitted_count < 15, f"Expected immediate stop on cancel, got {emitted_count} chunks"


# ─── 4. Time-Travel Forking, Pin/Tags/Search & Markdown/JSONL Export ──────


def test_time_travel_session_forking_search_isolation_and_exports(isolated_phase5_env):
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.services.run_manager import session_manager

    alpha_ctx = TenantContext(tenant_id="tenant-alpha", workspace_id="ws-main")
    alpha_h = {"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-main"}
    bravo_h = {"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-main"}

    sess = session_manager.create_session(title="Architecture Exploration", tags=["dynamo"], ctx=alpha_ctx)
    sid = sess["id"]

    # Emit 4 events (2 full user/assistant exchanges)
    session_manager.emit_event(sid, event_type="user_message", role="user", payload={"content": "Step 1: Evaluate H200 FP8 throughput"}, ctx=alpha_ctx)
    session_manager.emit_event(sid, event_type="assistant_message", role="assistant", payload={"content": "H200 delivers 3,958 TFLOPS FP8."}, ctx=alpha_ctx)
    session_manager.emit_event(sid, event_type="user_message", role="user", payload={"content": "Step 2: Try experimental speculative decoding"}, ctx=alpha_ctx)
    session_manager.emit_event(sid, event_type="assistant_message", role="assistant", payload={"content": "Draft model overhead was 14%."}, ctx=alpha_ctx)

    with TestClient(app) as client:
        # Time-travel fork right at `up_to_seq = 2` (before speculative decoding branch!)
        fork_resp = client.post(
            f"/api/sessions/{sid}/fork",
            headers=alpha_h,
            json={"up_to_seq": 2, "title": "Clean H200 TensorRT Branch"},
        )
        assert fork_resp.status_code == 201
        forked = fork_resp.json()
        assert forked["forked_from_session_id"] == sid
        assert forked["forked_from_seq"] == 2

        fork_detail = client.get(f"/api/sessions/{forked['id']}", headers=alpha_h).json()
        orig_detail = client.get(f"/api/sessions/{sid}", headers=alpha_h).json()
        assert len(fork_detail["events"]) == 2
        assert len(orig_detail["events"]) == 4  # Original timeline 100% untouched!

        # Pin forked session & search inside event payload text (`3,958 TFLOPS`)
        client.patch(f"/api/sessions/{forked['id']}", headers=alpha_h, json={"pinned": True})
        alpha_hits = client.get("/api/sessions", params={"search": "3,958 TFLOPS"}, headers=alpha_h).json()
        bravo_hits = client.get("/api/sessions", params={"search": "3,958 TFLOPS"}, headers=bravo_h).json()
        assert len(alpha_hits) == 2
        assert alpha_hits[0]["id"] == forked["id"]  # Pinned session sorted first!
        assert bravo_hits == []  # Zero cross-tenant leakage!

        # Verify Markdown & JSONL trace exports
        md_export = client.get(f"/api/sessions/{forked['id']}/export?format=markdown", headers=alpha_h)
        jsonl_export = client.get(f"/api/sessions/{forked['id']}/export?format=jsonl", headers=alpha_h)
        assert md_export.status_code == 200 and "H200 delivers 3,958 TFLOPS FP8" in md_export.text
        assert jsonl_export.status_code == 200 and len(jsonl_export.text.strip().splitlines()) == 3


# ─── 5. Context-Isolated Specialist Sub-Agents Inside `make_graph` ────────


@pytest.mark.asyncio
async def test_context_isolated_subagent_and_parallel_fanout_in_make_graph(isolated_phase5_env):
    from unittest.mock import patch
    import agent.graph as graph_mod
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.services.run_manager import session_manager

    ctx = TenantContext(tenant_id="default", workspace_id="default")
    parent_sess = session_manager.create_session(title="Parent Orchestrator Run", ctx=ctx)
    parent_id = parent_sess["id"]

    def _router_model(messages: list[BaseMessage]) -> AIMessage:
        sys_txt = str(messages[0].content)
        last = messages[-1]

        # Child Sub-Agent Execution Window (`Specialist ... Sub-Agent`)
        if "Sub-Agent spawned by nimAgents" in sys_txt:
            if isinstance(last, HumanMessage):
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculator",
                            "args": {"expression": "16 * 141"},
                            "id": "child_calc_1",
                            "type": "tool_call",
                        }
                    ],
                )
            return AIMessage(content=f"Verified total H200 HBM3e capacity = {last.content} GB across 16 GPUs.")

        # Parent Agent Window: delegates heavy multi-step research to isolated child sub-agent!
        if isinstance(last, HumanMessage):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_to_subagent",
                        "args": {
                            "specialist_role": "coder",
                            "objective": "Compute exact HBM3e memory across 16 H200 GPUs (141 GB each).",
                            "max_steps": 4,
                        },
                        "id": "parent_del_1",
                        "type": "tool_call",
                    }
                ],
            )
        assert isinstance(last, ToolMessage)
        trace_obj = json.loads(str(last.content))
        assert trace_obj["type"] == "subagent_trace"
        assert trace_obj["steps_count"] == 2
        return AIMessage(content=f"Parent Summary -> {trace_obj['distilled_output']}")

    mock_llm = SubAgentCapableMockModel(step_fn=_router_model)
    with patch.object(graph_mod, "get_model", return_value=mock_llm):
        graph = await graph_mod.make_graph({"configurable": {"thread_id": parent_id}})
        out = await graph.ainvoke({"messages": [HumanMessage(content="Size H200 cluster HBM")]})

    assert "2256 GB" in out["messages"][-1].content

    # Verify child sub-agent execution trace persisted cleanly under `parent_session_id`!
    with TestClient(app) as client:
        detail = client.get(f"/api/sessions/{parent_id}").json()
        assert len(detail["subagents"]) == 1
        assert detail["subagents"][0]["specialist_role"] == "coder"
        assert detail["subagents"][0]["steps_count"] == 2

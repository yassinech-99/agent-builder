"""Comprehensive Phase 2 Multi-User Concurrency, Tenant Isolation & GPU Gateway Verification Suite."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx
import jwt
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class DeterministicFakeEmbeddings:
    """Fast deterministic 16-dim unit-norm embedder for multi-tenant Qdrant & batcher tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if "hyperion" in lowered or "launch code" in lowered:
                vec = [1.0] + [0.05] * 15
            elif "gpu" in lowered or "nemotron" in lowered:
                vec = [0.05, 1.0] + [0.05] * 14
            else:
                digest = hashlib.sha256(lowered.encode("utf-8")).digest()
                vec = [((b / 255.0) * 2.0 - 1.0) for b in digest[:16]]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@pytest.fixture()
def isolated_scale_env(tmp_path, monkeypatch):
    db_file = tmp_path / "phase2_nimagents.db"
    qdrant_dir = tmp_path / "phase2_qdrant"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    monkeypatch.setenv("NIMAGENTS_QDRANT_PATH", str(qdrant_dir))
    monkeypatch.setenv("NIMAGENTS_JWT_SECRET", "phase2-unit-test-jwt-secret-key-32bytes!")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-placeholder-key-1234567890")

    from api import db
    from api.core.vector_store import vector_store
    from api.middleware.admission import admission_controller
    from api.routes import local_rag

    admission_controller.reset_for_tests()
    local_rag._EMBEDDER_CACHE.clear()
    db.init_db()
    yield tmp_path
    asyncio.run(vector_store.close())
    admission_controller.reset_for_tests()


# ─── 1. Infrastructure & Zero FAISS Pickle Audit ──────────────────────────


def test_zero_faiss_pickle_deserialization_and_rls_compose_infra():
    for folder in ("api", "agent"):
        for py_file in (REPO_ROOT / folder).rglob("*.py"):
            src = py_file.read_text(encoding="utf-8")
            assert "allow_dangerous_deserialization" not in src, f"Unsafe pickle load in {py_file}"
            assert "FAISS.load_local" not in src, f"Legacy FAISS disk load in {py_file}"

    rls_sql = (REPO_ROOT / "docker" / "postgres" / "init-rls.sql").read_text(encoding="utf-8")
    assert "ENABLE ROW LEVEL SECURITY" in rls_sql
    assert "FORCE ROW LEVEL SECURITY" in rls_sql
    assert "app.current_tenant_id" in rls_sql
    for required_table in (
        "models",
        "rag_sources",
        "local_rag_sources",
        "lexical_rag_sources",
        "storage_connections",
        "guardrail_rules",
        "guardrail_events",
        "background_jobs",
        "usage_ledger",
    ):
        assert required_table in rls_sql

    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for service_name in ("postgres:", "pgbouncer:", "redis:", "qdrant:", "meilisearch:", "minio:"):
        assert service_name in compose
    assert "POOL_MODE: transaction" in compose


# ─── 2. End-to-End API Tenant, Workspace, JWT & Meilisearch Token Isolation ──


def test_end_to_end_tenant_and_workspace_api_isolation(isolated_scale_env):
    from api.app import app
    from api.core.tenancy import mint_tenant_jwt

    alpha_jwt = mint_tenant_jwt(tenant_id="tenant-alpha", workspace_id="ws-prod", user_id="alice")
    bravo_headers = {"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-prod"}
    alpha_staging_headers = {"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-staging"}

    with TestClient(app) as client:
        # Path-traversal / injection tenant header MUST be rejected with HTTP 400
        bad = client.get("/api/models", headers={"X-Tenant-ID": "../../etc/passwd"})
        assert bad.status_code == 400

        # Create model in Tenant Alpha (via signed Bearer JWT)
        created = client.post(
            "/api/models",
            headers={"Authorization": f"Bearer {alpha_jwt}"},
            json={
                "name": "Alpha Private Nemotron",
                "provider": "nvidia_nim",
                "model_name": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
                "api_key": "nvapi-alpha-private-secret-99999999",
                "is_default": True,
            },
        )
        assert created.status_code == 201
        model_id = created.json()["id"]
        assert created.headers["X-Tenant-ID"] == "tenant-alpha"
        assert created.headers["X-Workspace-ID"] == "ws-prod"

        # Tenant Bravo sees ZERO models and gets 404 when probing Tenant Alpha's model ID
        assert client.get("/api/models", headers=bravo_headers).json() == []
        assert client.get(f"/api/models/{model_id}", headers=bravo_headers).status_code == 404
        assert client.delete(f"/api/models/{model_id}", headers=bravo_headers).status_code == 404

        # Secondary workspace inside Tenant Alpha (`ws-staging`) is also cleanly isolated
        assert client.get("/api/models", headers=alpha_staging_headers).json() == []
        assert len(client.get("/api/models", headers={"Authorization": f"Bearer {alpha_jwt}"}).json()) == 1

        # Storage connections return strictly tenant-scoped prefixes (`tenants/{tid}/workspaces/{wid}/`)
        st_alpha = client.post(
            "/api/storage",
            headers={"Authorization": f"Bearer {alpha_jwt}"},
            json={
                "name": "Shared Cluster MinIO",
                "endpoint_url": "http://minio:9000",
                "bucket_name": "enterprise-corpus",
                "skip_validation": True,
            },
        )
        assert st_alpha.status_code == 201
        assert st_alpha.json()["scoped_prefix"] == "tenants/tenant-alpha/workspaces/ws-prod/"
        assert client.get("/api/storage", headers=bravo_headers).json() == []

        # Signed HS256 Meilisearch Tenant Token embeds mandatory tenant + workspace search filter
        meili_signing_key = "meili-master-signing-secret-key-2026-prod"
        tok_resp = client.post(
            "/api/scale/meili-token",
            headers={"Authorization": f"Bearer {alpha_jwt}"},
            json={
                "api_key_uid": "0195f8d0-0000-7000-8000-000000000001",
                "signing_api_key": meili_signing_key,
                "indexes": ["nimrag_shared"],
            },
        )
        assert tok_resp.status_code == 200
        decoded = jwt.decode(tok_resp.json()["token"], meili_signing_key, algorithms=["HS256"])
        assert (
            decoded["searchRules"]["nimrag_shared"]["filter"]
            == 'tenant_id = "tenant-alpha" AND workspace_id = "ws-prod"'
        )


# ─── 3. Per-Tenant Isolated Agent Tool Workspaces ─────────────────────────


def test_isolated_per_tenant_agent_workspaces(isolated_scale_env, monkeypatch):
    from agent.tools import get_tools_by_ids
    from api.core.tenancy import tenant_scope

    monkeypatch.chdir(isolated_scale_env)

    with tenant_scope("tenant-alpha", "ws-1"):
        alpha_tools = {t.name: t for t in get_tools_by_ids(["file_management"])}
        alpha_tools["write_file"].invoke({"file_path": "secret.txt", "text": "ALPHA_ONLY_SECRET"})

    with tenant_scope("tenant-bravo", "ws-1"):
        bravo_tools = {t.name: t for t in get_tools_by_ids(["file_management"])}
        bravo_tools["write_file"].invoke({"file_path": "secret.txt", "text": "BRAVO_ONLY_SECRET"})
        read_bravo = bravo_tools["read_file"].invoke({"file_path": "secret.txt"})
        escaped = bravo_tools["read_file"].invoke({"file_path": "../../tenant-alpha/ws-1/secret.txt"})

    assert read_bravo == "BRAVO_ONLY_SECRET"
    assert "ALPHA_ONLY_SECRET" not in escaped


# ─── 4. Multi-Tenant Qdrant Vector Store End-to-End Isolation ─────────────


def test_qdrant_multitenant_vector_rag_end_to_end_zero_cross_tenant_bleed(isolated_scale_env):
    from api.app import app

    fake_emb = DeterministicFakeEmbeddings()
    alpha_headers = {"X-Tenant-ID": "tenant-alpha", "X-Workspace-ID": "ws-prod"}
    bravo_headers = {"X-Tenant-ID": "tenant-bravo", "X-Workspace-ID": "ws-prod"}

    with patch("api.routes.local_rag._get_embeddings", return_value=fake_emb):
        with TestClient(app) as client:
            up_a = client.post(
                "/api/local-rag/upload",
                headers=alpha_headers,
                files={"file": ("alpha_runbook.txt", b"Project Hyperion launch code is OMEGA-7749.", "text/plain")},
                data={"name": "Alpha Runbook", "chunk_size": "500", "chunk_overlap": "50"},
            )
            assert up_a.status_code == 200, up_a.text
            alpha_source_id = up_a.json()["id"]

            up_b = client.post(
                "/api/local-rag/upload",
                headers=bravo_headers,
                files={"file": ("bravo_runbook.txt", b"Project Hyperion launch code is DELTA-9912.", "text/plain")},
                data={"name": "Bravo Runbook", "chunk_size": "500", "chunk_overlap": "50"},
            )
            assert up_b.status_code == 200, up_b.text
            bravo_source_id = up_b.json()["id"]

            # Querying identical prompt returns strictly each tenant's own vector chunk from shared Qdrant collection!
            qa = client.post(
                f"/api/local-rag/{alpha_source_id}/query",
                headers=alpha_headers,
                json={"query": "What is the Project Hyperion launch code?", "k": 4, "rerank": False},
            )
            assert qa.status_code == 200
            alpha_texts = " ".join(r["page_content"] for r in qa.json()["results"])
            assert "OMEGA-7749" in alpha_texts
            assert "DELTA-9912" not in alpha_texts

            qb = client.post(
                f"/api/local-rag/{bravo_source_id}/query",
                headers=bravo_headers,
                json={"query": "What is the Project Hyperion launch code?", "k": 4, "rerank": False},
            )
            assert qb.status_code == 200
            bravo_texts = " ".join(r["page_content"] for r in qb.json()["results"])
            assert "DELTA-9912" in bravo_texts
            assert "OMEGA-7749" not in bravo_texts

            # Cross-tenant source UUID probing is strictly 404
            cross = client.post(
                f"/api/local-rag/{alpha_source_id}/query",
                headers=bravo_headers,
                json={"query": "Project Hyperion launch code", "k": 4},
            )
            assert cross.status_code == 404


# ─── 5. 50 Concurrent Multi-Tenant Users + Zero Event-Loop Stalls ─────────


@pytest.mark.asyncio
async def test_50_concurrent_users_zero_event_loop_stalls(isolated_scale_env):
    from api.app import app
    from api.core.http_pool import EventLoopWatchdog

    fake_emb = DeterministicFakeEmbeddings()
    watchdog = EventLoopWatchdog(interval_s=0.02, stall_threshold_ms=150.0)
    watchdog.start()

    try:
        with patch("api.routes.local_rag._get_embeddings", return_value=fake_emb):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
                # Seed 5 tenant vector indexes concurrently via streaming chunked upload
                upload_tasks = [
                    ac.post(
                        "/api/local-rag/upload",
                        headers={"X-Tenant-ID": f"org-{i}", "X-Workspace-ID": "prod"},
                        files={
                            "file": (
                                f"doc_{i}.md",
                                (f"Tenant org-{i} Nemotron GPU secret token is TOK-{1000 + i}. " * 40).encode(),
                                "text/markdown",
                            )
                        },
                        data={"name": f"Org {i} Docs", "chunk_size": "600", "chunk_overlap": "50"},
                    )
                    for i in range(5)
                ]
                upload_resps = await asyncio.gather(*upload_tasks)
                source_by_org = {i: r.json()["id"] for i, r in enumerate(upload_resps)}

                # Fire 50 simultaneous enterprise user sessions (100 concurrent HTTP calls!) across 5 tenants
                async def _simulate_user(user_idx: int):
                    org_idx = user_idx % 5
                    headers = {
                        "X-Tenant-ID": f"org-{org_idx}",
                        "X-Workspace-ID": "prod",
                        "X-User-ID": f"user-{user_idx}",
                        "X-Plan-Tier": "enterprise",
                    }
                    q_resp, g_resp = await asyncio.gather(
                        ac.post(
                            f"/api/local-rag/{source_by_org[org_idx]}/query",
                            headers=headers,
                            json={"query": "Nemotron GPU secret token", "k": 2, "rerank": False},
                        ),
                        ac.post(
                            "/api/guardrails/check",
                            headers=headers,
                            json={"text": f"Hello from user-{user_idx} at user{user_idx}@example.com"},
                        ),
                    )
                    assert q_resp.status_code == 200
                    assert f"TOK-{1000 + org_idx}" in q_resp.json()["results"][0]["page_content"]
                    assert g_resp.status_code == 200

                await asyncio.gather(*(_simulate_user(u) for u in range(50)))
                snap = watchdog.snapshot()
                assert snap["stall_events"] == 0, f"Detected event loop stall under 50-user load: {snap}"
    finally:
        await watchdog.stop()


# ─── 6. Atomic Admission Control: Burst 429 Shedding & Token Reconciliation ─


@pytest.mark.asyncio
async def test_atomic_admission_controller_burst_429_and_token_reconciliation(isolated_scale_env):
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.middleware.admission import TenantQuotaPolicy, admission_controller

    admission_controller.reset_for_tests()
    admission_controller.set_tenant_policy(
        "tenant-noisy",
        TenantQuotaPolicy(max_concurrent_runs=20, max_rpm=5, max_tpm=5000, max_daily_usd=10.0),
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        # 12 simultaneous burst requests from `tenant-noisy` -> exactly 5 admitted, 7 shed with HTTP 429!
        burst = await asyncio.gather(
            *(ac.get("/api/models", headers={"X-Tenant-ID": "tenant-noisy"}) for _ in range(12))
        )
        codes = [r.status_code for r in burst]
        assert codes.count(200) == 5
        assert codes.count(429) == 7
        rejected = next(r for r in burst if r.status_code == 429)
        assert int(rejected.headers["Retry-After"]) >= 1
        assert "X-RateLimit-Remaining-Requests" in rejected.headers

        # Also verify pure concurrency semaphore cap (`max_concurrent_runs=3`, `max_rpm=100`)
        admission_controller.set_tenant_policy(
            "tenant-conc-cap",
            TenantQuotaPolicy(max_concurrent_runs=3, max_rpm=100, max_tpm=50_000, max_daily_usd=10.0),
        )
        conc_burst = await asyncio.gather(
            *(ac.get("/api/models", headers={"X-Tenant-ID": "tenant-conc-cap"}) for _ in range(8))
        )
        assert [r.status_code for r in conc_burst].count(200) == 3
        assert [r.status_code for r in conc_burst].count(429) == 5

        # Meanwhile, peaceful neighbor `tenant-calm` is 100% unaffected (`HTTP 200`)!
        calm = await ac.get("/api/models", headers={"X-Tenant-ID": "tenant-calm"})
        assert calm.status_code == 200

    # Verify upfront TPM token reservation + post-stream refund reconciliation
    tok_ctx = TenantContext(tenant_id="tenant-reconcile", workspace_id="default")
    admission_controller.set_tenant_policy(
        "tenant-reconcile",
        TenantQuotaPolicy(max_concurrent_runs=5, max_rpm=50, max_tpm=2000, max_daily_usd=10.0),
    )
    d1 = await admission_controller.acquire(ctx=tok_ctx, estimated_tokens=1600)
    assert d1.allowed is True
    # Second 800-token request is blocked while 1600 reservation is active (`1600 + 800 > 2000`)
    d2 = await admission_controller.acquire(ctx=tok_ctx, estimated_tokens=800)
    assert d2.allowed is False and d2.reason == "tpm_limit_exceeded"
    # Reconciling actual stream usage down to `300` tokens immediately frees `1300` tokens!
    await admission_controller.release_and_reconcile(d1, actual_tokens=300, actual_usd=0.002)
    d3 = await admission_controller.acquire(ctx=tok_ctx, estimated_tokens=800)
    assert d3.allowed is True
    await admission_controller.release_and_reconcile(d3, actual_tokens=800)


# ─── 7. Two-Lane Priority Pool: P0 Chat Insulation & P1 Fair Queuing ──────


@pytest.mark.asyncio
async def test_two_lane_priority_pool_p0_insulation_and_per_tenant_fair_queuing():
    from api.core.tenancy import TenantContext
    from api.workers.priority_queue import PriorityWorkPool

    pool = PriorityWorkPool(p0_concurrency=8, p1_concurrency=1)
    completion_order: list[str] = []

    try:
        noisy_ctx = TenantContext(tenant_id="tenant-heavy")
        quiet_ctx = TenantContext(tenant_id="tenant-quiet")

        async def _heavy_job(label: str):
            await asyncio.sleep(0.03)
            completion_order.append(label)
            return label

        # Enqueue 6 heavy background jobs from `tenant-heavy`, then 1 job from `tenant-quiet`
        heavy_futs = [
            await pool.enqueue(lambda lbl=f"A-{i}": _heavy_job(lbl), lane="background", ctx=noisy_ctx)
            for i in range(6)
        ]
        quiet_fut = await pool.enqueue(lambda: _heavy_job("B-0"), lane="background", ctx=quiet_ctx)

        # While P1 background queue is backed up (~210ms remaining), fire interactive P0 chat turn
        t0 = asyncio.get_running_loop().time()
        chat_reply = await pool.submit_interactive(lambda: asyncio.sleep(0.001, result="INSTANT_TOKEN"))
        chat_latency_ms = (asyncio.get_running_loop().time() - t0) * 1000.0
        assert chat_reply == "INSTANT_TOKEN"
        assert chat_latency_ms < 50.0, f"P0 interactive turn was delayed by P1 queue: {chat_latency_ms}ms"

        await asyncio.gather(*heavy_futs, quiet_fut)
        # Round-robin fairness: `B-0` finishes at index 1 or 2 — NEVER starved behind A-5!
        assert completion_order.index("B-0") <= 2, f"Tenant B starved in P1 queue: {completion_order}"
    finally:
        await pool.stop()


# ─── 8. Dynamic GPU Micro-Batcher + L1 Exact Embedding Cache ──────────────


@pytest.mark.asyncio
async def test_gpu_micro_batcher_coalesces_32_concurrent_user_queries():
    from api.services.batcher import DynamicGPUBatcher

    batcher = DynamicGPUBatcher(window_ms=10.0, max_batch_size=64)
    gpu_calls: list[int] = []

    async def _mock_gpu_rpc(texts: list[str]) -> list[list[float]]:
        gpu_calls.append(len(texts))
        await asyncio.sleep(0.005)
        return [[float(len(t)), float(idx)] for idx, t in enumerate(texts)]

    # Fire 32 concurrent single-query calls across 32 users within the 10ms window
    queries = [f"user query number {i}" for i in range(32)]
    vectors = await asyncio.gather(*(batcher.embed_text("nim-embed-v2", q, _mock_gpu_rpc) for q in queries))

    assert len(vectors) == 32
    assert len(gpu_calls) == 1, f"Expected 32 concurrent queries to coalesce into 1 GPU RPC, got {gpu_calls}"
    assert gpu_calls[0] == 32
    for idx, q in enumerate(queries):
        assert vectors[idx] == [float(len(q)), float(idx)]

    # Repeat all 32 queries -> 100% L1 SHA-256 embedding cache hits (`0` new GPU RPCs!)
    cached_vectors = await asyncio.gather(
        *(batcher.embed_text("nim-embed-v2", q, _mock_gpu_rpc) for q in queries)
    )
    assert cached_vectors == vectors
    assert len(gpu_calls) == 1
    assert batcher.l1_cache.hits == 32


# ─── 9. Prefix KV Router Affinity/Spillover + Tenant Semantic Cache ───────


@pytest.mark.asyncio
async def test_prefix_consistent_router_and_tenant_semantic_cache(isolated_scale_env):
    from agent.inference_gateway import PrefixAwareEndpointRouter, TenantSemanticCache
    from api.core.tenancy import TenantContext

    router = PrefixAwareEndpointRouter(max_inflight_per_replica=2)
    cluster = "http://nim-pod-0:8000/v1,http://nim-pod-1:8000/v1,http://nim-pod-2:8000/v1"

    # Multi-turn agent calls for same `(model, prefix_hash, thread_id)` hit identical GPU pod
    picks = {
        router.select_endpoint(cluster, model_id="nemotron-49b", prefix_hash="pfx_abc", thread_id="thr_1")
        for _ in range(10)
    }
    assert len(picks) == 1
    primary_pod = next(iter(picks))

    # Saturating `max_inflight_per_replica=2` spills 3rd concurrent request cleanly to sibling pod
    with router.lease_slot(cluster, model_id="nemotron-49b", prefix_hash="pfx_abc", thread_id="thr_1") as s1:
        with router.lease_slot(cluster, model_id="nemotron-49b", prefix_hash="pfx_abc", thread_id="thr_1") as s2:
            with router.lease_slot(cluster, model_id="nemotron-49b", prefix_hash="pfx_abc", thread_id="thr_1") as s3:
                assert s1 == primary_pod and s2 == primary_pod
                assert s3 != primary_pod

    # Marking primary pod unhealthy immediately fails over to healthy pod
    router.mark_unhealthy(primary_pod)
    failover = router.select_endpoint(cluster, model_id="nemotron-49b", prefix_hash="pfx_abc", thread_id="thr_1")
    assert failover != primary_pod

    # Tenant Semantic Cache: hits on close vector for SAME `(tenant, workspace, agent, prompt_hash, kb_hash)`
    # and strictly misses for ANY different tenant OR mutated KB version!
    sem_cache = TenantSemanticCache(similarity_threshold=0.95)
    t_alpha = TenantContext(tenant_id="tenant-alpha", workspace_id="ws-1")
    t_bravo = TenantContext(tenant_id="tenant-bravo", workspace_id="ws-1")

    v_orig = [1.0, 0.0, 0.0, 0.0]
    v_paraphrase = [0.995, 0.0998, 0.0, 0.0]  # cosine ~ 0.995 >= 0.95

    await sem_cache.store_response(
        prompt_text="Summarize Q3 architecture metrics",
        query_vector=v_orig,
        response_text="Confidential Alpha Q3 Architecture Summary",
        ctx=t_alpha,
        agent_id="architect",
        system_prompt_hash="sp_v1",
        kb_version_hash="kb_v1",
    )

    hit_ans, tier = await sem_cache.lookup(
        prompt_text="Give me a summary of Q3 architecture metrics",
        query_vector=v_paraphrase,
        ctx=t_alpha,
        agent_id="architect",
        system_prompt_hash="sp_v1",
        kb_version_hash="kb_v1",
    )
    assert tier == "l2_semantic"
    assert hit_ans == "Confidential Alpha Q3 Architecture Summary"

    # Different tenant with identical prompt & vector -> ZERO cross-tenant bleed (`miss`)!
    bleed_ans, bleed_tier = await sem_cache.lookup(
        prompt_text="Give me a summary of Q3 architecture metrics",
        query_vector=v_paraphrase,
        ctx=t_bravo,
        agent_id="architect",
        system_prompt_hash="sp_v1",
        kb_version_hash="kb_v1",
    )
    assert bleed_ans is None and bleed_tier == "miss"

    # Same tenant after knowledge base re-index (`kb_v2`) -> automatic instant invalidation (`miss`)!
    stale_ans, stale_tier = await sem_cache.lookup(
        prompt_text="Give me a summary of Q3 architecture metrics",
        query_vector=v_paraphrase,
        ctx=t_alpha,
        agent_id="architect",
        system_prompt_hash="sp_v1",
        kb_version_hash="kb_v2",
    )
    assert stale_ans is None and stale_tier == "miss"


# ─── 10. Async Postgres RLS Binding + Write-Behind FinOps Telemetry ───────


@pytest.mark.asyncio
async def test_async_postgres_rls_binding_and_write_behind_telemetry_ledger(isolated_scale_env):
    from agent.inference_gateway import telemetry_collector
    from api.app import app
    from api.core.tenancy import TenantContext
    from api.db_async import apply_postgres_rls_context, check_pgbouncer_pool_budget

    # Verify PgBouncer pool budget math catches oversubscribed direct Postgres pools
    over = check_pgbouncer_pool_budget(app_replicas=8, pool_size=30, postgres_max_connections=200)
    safe = check_pgbouncer_pool_budget(app_replicas=4, pool_size=30, postgres_max_connections=200)
    assert over["within_budget"] is False
    assert safe["within_budget"] is True

    # Verify Postgres RLS binding executes transaction-local `set_config(..., true)`
    mock_session = AsyncMock()
    tid, wid = await apply_postgres_rls_context(
        mock_session,
        TenantContext(tenant_id="acme-corp", workspace_id="eu-west"),
    )
    assert (tid, wid) == ("acme-corp", "eu-west")
    assert mock_session.execute.await_count == 2
    sql_calls = [str(call.args[0]) for call in mock_session.execute.await_args_list]
    assert any("set_config('app.current_tenant_id'" in s for s in sql_calls)
    assert any("set_config('app.current_workspace_id'" in s for s in sql_calls)

    # Verify write-behind FinOps telemetry flushes cleanly per tenant to `/api/scale/status`
    telemetry_collector.record(
        provider="nvidia_nim",
        model_name="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        prompt_tokens=10_000,
        completion_tokens=2_000,
        latency_ms=412.5,
        ctx=TenantContext(tenant_id="acme-corp", workspace_id="eu-west", user_id="eng-1"),
    )
    flushed = await telemetry_collector.flush_now()
    assert flushed >= 1

    with TestClient(app) as client:
        st_acme = client.get(
            "/api/scale/status",
            headers={"X-Tenant-ID": "acme-corp", "X-Workspace-ID": "eu-west"},
        )
        assert st_acme.status_code == 200
        body = st_acme.json()
        assert body["tenant_usage"]["calls"] == 1
        assert body["tenant_usage"]["prompt_tokens"] == 10_000
        assert body["tenant_usage"]["estimated_usd"] > 0

        st_other = client.get(
            "/api/scale/status",
            headers={"X-Tenant-ID": "other-corp", "X-Workspace-ID": "eu-west"},
        )
        assert st_other.json()["tenant_usage"]["calls"] == 0

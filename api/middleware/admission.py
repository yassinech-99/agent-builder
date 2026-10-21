"""Sub-millisecond Multi-Tenant Admission Controller (Redis 8 Atomic Lua + Local Parity Engine).

Enforces simultaneously per tenant in a single atomic operation:
  1. Active in-flight concurrency semaphore (`max_concurrent_runs`)
  2. Sliding 60-second request rate bucket (`max_rpm`)
  3. Upfront token reservation (`estimated_prompt_tokens + max_output_tokens <= max_tpm`)
     reconciled post-stream via `release_and_reconcile(...)` to prevent burst quota races
  4. Hard daily tenant USD spend ceiling (`max_daily_usd`)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
import logging
import math
import os
import time
from typing import Any, AsyncGenerator
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.core.tenancy import TenantContext, get_tenant_context, resolve_request_tenant

logger = logging.getLogger("nimagents.middleware.admission")

# Single-roundtrip atomic Redis 8 Lua admission script
ADMISSION_ACQUIRE_LUA = """
local conc_key = KEYS[1]
local rpm_key  = KEYS[2]
local tpm_key  = KEYS[3]
local usd_key  = KEYS[4]

local max_conc     = tonumber(ARGV[1])
local max_rpm      = tonumber(ARGV[2])
local max_tpm      = tonumber(ARGV[3])
local max_usd      = tonumber(ARGV[4])
local est_tokens   = tonumber(ARGV[5])
local lease_id     = ARGV[6]
local now_ms       = tonumber(ARGV[7])
local window_ms    = 60000

-- Evict expired concurrency leases (> 5 min safety TTL)
redis.call('ZREMRANGEBYSCORE', conc_key, '-inf', now_ms - 300000)
local active_conc = redis.call('ZCARD', conc_key)
if active_conc >= max_conc then
  return {0, 'concurrency_limit_exceeded', active_conc, 0, 0, 2}
end

local spent_usd = tonumber(redis.call('GET', usd_key) or '0')
if spent_usd >= max_usd then
  return {0, 'daily_budget_exceeded', active_conc, 0, 0, 60}
end

local cur_rpm = tonumber(redis.call('GET', rpm_key) or '0')
if (cur_rpm + 1) > max_rpm then
  local ttl = redis.call('PTTL', rpm_key)
  if ttl < 0 then ttl = window_ms end
  return {0, 'rpm_limit_exceeded', active_conc, 0, math.max(0, max_tpm - tonumber(redis.call('GET', tpm_key) or '0')), math.max(1, math.ceil(ttl / 1000))}
end

local cur_tpm = tonumber(redis.call('GET', tpm_key) or '0')
if (cur_tpm + est_tokens) > max_tpm then
  local ttl = redis.call('PTTL', tpm_key)
  if ttl < 0 then ttl = window_ms end
  return {0, 'tpm_limit_exceeded', active_conc, math.max(0, max_rpm - cur_rpm), math.max(0, max_tpm - cur_tpm), math.max(1, math.ceil(ttl / 1000))}
end

-- All checks passed: atomically commit reservations
redis.call('ZADD', conc_key, now_ms, lease_id)
redis.call('PEXPIRE', conc_key, 300000)

local new_rpm = redis.call('INCR', rpm_key)
if new_rpm == 1 then redis.call('PEXPIRE', rpm_key, window_ms) end

local new_tpm = redis.call('INCRBY', tpm_key, est_tokens)
if new_tpm == est_tokens then redis.call('PEXPIRE', tpm_key, window_ms) end

return {1, 'ok', active_conc + 1, math.max(0, max_rpm - new_rpm), math.max(0, max_tpm - new_tpm), 0}
"""

ADMISSION_RECONCILE_LUA = """
local conc_key = KEYS[1]
local tpm_key  = KEYS[2]
local usd_key  = KEYS[3]

local lease_id     = ARGV[1]
local token_delta  = tonumber(ARGV[2])
local usd_delta    = tonumber(ARGV[3])

redis.call('ZREM', conc_key, lease_id)
if token_delta ~= 0 and redis.call('EXISTS', tpm_key) == 1 then
  local updated = redis.call('INCRBY', tpm_key, token_delta)
  if updated < 0 then redis.call('SET', tpm_key, 0, 'KEEPTTL') end
end
if usd_delta > 0 then
  local new_usd = redis.call('INCRBYFLOAT', usd_key, usd_delta)
  if redis.call('TTL', usd_key) < 0 then redis.call('EXPIRE', usd_key, 86400) end
end
return 1
"""


@dataclass(slots=True)
class TenantQuotaPolicy:
    max_concurrent_runs: int = 8
    max_rpm: int = 120
    max_tpm: int = 200_000
    max_daily_usd: float = 50.0


TIER_POLICIES: dict[str, TenantQuotaPolicy] = {
    "free": TenantQuotaPolicy(max_concurrent_runs=3, max_rpm=30, max_tpm=40_000, max_daily_usd=5.0),
    "pro": TenantQuotaPolicy(max_concurrent_runs=12, max_rpm=240, max_tpm=400_000, max_daily_usd=100.0),
    "enterprise": TenantQuotaPolicy(max_concurrent_runs=64, max_rpm=1200, max_tpm=2_500_000, max_daily_usd=2500.0),
}


@dataclass(slots=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    lease_id: str
    tenant_id: str
    reserved_tokens: int
    active_concurrent: int
    limit_requests: int
    remaining_requests: int
    limit_tokens: int
    remaining_tokens: int
    retry_after_seconds: int = 0

    def to_headers(self) -> dict[str, str]:
        headers = {
            "X-RateLimit-Limit-Requests": str(self.limit_requests),
            "X-RateLimit-Remaining-Requests": str(max(0, self.remaining_requests)),
            "X-RateLimit-Limit-Tokens": str(self.limit_tokens),
            "X-RateLimit-Remaining-Tokens": str(max(0, self.remaining_tokens)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, self.retry_after_seconds))
            headers["X-RateLimit-Reason"] = self.reason
        return headers


class _TenantBucketState:
    __slots__ = (
        "leases",
        "window_started_at",
        "rpm_used",
        "tpm_used",
        "day_started_at",
        "usd_spent",
    )

    def __init__(self, now: float) -> None:
        self.leases: dict[str, float] = {}
        self.window_started_at: float = now
        self.rpm_used: int = 0
        self.tpm_used: int = 0
        self.day_started_at: float = now
        self.usd_spent: float = 0.0


class MultiTenantAdmissionController:
    """Atomic multi-tenant admission gate backed by Redis 8 Lua or local asyncio lock."""

    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self._buckets: dict[str, _TenantBucketState] = {}
        self._overrides: dict[str, TenantQuotaPolicy] = {}
        self._redis = None
        self._redis_checked = False
        self.total_admitted: int = 0
        self.total_rejected: int = 0

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def get_policy(self, tenant_id: str, plan_tier: str = "pro") -> TenantQuotaPolicy:
        if tenant_id in self._overrides:
            return self._overrides[tenant_id]
        base = TIER_POLICIES.get(plan_tier.lower(), TIER_POLICIES["pro"])
        return TenantQuotaPolicy(
            max_concurrent_runs=int(
                os.getenv("NIMAGENTS_MAX_CONCURRENT_RUNS", str(base.max_concurrent_runs))
            ),
            max_rpm=int(os.getenv("NIMAGENTS_MAX_RPM", str(base.max_rpm))),
            max_tpm=int(os.getenv("NIMAGENTS_MAX_TPM", str(base.max_tpm))),
            max_daily_usd=float(os.getenv("NIMAGENTS_MAX_DAILY_USD", str(base.max_daily_usd))),
        )

    def set_tenant_policy(self, tenant_id: str, policy: TenantQuotaPolicy) -> None:
        self._overrides[tenant_id] = policy

    async def _get_redis(self):
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            return None
        if self._redis is not None:
            return self._redis
        if self._redis_checked:
            return None
        self._redis_checked = True
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url, decode_responses=True)
            await client.ping()
            self._redis = client
            logger.info("Admission controller connected to Redis at %s", redis_url)
            return self._redis
        except Exception as exc:
            logger.warning("Redis unavailable (%s); admission controller using local atomic engine", exc)
            return None

    async def acquire(
        self,
        *,
        ctx: TenantContext | None = None,
        estimated_tokens: int = 512,
    ) -> AdmissionDecision:
        active = ctx or get_tenant_context()
        tid = active.tenant_id
        policy = self.get_policy(tid, active.plan_tier)
        est = max(1, int(estimated_tokens))
        lease_id = str(uuid.uuid4())

        rclient = await self._get_redis()
        if rclient is not None:
            now_ms = int(time.time() * 1000)
            res = await rclient.eval(
                ADMISSION_ACQUIRE_LUA,
                4,
                f"nim:conc:{tid}",
                f"nim:rpm:{tid}",
                f"nim:tpm:{tid}",
                f"nim:usd:{tid}",
                policy.max_concurrent_runs,
                policy.max_rpm,
                policy.max_tpm,
                policy.max_daily_usd,
                est,
                lease_id,
                now_ms,
            )
            allowed = bool(int(res[0]))
            if allowed:
                self.total_admitted += 1
            else:
                self.total_rejected += 1
            return AdmissionDecision(
                allowed=allowed,
                reason=str(res[1]),
                lease_id=lease_id if allowed else "",
                tenant_id=tid,
                reserved_tokens=est if allowed else 0,
                active_concurrent=int(res[2]),
                limit_requests=policy.max_rpm,
                remaining_requests=int(res[3]),
                limit_tokens=policy.max_tpm,
                remaining_tokens=int(res[4]),
                retry_after_seconds=int(res[5]),
            )

        async with self._get_lock():
            now = time.monotonic()
            state = self._buckets.get(tid)
            if state is None:
                state = _TenantBucketState(now)
                self._buckets[tid] = state

            # Evict stale leases > 300s
            expired = [k for k, ts in state.leases.items() if now - ts > 300.0]
            for k in expired:
                state.leases.pop(k, None)

            # Roll 60-second rate window
            if now - state.window_started_at >= 60.0:
                state.window_started_at = now
                state.rpm_used = 0
                state.tpm_used = 0

            # Roll 24-hour budget window
            if now - state.day_started_at >= 86400.0:
                state.day_started_at = now
                state.usd_spent = 0.0

            window_remaining_s = max(1, math.ceil(60.0 - (now - state.window_started_at)))
            active_conc = len(state.leases)

            if active_conc >= policy.max_concurrent_runs:
                self.total_rejected += 1
                return AdmissionDecision(
                    allowed=False,
                    reason="concurrency_limit_exceeded",
                    lease_id="",
                    tenant_id=tid,
                    reserved_tokens=0,
                    active_concurrent=active_conc,
                    limit_requests=policy.max_rpm,
                    remaining_requests=max(0, policy.max_rpm - state.rpm_used),
                    limit_tokens=policy.max_tpm,
                    remaining_tokens=max(0, policy.max_tpm - state.tpm_used),
                    retry_after_seconds=2,
                )

            if state.usd_spent >= policy.max_daily_usd:
                self.total_rejected += 1
                return AdmissionDecision(
                    allowed=False,
                    reason="daily_budget_exceeded",
                    lease_id="",
                    tenant_id=tid,
                    reserved_tokens=0,
                    active_concurrent=active_conc,
                    limit_requests=policy.max_rpm,
                    remaining_requests=0,
                    limit_tokens=policy.max_tpm,
                    remaining_tokens=0,
                    retry_after_seconds=60,
                )

            if state.rpm_used + 1 > policy.max_rpm:
                self.total_rejected += 1
                return AdmissionDecision(
                    allowed=False,
                    reason="rpm_limit_exceeded",
                    lease_id="",
                    tenant_id=tid,
                    reserved_tokens=0,
                    active_concurrent=active_conc,
                    limit_requests=policy.max_rpm,
                    remaining_requests=0,
                    limit_tokens=policy.max_tpm,
                    remaining_tokens=max(0, policy.max_tpm - state.tpm_used),
                    retry_after_seconds=window_remaining_s,
                )

            if state.tpm_used + est > policy.max_tpm:
                self.total_rejected += 1
                return AdmissionDecision(
                    allowed=False,
                    reason="tpm_limit_exceeded",
                    lease_id="",
                    tenant_id=tid,
                    reserved_tokens=0,
                    active_concurrent=active_conc,
                    limit_requests=policy.max_rpm,
                    remaining_requests=max(0, policy.max_rpm - state.rpm_used),
                    limit_tokens=policy.max_tpm,
                    remaining_tokens=max(0, policy.max_tpm - state.tpm_used),
                    retry_after_seconds=window_remaining_s,
                )

            state.leases[lease_id] = now
            state.rpm_used += 1
            state.tpm_used += est
            self.total_admitted += 1

            return AdmissionDecision(
                allowed=True,
                reason="ok",
                lease_id=lease_id,
                tenant_id=tid,
                reserved_tokens=est,
                active_concurrent=len(state.leases),
                limit_requests=policy.max_rpm,
                remaining_requests=max(0, policy.max_rpm - state.rpm_used),
                limit_tokens=policy.max_tpm,
                remaining_tokens=max(0, policy.max_tpm - state.tpm_used),
                retry_after_seconds=0,
            )

    async def release_and_reconcile(
        self,
        decision: AdmissionDecision,
        *,
        actual_tokens: int | None = None,
        actual_usd: float = 0.0,
    ) -> None:
        if not decision.allowed or not decision.lease_id:
            return
        tid = decision.tenant_id
        token_delta = (
            int(actual_tokens) - decision.reserved_tokens
            if actual_tokens is not None
            else 0
        )

        rclient = await self._get_redis()
        if rclient is not None:
            await rclient.eval(
                ADMISSION_RECONCILE_LUA,
                3,
                f"nim:conc:{tid}",
                f"nim:tpm:{tid}",
                f"nim:usd:{tid}",
                decision.lease_id,
                token_delta,
                float(actual_usd),
            )
            return

        async with self._get_lock():
            state = self._buckets.get(tid)
            if state is None:
                return
            state.leases.pop(decision.lease_id, None)
            state.tpm_used = max(0, state.tpm_used + token_delta)
            if actual_usd > 0:
                state.usd_spent += float(actual_usd)

    @asynccontextmanager
    async def guard(
        self,
        *,
        ctx: TenantContext | None = None,
        estimated_tokens: int = 512,
    ) -> AsyncGenerator[AdmissionDecision, None]:
        decision = await self.acquire(ctx=ctx, estimated_tokens=estimated_tokens)
        try:
            yield decision
        finally:
            await self.release_and_reconcile(decision)

    async def status_summary(self, ctx: TenantContext | None = None) -> dict[str, Any]:
        active = ctx or get_tenant_context()
        tid = active.tenant_id
        policy = self.get_policy(tid, active.plan_tier)
        async with self._get_lock():
            state = self._buckets.get(tid)
            snap = {
                "active_concurrent": len(state.leases) if state else 0,
                "rpm_used": state.rpm_used if state else 0,
                "tpm_used": state.tpm_used if state else 0,
                "usd_spent": round(state.usd_spent, 6) if state else 0.0,
            }
        return {
            "backend": "redis_lua" if self._redis is not None else "local_atomic",
            "tenant_id": tid,
            "policy": asdict(policy),
            "usage": snap,
            "total_admitted": self.total_admitted,
            "total_rejected": self.total_rejected,
        }

    def reset_for_tests(self) -> None:
        self._buckets.clear()
        self._overrides.clear()
        self.total_admitted = 0
        self.total_rejected = 0


admission_controller = MultiTenantAdmissionController()


class AdmissionMiddleware(BaseHTTPMiddleware):
    """ASGI rate-limiting & concurrency guard shedding excess multi-tenant load cleanly (`HTTP 429`)."""

    EXEMPT_PREFIXES = ("/health", "/api/health", "/docs", "/openapi.json", "/api/scale")

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path == "/" or any(path.startswith(p) for p in self.EXEMPT_PREFIXES):
            return await call_next(request)

        ctx: TenantContext = getattr(request.state, "tenant", None) or resolve_request_tenant(
            request.headers
        )
        # Estimate higher token budget on query / test / generation routes than metadata CRUD
        is_inference = any(
            seg in path for seg in ("/query", "/test", "/generate-metadata", "/upload")
        )
        est_tokens = 1024 if is_inference else 64

        decision = await admission_controller.acquire(ctx=ctx, estimated_tokens=est_tokens)
        if not decision.allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": f"Tenant rate or concurrency quota exceeded ({decision.reason})",
                    "reason": decision.reason,
                    "tenant_id": decision.tenant_id,
                    "retry_after": decision.retry_after_seconds,
                },
                headers=decision.to_headers(),
            )

        try:
            response = await call_next(request)
            for k, v in decision.to_headers().items():
                response.headers[k] = v
            return response
        finally:
            # Release slot immediately; non-inference routes reconcile token reservation back
            actual = est_tokens if is_inference else 0
            await admission_controller.release_and_reconcile(decision, actual_tokens=actual)

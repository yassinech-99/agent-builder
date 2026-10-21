"""End-to-end Multi-Tenant & Workspace Security Context (`ContextVar`, JWT, Meilisearch Tenant Tokens & Streaming Uploads)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
from typing import Any, Generator
import uuid

from fastapi import HTTPException, Request, UploadFile
import jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("nimagents.core.tenancy")

DEFAULT_TENANT_ID = "default"
DEFAULT_WORKSPACE_ID = "default"
DEFAULT_USER_ID = "anonymous"
DEFAULT_PLAN_TIER = "pro"

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB bounded streaming chunks


@dataclass(frozen=True, slots=True)
class TenantContext:
    """Immutable per-request / per-task tenant identity."""

    tenant_id: str = DEFAULT_TENANT_ID
    workspace_id: str = DEFAULT_WORKSPACE_ID
    user_id: str = DEFAULT_USER_ID
    plan_tier: str = DEFAULT_PLAN_TIER

    @property
    def storage_prefix(self) -> str:
        return f"tenants/{self.tenant_id}/workspaces/{self.workspace_id}"


_CURRENT_TENANT: ContextVar[TenantContext] = ContextVar(
    "nimagents_tenant_context",
    default=TenantContext(),
)


def sanitize_tenant_identifier(raw: str | None, fallback: str = DEFAULT_TENANT_ID) -> str:
    """Validate and normalize tenant/workspace/user identifiers against injection & path traversal."""
    if raw is None:
        return fallback
    cleaned = raw.strip()
    if not cleaned:
        return fallback
    if not _SAFE_ID_RE.match(cleaned) or ".." in cleaned:
        raise ValueError(
            f"Invalid tenant/workspace identifier '{cleaned[:32]}': "
            "must match ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
        )
    return cleaned


def get_tenant_context() -> TenantContext:
    """Return the active async/thread-local `TenantContext`."""
    return _CURRENT_TENANT.get()


def set_tenant_context(ctx: TenantContext) -> Token[TenantContext]:
    return _CURRENT_TENANT.set(ctx)


def reset_tenant_context(token: Token[TenantContext]) -> None:
    _CURRENT_TENANT.reset(token)


@contextmanager
def tenant_scope(
    tenant_id: str = DEFAULT_TENANT_ID,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    user_id: str = DEFAULT_USER_ID,
    plan_tier: str = DEFAULT_PLAN_TIER,
) -> Generator[TenantContext, None, None]:
    """Bind a `TenantContext` cleanly for the duration of a sync or async block."""
    ctx = TenantContext(
        tenant_id=sanitize_tenant_identifier(tenant_id, DEFAULT_TENANT_ID),
        workspace_id=sanitize_tenant_identifier(workspace_id, DEFAULT_WORKSPACE_ID),
        user_id=sanitize_tenant_identifier(user_id, DEFAULT_USER_ID),
        plan_tier=sanitize_tenant_identifier(plan_tier, DEFAULT_PLAN_TIER).lower(),
    )
    token = set_tenant_context(ctx)
    try:
        yield ctx
    finally:
        reset_tenant_context(token)


def mint_tenant_jwt(
    tenant_id: str,
    workspace_id: str = DEFAULT_WORKSPACE_ID,
    user_id: str = DEFAULT_USER_ID,
    plan_tier: str = DEFAULT_PLAN_TIER,
    secret: str | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Create a signed HS256 JWT carrying multi-tenant claims."""
    signing_secret = secret or os.getenv("NIMAGENTS_JWT_SECRET", "nimagents-dev-secret-change-me")
    now = datetime.now(tz=timezone.utc)
    payload = {
        "sub": sanitize_tenant_identifier(user_id, DEFAULT_USER_ID),
        "tenant_id": sanitize_tenant_identifier(tenant_id, DEFAULT_TENANT_ID),
        "workspace_id": sanitize_tenant_identifier(workspace_id, DEFAULT_WORKSPACE_ID),
        "tier": sanitize_tenant_identifier(plan_tier, DEFAULT_PLAN_TIER).lower(),
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp()),
    }
    return jwt.encode(payload, signing_secret, algorithm="HS256")


def decode_bearer_token(token_str: str, secret: str | None = None) -> TenantContext:
    """Verify HS256 Bearer JWT and extract `TenantContext`."""
    signing_secret = secret or os.getenv("NIMAGENTS_JWT_SECRET", "nimagents-dev-secret-change-me")
    claims = jwt.decode(token_str, signing_secret, algorithms=["HS256"])
    return TenantContext(
        tenant_id=sanitize_tenant_identifier(
            claims.get("tenant_id") or claims.get("org_id"), DEFAULT_TENANT_ID
        ),
        workspace_id=sanitize_tenant_identifier(
            claims.get("workspace_id"), DEFAULT_WORKSPACE_ID
        ),
        user_id=sanitize_tenant_identifier(claims.get("sub"), DEFAULT_USER_ID),
        plan_tier=sanitize_tenant_identifier(
            claims.get("tier") or claims.get("plan_tier"), DEFAULT_PLAN_TIER
        ).lower(),
    )


def resolve_request_tenant(headers: Any) -> TenantContext:
    """Resolve `TenantContext` from Authorization Bearer JWT or `X-Tenant-ID` headers."""
    require_auth = os.getenv("NIMAGENTS_REQUIRE_AUTH", "0").strip() in ("1", "true", "yes")
    auth_header = (headers.get("authorization") or "").strip()

    if auth_header.lower().startswith("bearer "):
        raw_token = auth_header[7:].strip()
        try:
            return decode_bearer_token(raw_token)
        except jwt.PyJWTError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid bearer token: {exc}") from exc

    if require_auth:
        raise HTTPException(status_code=401, detail="Missing Authorization Bearer token")

    try:
        return TenantContext(
            tenant_id=sanitize_tenant_identifier(
                headers.get("x-tenant-id"), DEFAULT_TENANT_ID
            ),
            workspace_id=sanitize_tenant_identifier(
                headers.get("x-workspace-id"), DEFAULT_WORKSPACE_ID
            ),
            user_id=sanitize_tenant_identifier(
                headers.get("x-user-id"), DEFAULT_USER_ID
            ),
            plan_tier=sanitize_tenant_identifier(
                headers.get("x-plan-tier"), DEFAULT_PLAN_TIER
            ).lower(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


class TenancyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware propagating `TenantContext` via `ContextVar` across all requests."""

    EXEMPT_PATHS = {"/", "/health", "/api/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        try:
            ctx = resolve_request_tenant(request.headers)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        token = set_tenant_context(ctx)
        request.state.tenant = ctx
        try:
            response = await call_next(request)
            response.headers["X-Tenant-ID"] = ctx.tenant_id
            response.headers["X-Workspace-ID"] = ctx.workspace_id
            return response
        finally:
            reset_tenant_context(token)


def scoped_object_prefix(
    extra_prefix: str = "",
    ctx: TenantContext | None = None,
) -> str:
    """Build strict tenant-isolated S3/MinIO object key prefix:
    `tenants/{tenant_id}/workspaces/{workspace_id}/{extra_prefix}`
    """
    active = ctx or get_tenant_context()
    cleaned_extra = extra_prefix.strip().strip("/")
    base = active.storage_prefix
    return f"{base}/{cleaned_extra}/" if cleaned_extra else f"{base}/"


def scoped_workspace_dir(
    root: str | Path = "./workspace",
    ctx: TenantContext | None = None,
) -> Path:
    """Return an isolated per-tenant/per-workspace filesystem directory (replaces shared `./workspace`)."""
    active = ctx or get_tenant_context()
    target = (
        Path(root).resolve()
        / sanitize_tenant_identifier(active.tenant_id)
        / sanitize_tenant_identifier(active.workspace_id)
    )
    target.mkdir(parents=True, exist_ok=True)
    return target


def build_meili_tenant_filter(
    ctx: TenantContext | None = None,
    source_id: str | None = None,
) -> str:
    """Build mandatory Meilisearch filter expression locking queries to `tenant_id` & `workspace_id`."""
    active = ctx or get_tenant_context()
    tid = sanitize_tenant_identifier(active.tenant_id)
    wid = sanitize_tenant_identifier(active.workspace_id)
    clauses = [f'tenant_id = "{tid}"', f'workspace_id = "{wid}"']
    if source_id:
        safe_sid = sanitize_tenant_identifier(source_id)
        clauses.append(f'source_id = "{safe_sid}"')
    return " AND ".join(clauses)


def generate_meili_tenant_token(
    api_key_uid: str,
    api_key: str,
    ctx: TenantContext | None = None,
    indexes: list[str] | None = None,
    expires_in_seconds: int = 3600,
) -> str:
    """Generate a standard HS256-signed Meilisearch Tenant Token enforcing tenant filter rules."""
    if not api_key_uid or not api_key:
        raise ValueError("Meilisearch api_key_uid and signing api_key are required")
    active = ctx or get_tenant_context()
    filter_expr = build_meili_tenant_filter(active)
    target_indexes = indexes or ["*"]
    search_rules = {idx: {"filter": filter_expr} for idx in target_indexes}
    exp = datetime.now(tz=timezone.utc) + timedelta(seconds=expires_in_seconds)
    payload = {
        "searchRules": search_rules,
        "apiKeyUid": api_key_uid,
        "exp": int(exp.timestamp()),
    }
    return jwt.encode(payload, api_key, algorithm="HS256")


async def stream_upload_to_path(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
    chunk_size: int = UPLOAD_CHUNK_BYTES,
) -> tuple[int, str]:
    """Stream an `UploadFile` to disk in fixed `1 MiB` async chunks (`O(1)` worker RAM).

    Never buffers an entire 50 MB upload in memory across concurrent tenants.
    Returns `(total_bytes, sha256_hex)`.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    digest = hashlib.sha256()

    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit",
                    )
                digest.update(chunk)
                handle.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Streaming upload failed: {exc}") from exc

    return total, digest.hexdigest()

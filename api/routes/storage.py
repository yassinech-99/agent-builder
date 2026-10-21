"""Multi-tenant S3-compatible object storage CRUD & prefix-enforced endpoints powered by MinIO."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from urllib.parse import urlparse
import uuid

from fastapi import APIRouter, HTTPException
from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field

from api.core.tenancy import TenantContext, get_tenant_context, scoped_object_prefix
from api.db import get_db, mask_secret

logger = logging.getLogger("nimagents.api.storage")
router = APIRouter(prefix="/api/storage", tags=["storage"])

DEFAULT_MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")


class StorageConnectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    endpoint_url: str = Field(default=DEFAULT_MINIO_ENDPOINT, max_length=512)
    bucket_name: str = Field(..., min_length=1, max_length=255)
    region: str = Field(default="us-east-1", max_length=64)
    access_key: str = Field(default="", max_length=256)
    secret_key: str = Field(default="", max_length=256)
    skip_validation: bool = False


class StorageConnectionOut(BaseModel):
    id: str
    tenant_id: str = "default"
    workspace_id: str = "default"
    scoped_prefix: str = "tenants/default/workspaces/default/"
    name: str
    endpoint_url: str
    bucket_name: str
    region: str
    access_key: str = ""
    access_key_masked: str = ""
    secret_key: str = ""
    secret_key_masked: str = ""
    has_credentials: bool = False
    secure: bool = False
    created_at: str


def _parse_minio_endpoint(raw_url: str) -> tuple[str, bool]:
    cleaned = (raw_url or DEFAULT_MINIO_ENDPOINT).strip()
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        host = parsed.netloc or parsed.path
        secure = parsed.scheme.lower() == "https"
        return host, secure
    return cleaned, False


def build_minio_client(
    endpoint_url: str = "",
    access_key: str = "",
    secret_key: str = "",
    region: str = "",
) -> Minio:
    host, secure = _parse_minio_endpoint(endpoint_url)
    resolved_access = access_key.strip() or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    resolved_secret = secret_key.strip() or os.getenv("MINIO_SECRET_KEY", "minioadmin")
    return Minio(
        endpoint=host,
        access_key=resolved_access,
        secret_key=resolved_secret,
        secure=secure,
        region=region.strip() or None,
    )


def build_minio_client_for_bucket(bucket_name: str, ctx: TenantContext | None = None) -> Minio:
    active = ctx or get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM storage_connections
               WHERE bucket_name = ? AND tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (bucket_name, active.tenant_id, active.workspace_id),
        ).fetchone()
    if row:
        return build_minio_client(
            endpoint_url=row["endpoint_url"],
            access_key=row["access_key"],
            secret_key=row["secret_key"],
            region=row["region"],
        )
    return build_minio_client()


def _serialize_storage_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    raw_access = data.pop("access_key", "") or ""
    raw_secret = data.pop("secret_key", "") or ""
    tid = data.get("tenant_id") or "default"
    wid = data.get("workspace_id") or "default"
    return {
        **data,
        "scoped_prefix": scoped_object_prefix("", TenantContext(tenant_id=tid, workspace_id=wid)),
        "access_key": "",
        "access_key_masked": mask_secret(raw_access),
        "secret_key": "",
        "secret_key_masked": mask_secret(raw_secret),
        "has_credentials": bool(raw_access.strip() and raw_secret.strip()),
        "secure": bool(data.get("secure", 0)),
    }


@router.get("", response_model=list[StorageConnectionOut])
def list_connections():
    ctx = get_tenant_context()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM storage_connections
               WHERE tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC""",
            (ctx.tenant_id, ctx.workspace_id),
        ).fetchall()
    return [_serialize_storage_row(r) for r in rows]


@router.post("", response_model=StorageConnectionOut, status_code=201)
async def create_connection(body: StorageConnectionCreate):
    host, secure = _parse_minio_endpoint(body.endpoint_url)
    normalized_url = f"{'https' if secure else 'http'}://{host}"

    if not body.skip_validation:
        try:
            client = build_minio_client(
                endpoint_url=normalized_url,
                access_key=body.access_key,
                secret_key=body.secret_key,
                region=body.region,
            )
            exists = await asyncio.to_thread(client.bucket_exists, body.bucket_name)
            if not exists:
                raise HTTPException(
                    status_code=400,
                    detail=f"Bucket '{body.bucket_name}' does not exist at {normalized_url}",
                )
        except HTTPException:
            raise
        except S3Error as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Object storage rejected request ({exc.code}): {exc.message}",
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not connect to object storage at {normalized_url}: {exc}",
            ) from exc

    ctx = get_tenant_context()
    conn_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO storage_connections
               (id, tenant_id, workspace_id, name, endpoint_url, bucket_name, region, access_key, secret_key, secure)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                conn_id,
                ctx.tenant_id,
                ctx.workspace_id,
                body.name.strip(),
                normalized_url,
                body.bucket_name.strip(),
                body.region.strip(),
                body.access_key.strip(),
                body.secret_key.strip(),
                int(secure),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM storage_connections WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (conn_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
    return _serialize_storage_row(row)


@router.delete("/{connection_id}")
def delete_connection(connection_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        cur = conn.execute(
            "DELETE FROM storage_connections WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (connection_id, ctx.tenant_id, ctx.workspace_id),
        )
        conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Connection not found")
    return {"ok": True}


@router.post("/{connection_id}/test")
async def test_connection(connection_id: str):
    ctx = get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM storage_connections WHERE id = ? AND tenant_id = ? AND workspace_id = ?",
            (connection_id, ctx.tenant_id, ctx.workspace_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Connection not found")
    record = dict(row)
    try:
        client = build_minio_client(
            endpoint_url=record["endpoint_url"],
            access_key=record["access_key"],
            secret_key=record["secret_key"],
            region=record["region"],
        )
        exists = await asyncio.to_thread(client.bucket_exists, record["bucket_name"])
        if not exists:
            return {
                "ok": False,
                "message": f"Bucket '{record['bucket_name']}' was not found on {record['endpoint_url']}",
            }
        return {
            "ok": True,
            "message": f"Connected to {record['endpoint_url']} · bucket '{record['bucket_name']}' is ready",
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}

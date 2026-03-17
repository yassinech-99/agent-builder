"""S3 connection CRUD endpoints."""

from __future__ import annotations

import logging
import uuid

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.db import get_db

logger = logging.getLogger("novagrid.api.storage")
router = APIRouter(prefix="/api/storage", tags=["storage"])


class S3ConnectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    bucket_name: str = Field(..., min_length=1, max_length=255)
    region: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""


class S3ConnectionOut(S3ConnectionCreate):
    id: str
    created_at: str


def _s3_client(region: str = "", access_key_id: str = "", secret_access_key: str = ""):
    kwargs: dict = {}
    if region:
        kwargs["region_name"] = region
    if access_key_id and secret_access_key:
        kwargs["aws_access_key_id"] = access_key_id
        kwargs["aws_secret_access_key"] = secret_access_key
    return boto3.client("s3", **kwargs)


@router.get("", response_model=list[S3ConnectionOut])
def list_connections():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM s3_connections ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("", response_model=S3ConnectionOut, status_code=201)
def create_connection(body: S3ConnectionCreate):
    try:
        client = _s3_client(body.region, body.access_key_id, body.secret_access_key)
        client.head_bucket(Bucket=body.bucket_name)
    except NoCredentialsError:
        raise HTTPException(status_code=400, detail="AWS credentials not found")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "404":
            raise HTTPException(status_code=400, detail=f"Bucket '{body.bucket_name}' not found")
        if code == "403":
            raise HTTPException(status_code=400, detail=f"Access denied to bucket '{body.bucket_name}'")
        raise HTTPException(status_code=400, detail=str(e))

    conn_id = str(uuid.uuid4())
    with get_db() as conn:
        conn.execute(
            """INSERT INTO s3_connections (id, name, bucket_name, region, access_key_id, secret_access_key)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (conn_id, body.name, body.bucket_name, body.region, body.access_key_id, body.secret_access_key),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM s3_connections WHERE id = ?", (conn_id,)).fetchone()
    return dict(row)


@router.delete("/{connection_id}")
def delete_connection(connection_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM s3_connections WHERE id = ?", (connection_id,))
        conn.commit()
    return {"ok": True}


@router.post("/{connection_id}/test")
def test_connection(connection_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM s3_connections WHERE id = ?", (connection_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Connection not found")
    s3conn = dict(row)
    try:
        client = _s3_client(s3conn["region"], s3conn["access_key_id"], s3conn["secret_access_key"])
        client.head_bucket(Bucket=s3conn["bucket_name"])
        return {"ok": True, "message": f"Successfully connected to s3://{s3conn['bucket_name']}"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

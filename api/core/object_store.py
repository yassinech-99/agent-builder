"""Pure-Python S3 REST Signature v4 (`AWS4-HMAC-SHA256`) client for RustFS & S3-compatible stores.

Implements standards-compliant path-style HTTP/1.1 & HTTP/2 S3 requests directly over `httpx` with
zero third-party cloud vendor SDK dependencies, supporting:
  - RustFS (`http://localhost:9000` / `http://rustfs:9000`, Apache-2.0 distributed object storage)
  - Cloudflare R2, SeaweedFS, Garage, Ceph RGW, and any S3-compatible object store.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import mimetypes
import os
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

import httpx

from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db

DEFAULT_RUSTFS_ENDPOINT = os.getenv("RUSTFS_ENDPOINT", "http://localhost:9000")
DEFAULT_RUSTFS_ACCESS_KEY = "nimagentsadmin"
DEFAULT_RUSTFS_SECRET_KEY = "nimagentssecretkey"
DEFAULT_RUSTFS_REGION = "us-east-1"


class ObjectStorageError(Exception):
    """Structured error raised when an object store returns a non-2xx S3 error response."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.status_code = status_code


@dataclass(slots=True)
class StoredObjectEntry:
    object_name: str
    size: int = 0
    etag: str = ""
    last_modified: str = ""
    is_dir: bool = False


def parse_storage_endpoint(raw_url: str) -> tuple[str, bool]:
    """Return `(host_port, is_https)` from an endpoint URL or `host:port` string."""
    cleaned = (raw_url or os.getenv("RUSTFS_ENDPOINT", DEFAULT_RUSTFS_ENDPOINT)).strip()
    if "://" in cleaned:
        parsed = urlparse(cleaned)
        host = parsed.netloc or parsed.path
        secure = parsed.scheme.lower() == "https"
        return host.rstrip("/"), secure
    return cleaned.rstrip("/"), False


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _derive_sigv4_signing_key(secret_key: str, date_stamp: str, region: str, service: str = "s3") -> bytes:
    k_date = _hmac_sha256(f"AWS4{secret_key}".encode("utf-8"), date_stamp)
    k_region = _hmac_sha256(k_date, region)
    k_service = _hmac_sha256(k_region, service)
    return _hmac_sha256(k_service, "aws4_request")


def _parse_s3_error_xml(status_code: int, raw_body: bytes) -> ObjectStorageError:
    if raw_body:
        try:
            root = ET.fromstring(raw_body)
            code = (root.findtext("Code") or root.findtext("{*}Code") or f"HTTP_{status_code}").strip()
            msg = (root.findtext("Message") or root.findtext("{*}Message") or "Object storage error").strip()
            return ObjectStorageError(code=code, message=msg, status_code=status_code)
        except ET.ParseError:
            pass
    return ObjectStorageError(
        code=f"HTTP_{status_code}",
        message=(raw_body.decode("utf-8", errors="replace")[:240] or f"HTTP {status_code}"),
        status_code=status_code,
    )


class RustFSClient:
    """Lightweight path-style S3 SigV4 client tailored for RustFS & S3-compatible object storage."""

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
        region: str = DEFAULT_RUSTFS_REGION,
        timeout_s: float = 20.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = endpoint.strip().rstrip("/")
        self.access_key = access_key.strip()
        self.secret_key = secret_key.strip()
        self.secure = bool(secure)
        self.region = (region or DEFAULT_RUSTFS_REGION).strip() or DEFAULT_RUSTFS_REGION
        self.base_url = f"{'https' if self.secure else 'http'}://{self.host}"
        self._client = httpx.Client(timeout=timeout_s, transport=transport, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def _sign_headers(
        self,
        method: str,
        canonical_uri: str,
        query_params: dict[str, str] | None,
        payload_sha256: str,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        now = datetime.now(timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")

        headers: dict[str, str] = {
            "host": self.host,
            "x-amz-content-sha256": payload_sha256,
            "x-amz-date": amz_date,
        }
        if extra_headers:
            for k, v in extra_headers.items():
                headers[k.lower()] = v.strip()

        sorted_q = "&".join(
            f"{quote(k, safe='-_.~')}={quote(str(v), safe='-_.~')}"
            for k, v in sorted((query_params or {}).items())
        )
        sorted_hdr_items = sorted(headers.items())
        canonical_headers = "".join(f"{k}:{v}\n" for k, v in sorted_hdr_items)
        signed_headers = ";".join(k for k, _ in sorted_hdr_items)

        canonical_request = "\n".join(
            [
                method.upper(),
                canonical_uri,
                sorted_q,
                canonical_headers,
                signed_headers,
                payload_sha256,
            ]
        )
        credential_scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
            ]
        )
        signing_key = _derive_sigv4_signing_key(self.secret_key, date_stamp, self.region, "s3")
        signature = hmac.new(
            signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        out_headers = dict(headers)
        out_headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self.access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return out_headers

    def _request(
        self,
        method: str,
        bucket_name: str,
        object_name: str = "",
        *,
        query_params: dict[str, str] | None = None,
        content: bytes = b"",
        content_type: str | None = None,
    ) -> httpx.Response:
        b_enc = quote(bucket_name.strip("/"), safe="-_.~")
        if object_name:
            k_enc = quote(object_name.lstrip("/"), safe="/-_.~")
            canonical_uri = f"/{b_enc}/{k_enc}"
        else:
            canonical_uri = f"/{b_enc}"

        payload_hash = hashlib.sha256(content).hexdigest()
        extra: dict[str, str] = {}
        if content_type:
            extra["content-type"] = content_type

        headers = self._sign_headers(
            method,
            canonical_uri,
            query_params=query_params,
            payload_sha256=payload_hash,
            extra_headers=extra,
        )
        url = f"{self.base_url}{canonical_uri}"
        return self._client.request(
            method.upper(),
            url,
            params=query_params or None,
            content=content,
            headers=headers,
        )

    def bucket_exists(self, bucket_name: str) -> bool:
        resp = self._request("HEAD", bucket_name)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise _parse_s3_error_xml(resp.status_code, resp.content)

    def make_bucket(self, bucket_name: str) -> None:
        resp = self._request("PUT", bucket_name)
        if resp.status_code not in (200, 204, 409):
            raise _parse_s3_error_xml(resp.status_code, resp.content)

    def put_object(
        self,
        bucket_name: str,
        object_name: str,
        data: bytes | BinaryIO,
        length: int | None = None,
        content_type: str = "application/octet-stream",
    ) -> dict[str, str]:
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raw = data.read(length) if length is not None and length >= 0 else data.read()
        resp = self._request(
            "PUT",
            bucket_name,
            object_name,
            content=raw,
            content_type=content_type,
        )
        if resp.status_code not in (200, 201, 204):
            raise _parse_s3_error_xml(resp.status_code, resp.content)
        return {"etag": resp.headers.get("etag", "").strip('"')}

    def fput_object(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str | Path,
        content_type: str | None = None,
    ) -> dict[str, str]:
        target = Path(file_path)
        guessed_ct = content_type or mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return self.put_object(
            bucket_name=bucket_name,
            object_name=object_name,
            data=target.read_bytes(),
            content_type=guessed_ct,
        )

    def get_object_bytes(self, bucket_name: str, object_name: str) -> bytes:
        resp = self._request("GET", bucket_name, object_name)
        if resp.status_code != 200:
            raise _parse_s3_error_xml(resp.status_code, resp.content)
        return resp.content

    def fget_object(self, bucket_name: str, object_name: str, file_path: str | Path) -> Path:
        dest = Path(file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.get_object_bytes(bucket_name, object_name))
        return dest

    def remove_object(self, bucket_name: str, object_name: str) -> None:
        resp = self._request("DELETE", bucket_name, object_name)
        if resp.status_code not in (200, 204, 404):
            raise _parse_s3_error_xml(resp.status_code, resp.content)

    def list_objects(
        self,
        bucket_name: str,
        prefix: str = "",
        recursive: bool = True,
    ) -> list[StoredObjectEntry]:
        entries: list[StoredObjectEntry] = []
        continuation: str | None = None

        while True:
            params: dict[str, str] = {"list-type": "2"}
            if prefix:
                params["prefix"] = prefix
            if not recursive:
                params["delimiter"] = "/"
            if continuation:
                params["continuation-token"] = continuation

            resp = self._request("GET", bucket_name, query_params=params)
            if resp.status_code != 200:
                raise _parse_s3_error_xml(resp.status_code, resp.content)

            root = ET.fromstring(resp.content)
            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag == "Contents":
                    key = ""
                    size = 0
                    etag = ""
                    last_mod = ""
                    for child in elem:
                        ctag = child.tag.split("}")[-1]
                        txt = (child.text or "").strip()
                        if ctag == "Key":
                            key = txt
                        elif ctag == "Size" and txt.isdigit():
                            size = int(txt)
                        elif ctag == "ETag":
                            etag = txt.strip('"')
                        elif ctag == "LastModified":
                            last_mod = txt
                    if key:
                        entries.append(
                            StoredObjectEntry(
                                object_name=key,
                                size=size,
                                etag=etag,
                                last_modified=last_mod,
                                is_dir=key.endswith("/"),
                            )
                        )
                elif tag == "CommonPrefixes":
                    for child in elem:
                        if child.tag.split("}")[-1] == "Prefix" and (child.text or "").strip():
                            entries.append(
                                StoredObjectEntry(
                                    object_name=(child.text or "").strip(),
                                    is_dir=True,
                                )
                            )

            is_truncated = False
            next_token = ""
            for child in root:
                ctag = child.tag.split("}")[-1]
                if ctag == "IsTruncated" and (child.text or "").strip().lower() == "true":
                    is_truncated = True
                elif ctag == "NextContinuationToken":
                    next_token = (child.text or "").strip()
            if is_truncated and next_token:
                continuation = next_token
            else:
                break

        return entries


def build_object_store_client(
    endpoint_url: str = "",
    access_key: str = "",
    secret_key: str = "",
    region: str = "",
    transport: httpx.BaseTransport | None = None,
) -> RustFSClient:
    host, secure = parse_storage_endpoint(endpoint_url)
    resolved_access = access_key.strip() or os.getenv("RUSTFS_ACCESS_KEY", DEFAULT_RUSTFS_ACCESS_KEY)
    resolved_secret = secret_key.strip() or os.getenv("RUSTFS_SECRET_KEY", DEFAULT_RUSTFS_SECRET_KEY)
    return RustFSClient(
        endpoint=host,
        access_key=resolved_access,
        secret_key=resolved_secret,
        secure=secure,
        region=region.strip() or os.getenv("RUSTFS_REGION", DEFAULT_RUSTFS_REGION),
        transport=transport,
    )


def build_object_store_client_for_bucket(
    bucket_name: str,
    ctx: TenantContext | None = None,
) -> RustFSClient:
    active = ctx or get_tenant_context()
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM storage_connections
               WHERE bucket_name = ? AND tenant_id = ? AND workspace_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (bucket_name, active.tenant_id, active.workspace_id),
        ).fetchone()
    if row:
        return build_object_store_client(
            endpoint_url=row["endpoint_url"],
            access_key=row["access_key"],
            secret_key=row["secret_key"],
            region=row["region"],
        )
    return build_object_store_client()

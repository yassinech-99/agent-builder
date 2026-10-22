"""Verification suite for RustFS S3-compatible object storage & complete MinIO removal."""

from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, quote, urlparse

from fastapi.testclient import TestClient
import pytest

from agent.context_engine import persist_offloaded_artifact
from api.core.object_store import (
    ObjectStorageError,
    RustFSClient,
    build_object_store_client_for_bucket,
)
from api.core.tenancy import TenantContext, reset_tenant_context, set_tenant_context

REPO_ROOT = Path(__file__).resolve().parent.parent


def _hmac_bytes(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _verify_sigv4_request(
    *,
    method: str,
    raw_path: str,
    headers: dict[str, str],
    body: bytes,
    expected_access_key: str,
    expected_secret_key: str,
) -> tuple[bool, str]:
    auth = headers.get("authorization", "")
    if not auth.startswith("AWS4-HMAC-SHA256 "):
        return False, "Missing AWS4-HMAC-SHA256 header"

    parts = {}
    for chunk in auth[len("AWS4-HMAC-SHA256 ") :].split(","):
        k, _, v = chunk.strip().partition("=")
        parts[k] = v

    cred = parts.get("Credential", "")
    signed_headers_raw = parts.get("SignedHeaders", "")
    client_sig = parts.get("Signature", "")

    cred_split = cred.split("/")
    if len(cred_split) != 5:
        return False, "Malformed Credential scope"
    access_key, date_stamp, region, service, term = cred_split
    if access_key != expected_access_key or service != "s3" or term != "aws4_request":
        return False, "Invalid AccessKeyId"

    parsed = urlparse(raw_path)
    qs_pairs = parse_qs(parsed.query, keep_blank_values=True)
    sorted_q = "&".join(
        f"{quote(k, safe='-_.~')}={quote(qs_pairs[k][0], safe='-_.~')}"
        for k in sorted(qs_pairs.keys())
    )

    signed_header_names = [h.strip().lower() for h in signed_headers_raw.split(";") if h.strip()]
    canonical_headers = "".join(f"{h}:{headers.get(h, '').strip()}\n" for h in signed_header_names)
    payload_hash = hashlib.sha256(body).hexdigest()
    if headers.get("x-amz-content-sha256") != payload_hash:
        return False, "Payload SHA256 mismatch"

    canonical_req = "\n".join(
        [
            method.upper(),
            parsed.path,
            sorted_q,
            canonical_headers,
            ";".join(signed_header_names),
            payload_hash,
        ]
    )
    amz_date = headers.get("x-amz-date", "")
    scope = f"{date_stamp}/{region}/s3/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_req.encode("utf-8")).hexdigest(),
        ]
    )
    k_date = _hmac_bytes(f"AWS4{expected_secret_key}".encode("utf-8"), date_stamp)
    k_region = _hmac_bytes(k_date, region)
    k_service = _hmac_bytes(k_region, "s3")
    signing_key = _hmac_bytes(k_service, "aws4_request")
    expected_sig = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, client_sig):
        return False, "SignatureDoesNotMatch"
    return True, "ok"


@pytest.fixture()
def mock_rustfs_server():
    """Spin up an actual local TCP HTTP server implementing strict AWS SigV4 verification & RustFS S3 API."""
    access_key = "nimagentsadmin"
    secret_key = "nimagentssecretkey"
    buckets: dict[str, dict[str, bytes]] = {"corpus": {}}

    class RustFSHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            return

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or "0")
            return self.rfile.read(length) if length > 0 else b""

        def _send_s3_error(self, status: int, code: str, message: str) -> None:
            xml = (
                f'<?xml version="1.0" encoding="UTF-8"?>'
                f"<Error><Code>{code}</Code><Message>{message}</Message></Error>"
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Length", str(len(xml)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(xml)

        def _authenticate(self, body: bytes) -> bool:
            lowered = {k.lower(): v for k, v in self.headers.items()}
            ok, reason = _verify_sigv4_request(
                method=self.command,
                raw_path=self.path,
                headers=lowered,
                body=body,
                expected_access_key=access_key,
                expected_secret_key=secret_key,
            )
            if not ok:
                self._send_s3_error(403, "SignatureDoesNotMatch", reason)
                return False
            return True

        def _split_bucket_key(self) -> tuple[str, str]:
            parsed = urlparse(self.path)
            trimmed = parsed.path.lstrip("/")
            bucket, _, key = trimmed.partition("/")
            return bucket, key

        def do_HEAD(self):
            if not self._authenticate(b""):
                return
            bucket, key = self._split_bucket_key()
            if not key:
                status = 200 if bucket in buckets else 404
                self.send_response(status)
                self.end_headers()
                return
            status = 200 if bucket in buckets and key in buckets[bucket] else 404
            self.send_response(status)
            self.end_headers()

        def do_PUT(self):
            body = self._read_body()
            if not self._authenticate(body):
                return
            bucket, key = self._split_bucket_key()
            if not key:
                buckets.setdefault(bucket, {})
                self.send_response(200)
                self.end_headers()
                return
            if bucket not in buckets:
                self._send_s3_error(404, "NoSuchBucket", f"Bucket {bucket} missing")
                return
            buckets[bucket][key] = body
            etag = hashlib.md5(body).hexdigest()
            self.send_response(200)
            self.send_header("ETag", f'"{etag}"')
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if not self._authenticate(b""):
                return
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            bucket, key = self._split_bucket_key()
            if bucket not in buckets:
                self._send_s3_error(404, "NoSuchBucket", f"Bucket {bucket} missing")
                return
            if not key and qs.get("list-type") == ["2"]:
                prefix = qs.get("prefix", [""])[0]
                token = int(qs.get("continuation-token", ["0"])[0])
                matching = sorted(k for k in buckets[bucket] if k.startswith(prefix))
                page_size = 2
                page = matching[token : token + page_size]
                next_idx = token + page_size
                truncated = next_idx < len(matching)
                items_xml = "".join(
                    f"<Contents><Key>{k}</Key><Size>{len(buckets[bucket][k])}</Size></Contents>"
                    for k in page
                )
                next_xml = (
                    f"<NextContinuationToken>{next_idx}</NextContinuationToken>"
                    if truncated
                    else ""
                )
                xml = (
                    '<?xml version="1.0" encoding="UTF-8"?>'
                    '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                    f"<Name>{bucket}</Name>"
                    f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"
                    f"{next_xml}"
                    f"{items_xml}"
                    "</ListBucketResult>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(xml)))
                self.end_headers()
                self.wfile.write(xml)
                return

            if key not in buckets[bucket]:
                self._send_s3_error(404, "NoSuchKey", f"Key {key} missing")
                return
            raw = buckets[bucket][key]
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_DELETE(self):
            if not self._authenticate(b""):
                return
            bucket, key = self._split_bucket_key()
            if bucket in buckets:
                buckets[bucket].pop(key, None)
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), RustFSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield {
            "endpoint_url": f"http://{host}:{port}",
            "host_port": f"{host}:{port}",
            "access_key": access_key,
            "secret_key": secret_key,
            "buckets": buckets,
        }
    finally:
        server.shutdown()
        server.server_close()


def test_complete_minio_removal_and_rustfs_docker_configuration():
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "rustfs:" in compose
    assert "image: rustfs/rustfs:latest" in compose
    assert "RUSTFS_VOLUMES: /data" in compose
    assert "RUSTFS_CONSOLE_ENABLE:" in compose
    assert "nimagents_rustfs_data:" in compose
    assert "minio" not in compose.lower()

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "minio" not in pyproject.lower()

    uv_lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "minio"' not in uv_lock


def test_rustfs_sigv4_client_crud_pagination_and_auth_enforcement(mock_rustfs_server, tmp_path):
    client = RustFSClient(
        endpoint=mock_rustfs_server["host_port"],
        access_key=mock_rustfs_server["access_key"],
        secret_key=mock_rustfs_server["secret_key"],
        secure=False,
    )
    try:
        assert client.bucket_exists("corpus") is True
        assert client.bucket_exists("missing-bucket") is False

        client.make_bucket("artifacts")
        assert client.bucket_exists("artifacts") is True

        # Put 3 objects to exercise multi-page ListObjectsV2 continuation tokens (`page_size=2`)
        for idx in range(3):
            client.put_object(
                "corpus",
                f"tenants/acme/workspaces/prod/doc_{idx}.md",
                f"# Doc {idx}\nRustFS content {idx}".encode("utf-8"),
                content_type="text/markdown",
            )

        listed = client.list_objects("corpus", prefix="tenants/acme/workspaces/prod/")
        assert [o.object_name for o in listed] == [
            "tenants/acme/workspaces/prod/doc_0.md",
            "tenants/acme/workspaces/prod/doc_1.md",
            "tenants/acme/workspaces/prod/doc_2.md",
        ]

        dl_path = tmp_path / "downloaded.md"
        client.fget_object("corpus", "tenants/acme/workspaces/prod/doc_1.md", dl_path)
        assert dl_path.read_text(encoding="utf-8") == "# Doc 1\nRustFS content 1"

        client.remove_object("corpus", "tenants/acme/workspaces/prod/doc_2.md")
        after_delete = client.list_objects("corpus", prefix="tenants/acme/workspaces/prod/")
        assert len(after_delete) == 2

        # Wrong secret key MUST fail SigV4 check with 403 & XML `SignatureDoesNotMatch`
        bad_client = RustFSClient(
            endpoint=mock_rustfs_server["host_port"],
            access_key=mock_rustfs_server["access_key"],
            secret_key="wrong-secret-key",
            secure=False,
        )
        try:
            with pytest.raises(ObjectStorageError) as head_exc:
                bad_client.bucket_exists("corpus")
            assert head_exc.value.status_code == 403

            with pytest.raises(ObjectStorageError) as list_exc:
                bad_client.list_objects("corpus")
            assert list_exc.value.status_code == 403
            assert list_exc.value.code == "SignatureDoesNotMatch"
        finally:
            bad_client.close()
    finally:
        client.close()


def test_storage_api_live_validation_and_context_artifact_rustfs_upload(
    mock_rustfs_server, tmp_path, monkeypatch
):
    db_file = tmp_path / "rustfs_api_test.db"
    monkeypatch.setenv("NIMAGENTS_DB_PATH", str(db_file))
    from api import db
    from api.app import app

    db.init_db()
    with TestClient(app) as api_client:
        headers = {"X-Tenant-ID": "org_rust", "X-Workspace-ID": "ws_alpha"}
        # Non-existent bucket is rejected during live SigV4 check (`skip_validation=False`)
        missing_res = api_client.post(
            "/api/storage",
            headers=headers,
            json={
                "name": "Missing Bucket",
                "endpoint_url": mock_rustfs_server["endpoint_url"],
                "bucket_name": "does-not-exist",
                "access_key": mock_rustfs_server["access_key"],
                "secret_key": mock_rustfs_server["secret_key"],
                "skip_validation": False,
            },
        )
        assert missing_res.status_code == 400

        # Valid RustFS bucket passes live SigV4 `HEAD /corpus` check AND masks credentials
        create_res = api_client.post(
            "/api/storage",
            headers=headers,
            json={
                "name": "Primary RustFS Cluster",
                "endpoint_url": mock_rustfs_server["endpoint_url"],
                "bucket_name": "corpus",
                "access_key": mock_rustfs_server["access_key"],
                "secret_key": mock_rustfs_server["secret_key"],
                "skip_validation": False,
            },
        )
        assert create_res.status_code == 201, create_res.text
        created = create_res.json()
        assert created["scoped_prefix"] == "tenants/org_rust/workspaces/ws_alpha/"
        assert mock_rustfs_server["secret_key"] not in str(created)

        # Live `/test` endpoint succeeds via RustFS SigV4
        ping_res = api_client.post(f"/api/storage/{created['id']}/test", headers=headers)
        assert ping_res.status_code == 200
        assert ping_res.json()["ok"] is True

        # Stage 1 Context Offloading uploads large artifacts directly into tenant-isolated RustFS prefix
        ctx = TenantContext(tenant_id="org_rust", workspace_id="ws_alpha")
        tok = set_tenant_context(ctx)
        try:
            art = persist_offloaded_artifact(
                raw_text="A" * 16_000,
                tool_name="search_logs",
                thread_id="t_rustfs",
                ctx=ctx,
            )
            assert art["storage_uri"].startswith(
                "s3://corpus/tenants/org_rust/workspaces/ws_alpha/artifacts/"
            )
            expected_key = f"tenants/org_rust/workspaces/ws_alpha/artifacts/{art['id']}.txt"
            assert expected_key in mock_rustfs_server["buckets"]["corpus"]
        finally:
            reset_tenant_context(tok)

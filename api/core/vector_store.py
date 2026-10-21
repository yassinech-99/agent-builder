"""Multi-Tenant Qdrant Vector Engine (`is_tenant=True` Co-Located Shards + `INT8` Scalar Quantization).

Replaces legacy per-query in-memory disk pickle vector indices with Qdrant's multi-tenant engine:
  - One shared collection per embedding vector dimension (`nimagents_{namespace}_dim_{dim}`).
  - `KeywordIndexParams(type=KEYWORD, is_tenant=True)` physically co-locating each tenant's HNSW blocks
    so small tenants never suffer HNSW post-filter empty results at 10,000+ tenant scale.
  - `ScalarQuantization(INT8, quantile=0.99, always_ram=True)` cutting vector RAM usage by 4x.
  - True non-blocking dual mode via `_ThreadPooledLocalQdrantAdapter`:
      * Remote cluster `QDRANT_URL`: native async `AsyncQdrantClient` over HTTP/gRPC.
      * Embedded local `path`: thread-pooled `QdrantClient` serialized by `threading.RLock` so
        embedded SQLite `fsync` commits NEVER stall the main `asyncio` event loop under burst load!
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
import threading
from typing import Any
import uuid
import warnings

from qdrant_client import AsyncQdrantClient, QdrantClient, models

from api.core.tenancy import TenantContext, get_tenant_context, sanitize_tenant_identifier

logger = logging.getLogger("nimagents.core.vector_store")


def default_qdrant_path() -> Path:
    custom = os.getenv("NIMAGENTS_QDRANT_PATH")
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent.parent.parent / "data" / "qdrant"


class _ThreadPooledLocalQdrantAdapter:
    """Async adapter wrapping embedded `QdrantClient` on worker threads so SQLite `fsync` never blocks `asyncio`."""

    def __init__(self, sync_client: QdrantClient, rlock: threading.RLock) -> None:
        self._sync = sync_client
        self._rlock = rlock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._sync, name)
        if not callable(attr):
            return attr

        async def _async_call(*args: Any, **kwargs: Any) -> Any:
            def _locked() -> Any:
                with self._rlock:
                    return attr(*args, **kwargs)

            return await asyncio.to_thread(_locked)

        return _async_call


class MultiTenantVectorStore:
    """Process-wide non-blocking Qdrant client manager enforcing strict tenant/workspace/source partitioning."""

    def __init__(self, url: str | None = None, local_path: Path | None = None) -> None:
        self._explicit_url = url
        self._explicit_path = local_path
        self._client: Any = None
        self._local_rlock = threading.RLock()
        self._client_key: str = ""
        self._ready_collections: set[str] = set()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None
        self.total_upserts: int = 0
        self.total_queries: int = 0

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _desired_backend_key(self) -> tuple[str, str]:
        url = (self._explicit_url if self._explicit_url is not None else os.getenv("QDRANT_URL", "")).strip()
        if url:
            return ("remote", url)
        path = str((self._explicit_path or default_qdrant_path()).resolve())
        return ("local", path)

    async def get_client(self) -> Any:
        mode, target = self._desired_backend_key()
        key = f"{mode}:{target}"
        if self._client is not None and self._client_key == key:
            return self._client

        async with self._get_lock():
            if self._client is not None and self._client_key == key:
                return self._client
            if self._client is not None:
                await self._client.close()
                self._ready_collections.clear()

            if mode == "remote":
                api_key = os.getenv("QDRANT_API_KEY") or None
                self._client = AsyncQdrantClient(url=target, api_key=api_key, timeout=15)
            else:
                def _open_local() -> QdrantClient:
                    with self._local_rlock:
                        Path(target).mkdir(parents=True, exist_ok=True)
                        return QdrantClient(path=target)

                sync_q = await asyncio.to_thread(_open_local)
                self._client = _ThreadPooledLocalQdrantAdapter(sync_q, self._local_rlock)

            self._client_key = key
            return self._client

    @staticmethod
    def collection_name_for_dim(dim: int, namespace: str = "rag") -> str:
        if dim <= 0:
            raise ValueError(f"Invalid vector dimension: {dim}")
        return f"nimagents_{namespace}_dim_{dim}"

    async def ensure_collection(self, dim: int, namespace: str = "rag") -> str:
        client = await self.get_client()
        collection_name = self.collection_name_for_dim(dim, namespace=namespace)
        if collection_name in self._ready_collections:
            return collection_name

        async with self._get_lock():
            if collection_name in self._ready_collections:
                return collection_name

            if not await client.collection_exists(collection_name):
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dim,
                        distance=models.Distance.COSINE,
                        on_disk=True,
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True,
                        )
                    ),
                    on_disk_payload=True,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    await client.create_payload_index(
                        collection_name=collection_name,
                        field_name="tenant_id",
                        field_schema=models.KeywordIndexParams(
                            type=models.KeywordIndexType.KEYWORD,
                            is_tenant=True,
                        ),
                    )
                    for indexed_field in ("workspace_id", "source_id", "cache_partition"):
                        await client.create_payload_index(
                            collection_name=collection_name,
                            field_name=indexed_field,
                            field_schema=models.PayloadSchemaType.KEYWORD,
                        )

            self._ready_collections.add(collection_name)
            return collection_name

    async def upsert_chunks(
        self,
        *,
        source_id: str,
        vectors: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        namespace: str = "rag",
        ctx: TenantContext | None = None,
    ) -> int:
        if not vectors or not texts:
            return 0
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts length mismatch")

        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        sid = sanitize_tenant_identifier(source_id)

        dim = len(vectors[0])
        collection = await self.ensure_collection(dim, namespace=namespace)
        client = await self.get_client()

        metas = metadatas or [{} for _ in texts]
        points: list[models.PointStruct] = []
        for idx, (vec, content, meta) in enumerate(zip(vectors, texts, metas)):
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{namespace}:{tid}:{wid}:{sid}:{idx}"))
            points.append(
                models.PointStruct(
                    id=point_uuid,
                    vector=vec,
                    payload={
                        "tenant_id": tid,
                        "workspace_id": wid,
                        "source_id": sid,
                        "chunk_index": idx,
                        "page_content": content,
                        "metadata": meta or {},
                    },
                )
            )

        batch_size = 128
        for start in range(0, len(points), batch_size):
            await client.upsert(
                collection_name=collection,
                points=points[start : start + batch_size],
            )

        self.total_upserts += len(points)
        return len(points)

    async def search(
        self,
        *,
        source_id: str | None = None,
        query_vector: list[float],
        k: int = 4,
        namespace: str = "rag",
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        if not query_vector:
            return []

        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.tenant_id if active.workspace_id is None else active.workspace_id)

        dim = len(query_vector)
        collection = self.collection_name_for_dim(dim, namespace=namespace)
        client = await self.get_client()
        if not await client.collection_exists(collection):
            return []

        must_conditions = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tid)),
            models.FieldCondition(key="workspace_id", match=models.MatchValue(value=wid)),
        ]
        if source_id is not None:
            sid = sanitize_tenant_identifier(source_id)
            must_conditions.append(
                models.FieldCondition(key="source_id", match=models.MatchValue(value=sid))
            )

        flt = models.Filter(must=must_conditions)
        res = await client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=flt,
            limit=k,
            with_payload=True,
        )
        self.total_queries += 1
        out: list[dict[str, Any]] = []
        for pt in res.points:
            payload = pt.payload or {}
            out.append(
                {
                    "page_content": payload.get("page_content", ""),
                    "metadata": payload.get("metadata", {}),
                    "score": float(pt.score or 0.0),
                }
            )
        return out

    async def delete_source(
        self,
        *,
        source_id: str,
        dim: int | None = None,
        namespace: str = "rag",
        ctx: TenantContext | None = None,
    ) -> None:
        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        sid = sanitize_tenant_identifier(source_id)

        client = await self.get_client()
        collections_resp = await client.get_collections()
        prefix = f"nimagents_{namespace}_dim_"
        target_names = [
            c.name
            for c in collections_resp.collections
            if (dim is None and c.name.startswith(prefix))
            or (dim is not None and c.name == self.collection_name_for_dim(dim, namespace=namespace))
        ]
        flt = models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tid)),
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=wid)),
                models.FieldCondition(key="source_id", match=models.MatchValue(value=sid)),
            ]
        )
        for cname in target_names:
            await client.delete(
                collection_name=cname,
                points_selector=models.FilterSelector(filter=flt),
            )

    async def status_summary(self) -> dict[str, Any]:
        mode, target = self._desired_backend_key()
        collections: list[str] = []
        try:
            client = await self.get_client()
            resp = await client.get_collections()
            collections = [c.name for c in resp.collections]
        except Exception:
            pass
        return {
            "backend_mode": mode,
            "target": target if mode == "remote" else "embedded-qdrant",
            "quantization": "INT8",
            "tenant_payload_index": True,
            "collections": collections,
            "total_upserts": self.total_upserts,
            "total_queries": self.total_queries,
        }

    async def get_stats(self) -> dict[str, Any]:
        return await self.status_summary()

    async def close(self) -> None:
        await self.reset_for_tests()

    async def reset_for_tests(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None
        self._client_key = ""
        self._ready_collections.clear()
        self.total_upserts = 0
        self.total_queries = 0


vector_store = MultiTenantVectorStore()

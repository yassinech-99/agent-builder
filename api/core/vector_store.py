"""Multi-Tenant Qdrant Vector Engine (`is_tenant=True` Co-Located Shards + `INT8` Scalar Quantization).

Replaces legacy per-query in-memory disk pickle vector indices with Qdrant's multi-tenant engine:
  - One shared collection per embedding vector dimension (`nimagents_dim_{dim}`).
  - `KeywordIndexParams(type=KEYWORD, is_tenant=True)` physically co-locating each tenant's HNSW blocks
    so small tenants never suffer HNSW post-filter empty results at 10,000+ tenant scale.
  - `ScalarQuantization(INT8, quantile=0.99, always_ram=True)` cutting vector RAM usage by 4x.
  - Seamless dual mode: connects to cluster `QDRANT_URL` (HTTP/gRPC) OR persistent embedded local
    `AsyncQdrantClient(path=...)` when `QDRANT_URL` is unset.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
import uuid
import warnings

from qdrant_client import AsyncQdrantClient, models

from api.core.tenancy import TenantContext, get_tenant_context, sanitize_tenant_identifier

logger = logging.getLogger("nimagents.core.vector_store")


def default_qdrant_path() -> Path:
    custom = os.getenv("NIMAGENTS_QDRANT_PATH")
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent.parent.parent / "data" / "qdrant"


class MultiTenantVectorStore:
    """Process-wide async Qdrant client manager enforcing strict tenant/workspace/source partitioning."""

    def __init__(self, url: str | None = None, local_path: Path | None = None) -> None:
        self._explicit_url = url
        self._explicit_path = local_path
        self._client: AsyncQdrantClient | None = None
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

    async def get_client(self) -> AsyncQdrantClient:
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
                Path(target).mkdir(parents=True, exist_ok=True)
                self._client = AsyncQdrantClient(path=target)
            self._client_key = key
            return self._client

    @staticmethod
    def collection_name_for_dim(dim: int, namespace: str = "rag") -> str:
        if dim <= 0:
            raise ValueError(f"Invalid vector dimension: {dim}")
        return f"nimagents_{namespace}_dim_{dim}"

    async def ensure_collection(self, dim: int, namespace: str = "rag") -> str:
        collection_name = self.collection_name_for_dim(dim, namespace=namespace)
        if collection_name in self._ready_collections:
            return collection_name

        client = await self.get_client()
        async with self._get_lock():
            if collection_name in self._ready_collections:
                return collection_name

            exists = await client.collection_exists(collection_name)
            if not exists:
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
        collection = await self.ensure_collection(dim, namespace="rag")
        client = await self.get_client()

        metas = metadatas or [{} for _ in texts]
        points: list[models.PointStruct] = []
        for idx, (vec, content, meta) in enumerate(zip(vectors, texts, metas)):
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{tid}:{wid}:{sid}:{idx}"))
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
        source_id: str,
        query_vector: list[float],
        k: int = 4,
        ctx: TenantContext | None = None,
    ) -> list[dict[str, Any]]:
        if not query_vector:
            return []

        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        sid = sanitize_tenant_identifier(source_id)

        dim = len(query_vector)
        collection = self.collection_name_for_dim(dim, namespace="rag")
        client = await self.get_client()
        if not await client.collection_exists(collection):
            return []

        must_conditions: list[models.Condition] = [
            models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tid)),
            models.FieldCondition(key="workspace_id", match=models.MatchValue(value=wid)),
            models.FieldCondition(key="source_id", match=models.MatchValue(value=sid)),
        ]

        res = await client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=models.Filter(must=must_conditions),
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
        ctx: TenantContext | None = None,
    ) -> None:
        active = ctx or get_tenant_context()
        tid = sanitize_tenant_identifier(active.tenant_id)
        wid = sanitize_tenant_identifier(active.workspace_id)
        sid = sanitize_tenant_identifier(source_id)

        client = await self.get_client()
        collections_resp = await client.get_collections()
        target_names = [
            c.name
            for c in collections_resp.collections
            if (dim is None and c.name.startswith("nimagents_rag_dim_"))
            or (dim is not None and c.name == self.collection_name_for_dim(dim, namespace="rag"))
        ]
        flt = models.Filter(
            must=[
                models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tid)),
                models.FieldCondition(key="workspace_id", match=models.MatchValue(value=wid)),
                models.FieldCondition(key="source_id", match=models.MatchValue(value=sid)),
            ]
        )
        for col in target_names:
            await client.delete(collection_name=col, points_selector=models.FilterSelector(filter=flt))

    async def status_summary(self) -> dict[str, Any]:
        mode, target = self._desired_backend_key()
        client = await self.get_client()
        cols = await client.get_collections()
        return {
            "mode": mode,
            "target": target,
            "collections": [c.name for c in cols.collections],
            "total_upserts": self.total_upserts,
            "total_queries": self.total_queries,
            "quantization": "int8_scalar",
            "tenant_index": "is_tenant_keyword",
        }

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._ready_collections.clear()


vector_store = MultiTenantVectorStore()

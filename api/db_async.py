"""Async Multi-Tenant Database Layer (`PostgreSQL 17 RLS + PgBouncer` & `aiosqlite` Dual Mode).

Key production guarantees:
1. PgBouncer `transaction` pooling safety (`statement_cache_size=0`, `pool_pre_ping=True`,
   and explicit validation of `replicas * pool_size <= 0.7 * postgres_max_connections`).
2. Automatic PostgreSQL Row-Level Security binding on every transaction:
      `SELECT set_config('app.current_tenant_id', :tenant_id, true)`
      `SELECT set_config('app.current_workspace_id', :workspace_id, true)`
   Even if a SQL query accidentally omits a `WHERE` clause on Postgres, RLS isolates rows.
3. Seamless `sqlite+aiosqlite` local fallback honoring the exact same SQLAlchemy `text()` interface.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from api.core.tenancy import TenantContext, get_tenant_context
from api.db import get_db_path

logger = logging.getLogger("nimagents.db.async")

_engines: dict[str, AsyncEngine] = {}
_session_factories: dict[str, async_sessionmaker[AsyncSession]] = {}


def check_pgbouncer_pool_budget(
    app_replicas: int,
    pool_size: int,
    postgres_max_connections: int,
) -> dict[str, Any]:
    """Validate `replicas * pool_size <= 0.70 * postgres_max_connections` scaling budget."""
    total_requested = max(1, app_replicas) * max(1, pool_size)
    safe_ceiling = int(max(10, postgres_max_connections) * 0.70)
    within_budget = total_requested <= safe_ceiling
    return {
        "app_replicas": app_replicas,
        "pool_size": pool_size,
        "total_pool_connections": total_requested,
        "postgres_max_connections": postgres_max_connections,
        "safe_ceiling": safe_ceiling,
        "within_budget": within_budget,
    }


def resolve_database_url() -> str:
    """Resolve `DATABASE_URL` (`postgresql+asyncpg://...`) or fallback to `sqlite+aiosqlite`."""
    raw = os.getenv("DATABASE_URL", "").strip()
    if raw:
        if raw.startswith("postgres://"):
            return raw.replace("postgres://", "postgresql+asyncpg://", 1)
        if raw.startswith("postgresql://"):
            return raw.replace("postgresql://", "postgresql+asyncpg://", 1)
        return raw
    db_file = get_db_path()
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_file}"


def get_async_engine(url: str | None = None) -> AsyncEngine:
    """Create or reuse a pooled `AsyncEngine` configured for PgBouncer or aiosqlite."""
    target_url = url or resolve_database_url()
    cached = _engines.get(target_url)
    if cached is not None:
        return cached

    if target_url.startswith("postgresql+asyncpg://"):
        replicas = int(os.getenv("NIMAGENTS_APP_REPLICAS", "2"))
        pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "10"))
        pg_max_conns = int(os.getenv("POSTGRES_MAX_CONNECTIONS", "200"))
        budget = check_pgbouncer_pool_budget(replicas, pool_size + max_overflow, pg_max_conns)
        if not budget["within_budget"]:
            logger.warning(
                "DB pool (%d total across %d replicas) exceeds 70%% of Postgres max_connections (%d); "
                "route traffic through PgBouncer in transaction mode.",
                budget["total_pool_connections"],
                replicas,
                pg_max_conns,
            )
        engine = create_async_engine(
            target_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
            # Mandatory for PgBouncer `transaction` pooling mode!
            connect_args={
                "statement_cache_size": 0,
                "prepared_statement_cache_size": 0,
            },
        )
    else:
        engine = create_async_engine(
            target_url,
            connect_args={"timeout": 10.0},
        )

    _engines[target_url] = engine
    _session_factories[target_url] = async_sessionmaker(
        engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    return engine


async def apply_postgres_rls_context(
    conn_or_session: AsyncConnection | AsyncSession,
    ctx: TenantContext | None = None,
) -> tuple[str, str]:
    """Bind `app.current_tenant_id` & `app.current_workspace_id` via PostgreSQL `set_config(..., true)`.

    `is_local = true` scopes the setting strictly to the current transaction so PgBouncer
    connection reuse across tenants can NEVER leak session GUCs!
    """
    active = ctx or get_tenant_context()
    await conn_or_session.execute(
        text("SELECT set_config('app.current_tenant_id', :tid, true)"),
        {"tid": active.tenant_id},
    )
    await conn_or_session.execute(
        text("SELECT set_config('app.current_workspace_id', :wid, true)"),
        {"wid": active.workspace_id},
    )
    return active.tenant_id, active.workspace_id


@asynccontextmanager
async def async_db_session(
    ctx: TenantContext | None = None,
) -> AsyncGenerator[AsyncSession, None]:
    """Open a transaction-scoped `AsyncSession` with automatic PostgreSQL RLS binding."""
    target_url = resolve_database_url()
    engine = get_async_engine(target_url)
    factory = _session_factories[target_url]
    active = ctx or get_tenant_context()

    async with factory() as session:
        async with session.begin():
            if engine.dialect.name == "postgresql":
                await apply_postgres_rls_context(session, active)
            yield session


async def close_async_engines() -> None:
    for eng in list(_engines.values()):
        await eng.dispose()
    _engines.clear()
    _session_factories.clear()

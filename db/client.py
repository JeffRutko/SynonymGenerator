"""Async PostgreSQL client (Neon) for syn_* cache tables."""

from __future__ import annotations

import logging
from pathlib import Path

import asyncpg
from pgvector.asyncpg import register_vector

from models import models

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None
MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

SYN_SOURCE_DOCUMENTS = "syn_source_documents"
SYN_CHUNK_VECTORS = "syn_chunk_vectors"
SYN_SYNONYM_OUTPUTS = "syn_synonym_outputs"


async def _init_connection(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def connect() -> None:
    """Open the connection pool and ensure syn_* tables exist."""
    global _pool
    if not models.DB_ENABLED:
        return
    if _pool is not None:
        return

    pool = await asyncpg.create_pool(
        models.DATABASE_URL,
        min_size=1,
        max_size=5,
        command_timeout=30,
        init=_init_connection,
    )
    try:
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        _pool = pool
        await _apply_migrations()
        logger.info("PostgreSQL connected (syn_* tables)")
    except Exception:
        await pool.close()
        _pool = None
        raise


async def disconnect() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database is not connected; call connect() first")
    return _pool


async def ping() -> bool:
    """Return True if PostgreSQL is reachable (retries connect if needed)."""
    if not models.DB_ENABLED:
        return False
    if _pool is None:
        try:
            await connect()
        except Exception as exc:
            logger.warning("PostgreSQL reconnect failed: %s", exc)
            return False
    try:
        async with _pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        await disconnect()
        return False


async def _apply_migrations() -> None:
    migration_file = MIGRATIONS_DIR / "001_syn_tables.sql"
    if not migration_file.is_file():
        logger.warning("Migration file not found: %s", migration_file)
        return
    sql = migration_file.read_text(encoding="utf-8")
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    pool = get_pool()
    async with pool.acquire() as conn:
        for statement in statements:
            await conn.execute(statement)
    logger.info("Applied migration %s", migration_file.name)

"""Raw MCP search results cache."""

from __future__ import annotations

from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_sql
from db.client import SYN_SOURCE_DOCUMENTS, get_pool
from models import models
from rag.constants import GATHER_TOOLS


async def get_cached_raw_text(query_key: str, tool_name: str) -> str | None:
    if not models.DB_ENABLED:
        return None
    row = await get_pool().fetchrow(
        f"""
        SELECT raw_text FROM {SYN_SOURCE_DOCUMENTS}
        WHERE query_key = $1 AND tool_name = $2 AND {not_expired_sql()}
        """,
        query_key,
        tool_name,
    )
    if not row:
        return None
    text = row["raw_text"]
    return str(text) if text else None


async def has_all_cached(query_key: str) -> bool:
    if not models.DB_ENABLED:
        return False
    row = await get_pool().fetchrow(
        f"""
        SELECT COUNT(*) AS n FROM {SYN_SOURCE_DOCUMENTS}
        WHERE query_key = $1
          AND tool_name = ANY($2::text[])
          AND {not_expired_sql()}
        """,
        query_key,
        list(GATHER_TOOLS),
    )
    return int(row["n"]) == len(GATHER_TOOLS) if row else False


async def upsert(
    *,
    query_key: str,
    concept: str,
    context: str,
    tool_name: str,
    raw_text: str,
) -> None:
    if not models.DB_ENABLED:
        return
    now = datetime.now(UTC)
    await get_pool().execute(
        f"""
        INSERT INTO {SYN_SOURCE_DOCUMENTS} (
            query_key, tool_name, concept, context, raw_text, created_at, expires_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (query_key, tool_name) DO UPDATE SET
            concept = EXCLUDED.concept,
            context = EXCLUDED.context,
            raw_text = EXCLUDED.raw_text,
            created_at = EXCLUDED.created_at,
            expires_at = EXCLUDED.expires_at
        """,
        query_key,
        tool_name,
        concept,
        context,
        raw_text,
        now,
        cache_expires_at(),
    )

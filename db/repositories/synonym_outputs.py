"""Final synonym report cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_sql
from db.client import SYN_SYNONYM_OUTPUTS, get_pool
from models import models


@dataclass(frozen=True)
class CachedSynonymOutput:
    answer: str
    progress: str
    tools_used: list[str]


async def get_cached(query_key: str) -> CachedSynonymOutput | None:
    if not models.DB_ENABLED:
        return None
    row = await get_pool().fetchrow(
        f"""
        SELECT answer, progress, tools_used FROM {SYN_SYNONYM_OUTPUTS}
        WHERE query_key = $1 AND {not_expired_sql()}
        """,
        query_key,
    )
    if not row:
        return None
    answer = row["answer"]
    if not answer:
        return None
    return CachedSynonymOutput(
        answer=str(answer),
        progress=str(row["progress"] or ""),
        tools_used=list(row["tools_used"] or []),
    )


async def upsert(
    *,
    query_key: str,
    concept: str,
    context: str,
    answer: str,
    progress: str,
    tools_used: list[str],
) -> None:
    if not models.DB_ENABLED:
        return
    now = datetime.now(UTC)
    await get_pool().execute(
        f"""
        INSERT INTO {SYN_SYNONYM_OUTPUTS} (
            query_key, concept, context, answer, progress, tools_used,
            created_at, expires_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (query_key) DO UPDATE SET
            concept = EXCLUDED.concept,
            context = EXCLUDED.context,
            answer = EXCLUDED.answer,
            progress = EXCLUDED.progress,
            tools_used = EXCLUDED.tools_used,
            created_at = EXCLUDED.created_at,
            expires_at = EXCLUDED.expires_at
        """,
        query_key,
        concept,
        context,
        answer,
        progress,
        tools_used,
        now,
        cache_expires_at(),
    )

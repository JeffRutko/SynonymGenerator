"""Final synonym report cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_filter
from db.client import SYNONYM_OUTPUTS, get_db
from models import models


@dataclass(frozen=True)
class CachedSynonymOutput:
    answer: str
    progress: str
    tools_used: list[str]


async def get_cached(query_key: str) -> CachedSynonymOutput | None:
    if not models.MONGODB_ENABLED:
        return None
    doc = await get_db()[SYNONYM_OUTPUTS].find_one(
        {"query_key": query_key, **not_expired_filter()},
    )
    if not doc:
        return None
    answer = doc.get("answer")
    if not answer:
        return None
    return CachedSynonymOutput(
        answer=str(answer),
        progress=str(doc.get("progress") or ""),
        tools_used=list(doc.get("tools_used") or []),
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
    if not models.MONGODB_ENABLED:
        return
    now = datetime.now(UTC)
    await get_db()[SYNONYM_OUTPUTS].update_one(
        {"query_key": query_key},
        {
            "$set": {
                "query_key": query_key,
                "concept": concept,
                "context": context,
                "answer": answer,
                "progress": progress,
                "tools_used": tools_used,
                "created_at": now,
                "expires_at": cache_expires_at(),
            }
        },
        upsert=True,
    )

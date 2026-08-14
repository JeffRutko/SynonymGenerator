"""Raw MCP search results cache."""

from __future__ import annotations

from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_filter
from db.client import SOURCE_DOCUMENTS, get_db
from models import models
from rag.constants import GATHER_TOOLS


async def get_cached_raw_text(query_key: str, tool_name: str) -> str | None:
    if not models.MONGODB_ENABLED:
        return None
    doc = await get_db()[SOURCE_DOCUMENTS].find_one(
        {
            "query_key": query_key,
            "tool_name": tool_name,
            **not_expired_filter(),
        },
        projection={"raw_text": 1},
    )
    if not doc:
        return None
    text = doc.get("raw_text")
    return str(text) if text else None


async def has_all_cached(query_key: str) -> bool:
    if not models.MONGODB_ENABLED:
        return False
    count = await get_db()[SOURCE_DOCUMENTS].count_documents(
        {
            "query_key": query_key,
            "tool_name": {"$in": list(GATHER_TOOLS)},
            **not_expired_filter(),
        }
    )
    return count == len(GATHER_TOOLS)


async def upsert(
    *,
    query_key: str,
    concept: str,
    context: str,
    tool_name: str,
    raw_text: str,
) -> None:
    if not models.MONGODB_ENABLED:
        return
    now = datetime.now(UTC)
    await get_db()[SOURCE_DOCUMENTS].update_one(
        {"query_key": query_key, "tool_name": tool_name},
        {
            "$set": {
                "query_key": query_key,
                "concept": concept,
                "context": context,
                "tool_name": tool_name,
                "raw_text": raw_text,
                "created_at": now,
                "expires_at": cache_expires_at(),
            }
        },
        upsert=True,
    )

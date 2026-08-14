"""Async MongoDB client (Motor) for Atlas."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from models import models
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

if TYPE_CHECKING:
    from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

_client: AsyncIOMotorClient | None = None
SERVER_SELECTION_TIMEOUT_MS = 10_000

SOURCE_DOCUMENTS = "source_documents"
CHUNK_VECTORS = "chunk_vectors"
SYNONYM_OUTPUTS = "synonym_outputs"


async def connect() -> None:
    """Open the Motor client and ensure collection indexes."""
    global _client
    if not models.MONGODB_ENABLED:
        return
    if _client is not None:
        return

    _client = AsyncIOMotorClient(
        models.MONGODB_URI,
        serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS,
    )
    await _client.admin.command("ping")
    await _ensure_indexes()
    logger.info("MongoDB connected (database=%s)", models.MONGODB_DB_NAME)


async def disconnect() -> None:
    """Close the Motor client."""
    global _client
    if _client is None:
        return
    _client.close()
    _client = None


def get_client() -> AsyncIOMotorClient:
    if _client is None:
        raise RuntimeError("MongoDB is not connected; call connect() first")
    return _client


def get_db() -> AsyncIOMotorDatabase:
    return get_client()[models.MONGODB_DB_NAME]


async def ping() -> bool:
    """Return True if MongoDB is reachable."""
    if not models.MONGODB_ENABLED:
        return False
    if _client is None:
        return False
    try:
        await _client.admin.command("ping")
        return True
    except Exception:
        return False


async def _ensure_indexes() -> None:
    db = get_db()

    await db[SOURCE_DOCUMENTS].create_index(
        [("query_key", 1), ("tool_name", 1)],
        unique=True,
        name="query_key_tool_name_unique",
    )
    await db[SYNONYM_OUTPUTS].create_index(
        [("query_key", 1)],
        unique=True,
        name="query_key_unique",
    )
    await db[SYNONYM_OUTPUTS].create_index(
        [("created_at", -1)],
        name="created_at_desc",
    )
    await db[CHUNK_VECTORS].create_index(
        [("query_key", 1), ("chunk_id", 1)],
        unique=True,
        name="query_key_chunk_id_unique",
    )
    await db[CHUNK_VECTORS].create_index(
        [("query_key", 1)],
        name="query_key",
    )

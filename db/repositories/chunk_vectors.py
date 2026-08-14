"""Chunk embeddings cache and Atlas Vector Search."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_filter
from db.client import CHUNK_VECTORS, get_db
from models import models
from rag.chunking import TextChunk
from rag.store import RetrievedChunk

logger = logging.getLogger(__name__)


def chunk_id_for(chunk: TextChunk) -> str:
    return f"{chunk.source_tool}-{chunk.index}"


async def has_vectors(query_key: str) -> bool:
    if not models.MONGODB_ENABLED:
        return False
    doc = await get_db()[CHUNK_VECTORS].find_one(
        {"query_key": query_key, **not_expired_filter()},
        projection={"_id": 1},
    )
    return doc is not None


async def upsert_chunks(
    query_key: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> int:
    if not models.MONGODB_ENABLED or not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    now = datetime.now(UTC)
    collection = get_db()[CHUNK_VECTORS]
    count = 0
    for chunk, embedding in zip(chunks, embeddings):
        await collection.update_one(
            {"query_key": query_key, "chunk_id": chunk_id_for(chunk)},
            {
                "$set": {
                    "query_key": query_key,
                    "chunk_id": chunk_id_for(chunk),
                    "text": chunk.text,
                    "source_tool": chunk.source_tool,
                    "query": chunk.query,
                    "index": chunk.index,
                    "embedding": embedding,
                    "created_at": now,
                    "expires_at": cache_expires_at(),
                }
            },
            upsert=True,
        )
        count += 1
    return count


async def delete_by_query_key(query_key: str) -> int:
    """Remove all cached vectors for a query (e.g. on force_refresh)."""
    if not models.MONGODB_ENABLED:
        return 0
    result = await get_db()[CHUNK_VECTORS].delete_many({"query_key": query_key})
    return result.deleted_count


async def vector_search(
    query_key: str,
    query_embedding: list[float],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    if not models.MONGODB_ENABLED:
        return []

    shortlist_n = max(limit, 24)
    num_candidates = max(shortlist_n * 3, 24)
    pipeline = [
        {
            "$vectorSearch": {
                "index": models.MONGODB_VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": num_candidates,
                "limit": shortlist_n,
                "filter": {"query_key": {"$eq": query_key}},
            }
        },
        {
            "$project": {
                "text": 1,
                "source_tool": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    try:
        cursor = get_db()[CHUNK_VECTORS].aggregate(pipeline)
        docs = await cursor.to_list(length=shortlist_n)
    except Exception as exc:
        logger.warning(
            "Atlas $vectorSearch failed (%s); falling back to in-process cosine",
            exc,
        )
        return await _cosine_fallback(query_key, query_embedding, limit=shortlist_n)

    hits: list[RetrievedChunk] = []
    for doc in docs:
        text = doc.get("text")
        if not text:
            continue
        score = doc.get("score")
        hits.append(
            RetrievedChunk(
                text=str(text),
                source_tool=str(doc.get("source_tool") or ""),
                distance=None,
                relevance_score=float(score) if score is not None else None,
            )
        )
    return hits


async def _cosine_fallback(
    query_key: str,
    query_embedding: list[float],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    cursor = get_db()[CHUNK_VECTORS].find(
        {"query_key": query_key, **not_expired_filter()},
        projection={"text": 1, "source_tool": 1, "embedding": 1},
    )
    docs = await cursor.to_list(length=10_000)

    scored: list[tuple[float, dict]] = []
    for doc in docs:
        embedding = doc.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        score = _cosine_similarity(query_embedding, embedding)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    hits: list[RetrievedChunk] = []
    for score, doc in scored[:limit]:
        text = doc.get("text")
        if not text:
            continue
        hits.append(
            RetrievedChunk(
                text=str(text),
                source_tool=str(doc.get("source_tool") or ""),
                distance=1.0 - score,
                relevance_score=score,
            )
        )
    return hits


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)

"""Chunk embeddings cache and pgvector search."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from db.cache_utils import cache_expires_at, not_expired_sql
from db.client import SYN_CHUNK_VECTORS, get_pool
from models import models
from rag.chunking import TextChunk
from rag.store import RetrievedChunk

logger = logging.getLogger(__name__)


def chunk_id_for(chunk: TextChunk) -> str:
    return f"{chunk.source_tool}-{chunk.index}"


async def has_vectors(query_key: str) -> bool:
    if not models.DB_ENABLED:
        return False
    row = await get_pool().fetchrow(
        f"""
        SELECT 1 FROM {SYN_CHUNK_VECTORS}
        WHERE query_key = $1 AND {not_expired_sql()}
        LIMIT 1
        """,
        query_key,
    )
    return row is not None


async def upsert_chunks(
    query_key: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> int:
    if not models.DB_ENABLED or not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    now = datetime.now(UTC)
    expires_at = cache_expires_at()
    pool = get_pool()
    count = 0
    for chunk, embedding in zip(chunks, embeddings):
        await pool.execute(
            f"""
            INSERT INTO {SYN_CHUNK_VECTORS} (
                query_key, chunk_id, text, source_tool, query, "index",
                embedding, created_at, expires_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (query_key, chunk_id) DO UPDATE SET
                text = EXCLUDED.text,
                source_tool = EXCLUDED.source_tool,
                query = EXCLUDED.query,
                "index" = EXCLUDED."index",
                embedding = EXCLUDED.embedding,
                created_at = EXCLUDED.created_at,
                expires_at = EXCLUDED.expires_at
            """,
            query_key,
            chunk_id_for(chunk),
            chunk.text,
            chunk.source_tool,
            chunk.query,
            chunk.index,
            embedding,
            now,
            expires_at,
        )
        count += 1
    return count


async def delete_by_query_key(query_key: str) -> int:
    """Remove all cached vectors for a query (e.g. on force_refresh)."""
    if not models.DB_ENABLED:
        return 0
    result = await get_pool().execute(
        f"DELETE FROM {SYN_CHUNK_VECTORS} WHERE query_key = $1",
        query_key,
    )
    # asyncpg returns "DELETE N"
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def vector_search(
    query_key: str,
    query_embedding: list[float],
    *,
    limit: int,
) -> list[RetrievedChunk]:
    if not models.DB_ENABLED:
        return []

    shortlist_n = max(limit, 24)
    try:
        rows = await get_pool().fetch(
            f"""
            SELECT text, source_tool,
                   1 - (embedding <=> $1::vector) AS score
            FROM {SYN_CHUNK_VECTORS}
            WHERE query_key = $2 AND {not_expired_sql()}
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            query_embedding,
            query_key,
            shortlist_n,
        )
    except Exception as exc:
        logger.warning("pgvector search failed (%s)", exc)
        return []

    hits: list[RetrievedChunk] = []
    for row in rows:
        text = row["text"]
        if not text:
            continue
        score = row["score"]
        hits.append(
            RetrievedChunk(
                text=str(text),
                source_tool=str(row["source_tool"] or ""),
                distance=None,
                relevance_score=float(score) if score is not None else None,
            )
        )
    return hits

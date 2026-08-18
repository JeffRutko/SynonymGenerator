"""Smoke-test pgvector search on syn_chunk_vectors."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import client as db_client
from models import models


def _probe_embedding(dimensions: int) -> list[float]:
    vec = [0.0] * dimensions
    vec[0] = 1.0
    return vec


async def _run_probe() -> None:
    if not models.DB_ENABLED:
        print(
            "Error: DATABASE_URL is not set.\n"
            "Add your Neon connection string to .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    dimensions = models.DB_VECTOR_DIMENSIONS
    query_key = f"_probe_{uuid.uuid4().hex}"
    embedding = _probe_embedding(dimensions)
    probe_id = uuid.uuid4()

    await db_client.connect()
    try:
        pool = db_client.get_pool()
        await pool.fetchval("SELECT 1")
        print("OK: ping")

        await pool.execute(
            """
            INSERT INTO syn_chunk_vectors (
                id, query_key, chunk_id, text, source_tool, query, "index", embedding
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            probe_id,
            query_key,
            "probe-0",
            "probe chunk text",
            "_probe_tool",
            "probe query",
            0,
            embedding,
        )
        print("OK: inserted probe vector")

        row = await pool.fetchrow(
            """
            SELECT text, 1 - (embedding <=> $1::vector) AS score
            FROM syn_chunk_vectors
            WHERE query_key = $2
            ORDER BY embedding <=> $1::vector
            LIMIT 1
            """,
            embedding,
            query_key,
        )
        if not row:
            raise RuntimeError("Vector search returned no rows")

        print(f"OK: pgvector search (score={row['score']})")
        print("Vector search looks good.")
    finally:
        pool = db_client.get_pool()
        await pool.execute("DELETE FROM syn_chunk_vectors WHERE id = $1", probe_id)
        await db_client.disconnect()


def main() -> None:
    try:
        asyncio.run(_run_probe())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

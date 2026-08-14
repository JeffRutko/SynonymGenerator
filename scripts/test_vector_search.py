"""Smoke-test Atlas Vector Search on chunk_vectors ($vectorSearch probe)."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

load_dotenv()

DEFAULT_DB_NAME = "synonym_generator"
CHUNK_VECTORS = "chunk_vectors"
SERVER_SELECTION_TIMEOUT_MS = 10_000
# Atlas Vector Search indexes new writes asynchronously.
INDEX_PROPAGATION_DELAYS_S = (0, 2, 4, 8, 15)


def _get_uri() -> str:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        print(
            "Error: MONGODB_URI is not set.\n"
            "Add your Atlas connection string to .env.",
            file=sys.stderr,
        )
        sys.exit(1)
    return uri


def _probe_embedding(dimensions: int) -> list[float]:
    """Unit vector along first axis — sufficient for index/dimension probe."""
    vec = [0.0] * dimensions
    vec[0] = 1.0
    return vec


async def _vector_search(
    collection,
    *,
    index_name: str,
    query_vector: list[float],
    query_key: str,
    limit: int = 1,
) -> list[dict]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": max(limit * 10, 100),
                "limit": limit,
                "filter": {"query_key": {"$eq": query_key}},
            }
        },
        {
            "$project": {
                "text": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    cursor = collection.aggregate(pipeline)
    return await cursor.to_list(length=limit)


async def _verify_existing_indexed_doc(
    collection,
    *,
    index_name: str,
) -> bool:
    """Quick check against already-indexed data (no write propagation wait)."""
    sample = await collection.find_one(
        {"embedding": {"$exists": True}},
        projection={"query_key": 1, "embedding": 1},
    )
    if not sample:
        return False

    query_key = sample.get("query_key")
    embedding = sample.get("embedding")
    if not query_key or not isinstance(embedding, list) or not embedding:
        return False

    hits = await _vector_search(
        collection,
        index_name=index_name,
        query_vector=embedding,
        query_key=str(query_key),
        limit=1,
    )
    if not hits:
        return False

    score = hits[0].get("score")
    print(f"OK: $vectorSearch on existing doc (score={score})")
    return True


async def _run_probe() -> None:
    uri = _get_uri()
    db_name = os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME
    index_name = os.environ.get(
        "MONGODB_VECTOR_INDEX_NAME", "chunk_vectors_vector_index"
    ).strip()
    dimensions = int(os.environ.get("MONGODB_VECTOR_DIMENSIONS", "1024"))

    query_key = f"_probe_{uuid.uuid4().hex}"
    embedding = _probe_embedding(dimensions)

    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    insert_result = None
    try:
        await client.admin.command("ping")
        print("OK: ping")

        collection = client[db_name][CHUNK_VECTORS]

        if await _verify_existing_indexed_doc(collection, index_name=index_name):
            print("OK: index readable on existing chunk_vectors data")
        else:
            print(
                "Note: no existing indexed vectors to verify; "
                "proceeding with insert probe only."
            )

        chunk_id = "probe-0"
        insert_result = await collection.insert_one(
            {
                "query_key": query_key,
                "chunk_id": chunk_id,
                "text": "vector search probe document",
                "source_tool": "_probe",
                "query": "probe",
                "index": 0,
                "embedding": embedding,
            }
        )
        print(f"OK: inserted probe doc ({dimensions} dimensions)")

        hits: list[dict] = []
        try:
            for delay_s in INDEX_PROPAGATION_DELAYS_S:
                if delay_s:
                    print(f"Waiting {delay_s}s for Atlas index propagation…")
                    await asyncio.sleep(delay_s)
                hits = await _vector_search(
                    collection,
                    index_name=index_name,
                    query_vector=embedding,
                    query_key=query_key,
                    limit=1,
                )
                if hits:
                    break
        except Exception as exc:
            print(
                f"Error: $vectorSearch failed (index={index_name!r}).\n"
                "Ensure the Atlas Vector Search index exists on chunk_vectors.embedding "
                f"with numDimensions={dimensions} and a filter on query_key.",
                file=sys.stderr,
            )
            print(f"Detail: {exc}", file=sys.stderr)
            sys.exit(1)

        if not hits:
            print(
                "Error: $vectorSearch returned no results for the probe document "
                f"after {sum(INDEX_PROPAGATION_DELAYS_S)}s.\n"
                "The index may still be building, or the index definition may not "
                "match (path=embedding, filter=query_key, numDimensions="
                f"{dimensions}).",
                file=sys.stderr,
            )
            sys.exit(1)

        score = hits[0].get("score")
        print(f"OK: $vectorSearch hit on probe doc (score={score})")

        delete_result = await collection.delete_one({"_id": insert_result.inserted_id})
        insert_result = None
        if delete_result.deleted_count != 1:
            raise RuntimeError("Probe document was not deleted")

        print("OK: probe document deleted")
        print("Atlas Vector Search looks good.")
    finally:
        if insert_result is not None:
            try:
                db = client[db_name][CHUNK_VECTORS]
                await db.delete_one({"_id": insert_result.inserted_id})
            except Exception:
                pass
        client.close()


def main() -> None:
    try:
        asyncio.run(_run_probe())
    except ServerSelectionTimeoutError as exc:
        print(
            "Error: could not reach MongoDB (timeout).\n"
            "Check Atlas Network Access and the connection string.",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        sys.exit(1)
    except OperationFailure as exc:
        print(
            "Error: authentication or authorization failed.",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        sys.exit(1)
    except ConfigurationError as exc:
        print(
            "Error: invalid MongoDB configuration.\n"
            "For mongodb+srv:// URIs, install pymongo with SRV support: pymongo[srv]",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

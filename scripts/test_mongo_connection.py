"""Smoke-test MongoDB Atlas connectivity (ping + write/read/delete probe)."""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

load_dotenv()

DEFAULT_DB_NAME = "synonym_generator"
PROBE_COLLECTION = "_connection_test"
SERVER_SELECTION_TIMEOUT_MS = 10_000


def _get_uri() -> str:
    uri = os.environ.get("MONGODB_URI", "").strip()
    if not uri:
        print(
            "Error: MONGODB_URI is not set.\n"
            "Add your Atlas connection string to .env, for example:\n"
            "  MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority",
            file=sys.stderr,
        )
        sys.exit(1)
    return uri


async def _run_probe() -> None:
    uri = _get_uri()
    db_name = os.environ.get("MONGODB_DB_NAME", DEFAULT_DB_NAME).strip() or DEFAULT_DB_NAME

    client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=SERVER_SELECTION_TIMEOUT_MS)
    try:
        await client.admin.command("ping")
        print("OK: ping")

        db = client[db_name]
        print(f"OK: database {db_name}")

        collection = db[PROBE_COLLECTION]
        probe_doc = {"_probe": True, "source": "test_mongo_connection.py"}
        insert_result = await collection.insert_one(probe_doc)
        found = await collection.find_one({"_id": insert_result.inserted_id})
        if not found or found.get("_probe") is not True:
            raise RuntimeError("Probe document read back incorrectly")

        delete_result = await collection.delete_one({"_id": insert_result.inserted_id})
        if delete_result.deleted_count != 1:
            raise RuntimeError("Probe document was not deleted")

        print(f"OK: write/read/delete probe on {PROBE_COLLECTION}")
        print("MongoDB connection looks good.")
    finally:
        client.close()


def main() -> None:
    try:
        asyncio.run(_run_probe())
    except ServerSelectionTimeoutError as exc:
        print(
            "Error: could not reach MongoDB (timeout).\n"
            "Check Atlas Network Access (allow your IP or 0.0.0.0/0) and the connection string.",
            file=sys.stderr,
        )
        print(f"Detail: {exc}", file=sys.stderr)
        sys.exit(1)
    except OperationFailure as exc:
        print(
            "Error: authentication or authorization failed.\n"
            "Check the username/password in MONGODB_URI (URL-encode special characters).",
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

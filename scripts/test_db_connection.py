"""Smoke-test Neon PostgreSQL connectivity (ping + write/read/delete probe)."""

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


def _database_config_help() -> str:
    return (
        "Set Neon credentials in .env using either:\n"
        "  1. PGPASSWORD=... (with PGUSER/PGHOST already set), or\n"
        "  2. DATABASE_URL=postgresql://user:pass@host/neondb?sslmode=require\n"
        "Get the password from Neon: Project -> Connect -> Connection string."
    )


async def _run_probe() -> None:
    if not models.DB_ENABLED:
        print(
            f"Error: database is not configured.\n{_database_config_help()}",
            file=sys.stderr,
        )
        sys.exit(1)

    await db_client.connect()
    try:
        pool = db_client.get_pool()
        await pool.fetchval("SELECT 1")
        print("OK: ping")

        probe_id = uuid.uuid4()
        await pool.execute(
            """
            INSERT INTO syn_source_documents (
                id, query_key, tool_name, concept, context, raw_text
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            probe_id,
            "_probe_query_key",
            "_probe_tool",
            "probe",
            "",
            "probe text",
        )
        row = await pool.fetchrow(
            "SELECT raw_text FROM syn_source_documents WHERE id = $1",
            probe_id,
        )
        if not row or row["raw_text"] != "probe text":
            raise RuntimeError("Probe row read back incorrectly")

        result = await pool.execute(
            "DELETE FROM syn_source_documents WHERE id = $1",
            probe_id,
        )
        if not result.endswith("1"):
            raise RuntimeError("Probe row was not deleted")

        print("OK: write/read/delete probe on syn_source_documents")
        print("PostgreSQL connection looks good.")
    finally:
        await db_client.disconnect()


def main() -> None:
    try:
        asyncio.run(_run_probe())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Apply syn_* schema migrations to Neon."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import client as db_client
from models import models


async def _run() -> None:
    if not models.DB_ENABLED:
        print("Error: DATABASE_URL is not set.", file=sys.stderr)
        sys.exit(1)
    await db_client.connect()
    print("OK: migrations applied")
    await db_client.disconnect()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

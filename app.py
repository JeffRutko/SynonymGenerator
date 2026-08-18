"""Conceptual Search Helper — FastAPI + SSE UI."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from db import client as db_client
from models import models

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if models.DB_ENABLED:
        try:
            await db_client.connect()
        except Exception as exc:
            logger.error("PostgreSQL startup connect failed: %s", exc)
    yield
    if models.DB_ENABLED:
        await db_client.disconnect()


app = FastAPI(title="Conceptual Search Helper", lifespan=lifespan)


class SearchRequest(BaseModel):
    concept: str = Field(..., min_length=1)
    context: str = ""
    force_refresh: bool = False


@app.get("/health")
async def health() -> dict[str, str]:
    if not models.DB_ENABLED:
        db_status = "disabled"
    elif await db_client.ping():
        db_status = "connected"
    else:
        db_status = "error"
    return {"status": "ok", "db": db_status}


@app.post("/v1/search")
async def search(body: SearchRequest) -> StreamingResponse:
    # Import on demand so /health can pass without loading strands/chromadb.
    from synonym_agent import generate_synonyms_stream

    async def event_stream():
        async for progress, answer in generate_synonyms_stream(
            body.concept.strip(),
            (body.context or "").strip(),
            force_refresh=body.force_refresh,
        ):
            payload = json.dumps(
                {"progress": progress, "answer": answer},
                ensure_ascii=False,
            )
            yield f"data: {payload}\n\n"
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
def index(request: Request) -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    base_url = str(request.base_url).rstrip("/")
    return HTMLResponse(html.replace("__BASE_URL__", base_url))


@app.get("/favicon.ico")
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    open_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{open_host}:{port}", flush=True)
    uvicorn.run("app:app", host=host, port=port, reload=True)

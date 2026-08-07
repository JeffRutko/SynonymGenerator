"""Conceptual Search Helper — FastAPI + SSE UI."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from synonym_agent import generate_synonyms_stream

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Conceptual Search Helper")


class SearchRequest(BaseModel):
    concept: str = Field(..., min_length=1)
    context: str = ""


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/search")
async def search(body: SearchRequest) -> StreamingResponse:
    async def event_stream():
        async for progress, answer in generate_synonyms_stream(
            body.concept.strip(),
            (body.context or "").strip(),
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
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    # Default to loopback so browsers get a valid URL (0.0.0.0 is not navigable).
    # Set HOST=0.0.0.0 for container/Railway deploys.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "7860"))
    open_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Open http://{open_host}:{port}", flush=True)
    uvicorn.run("app:app", host=host, port=port, reload=True)

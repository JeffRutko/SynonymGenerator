"""Cohere MCP embeddings for RAG."""

from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from strands.tools.mcp import MCPClient

MAX_EMBED_BATCH = 96
EMBED_TOOL = "embed_texts"


def embed_texts(
    mcp_client: MCPClient,
    texts: Sequence[str],
    *,
    input_type: str = "search_document",
) -> list[list[float]]:
    """Embed texts via Cohere MCP ``embed_texts``. Raises on API failure."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), MAX_EMBED_BATCH):
        batch = [t.strip() for t in texts[start : start + MAX_EMBED_BATCH]]
        if any(not t for t in batch):
            raise ValueError("embed_texts requires non-empty strings")
        result = mcp_client.call_tool_sync(
            tool_use_id=str(uuid.uuid4()),
            name=EMBED_TOOL,
            arguments={"texts": batch, "input_type": input_type},
        )
        payload = _tool_result_to_dict(result)
        if "error" in payload:
            raise RuntimeError(f"embed_texts failed: {payload['error']}")
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise ValueError(
                f"Unexpected embeddings payload: got {type(embeddings)!r} "
                f"len={len(embeddings) if isinstance(embeddings, list) else 'n/a'}, "
                f"expected {len(batch)}"
            )
        for row in embeddings:
            if not isinstance(row, (list, tuple)) or not row:
                raise ValueError(f"Unexpected embedding row type: {type(row)!r}")
            vectors.append([float(x) for x in row])
    return vectors


def _tool_result_to_dict(result: Any) -> dict[str, Any]:
    """Parse MCP tool result into a dict (structured content or JSON text)."""
    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if structured is None and isinstance(result, dict):
        structured = result.get("structuredContent") or result.get(
            "structured_content"
        )
    if isinstance(structured, dict):
        return structured

    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                texts.append(str(block["text"]))
            elif hasattr(block, "text") and getattr(block, "text"):
                texts.append(str(block.text))
    elif isinstance(content, str):
        texts.append(content)

    for text in texts:
        text = text.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    if isinstance(result, dict) and (
        "embeddings" in result or "error" in result or "results" in result
    ):
        return result

    raise ValueError(f"Could not parse MCP tool result as dict: {result!r}")

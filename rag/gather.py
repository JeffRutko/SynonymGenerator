"""Gather MCP search hits and turn them into text chunks."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from strands.tools.mcp import MCPClient

from rag.chunking import TextChunk, chunk_text

GATHER_TOOLS = ("telecom_search", "patent_search", "web_text_search")
ProgressCallback = Callable[[str], None]


def iter_gather_mcp_documents(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
) -> Iterator[str | list[TextChunk]]:
    """
    Yield progress strings while gathering, then the chunk list.

    Final yield is always ``list[TextChunk]``.
    """
    query = _build_search_query(concept, context)
    chunks: list[TextChunk] = []
    for tool_name in GATHER_TOOLS:
        yield f"Calling search tool: `{tool_name}`…"
        before = len(chunks)
        try:
            result = mcp_client.call_tool_sync(
                tool_use_id=str(uuid.uuid4()),
                name=tool_name,
                arguments={"query": query},
            )
        except Exception as exc:
            yield f"`{tool_name}` failed ({exc}); skipping."
            continue
        text = _tool_result_to_text(result)
        if not text:
            yield f"`{tool_name}` returned no text."
            continue
        chunks.extend(
            chunk_text(text, source_tool=tool_name, query=query)
        )
        yield (
            f"`{tool_name}` → {len(chunks) - before} chunks "
            f"({len(chunks)} total)."
        )
    yield chunks


def gather_mcp_documents(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
    *,
    on_progress: ProgressCallback | None = None,
) -> list[TextChunk]:
    """Call search tools and return chunked passages ready to index."""
    chunks: list[TextChunk] = []
    for item in iter_gather_mcp_documents(mcp_client, concept, context):
        if isinstance(item, list):
            chunks = item
        elif on_progress:
            on_progress(item)
    return chunks


def _build_search_query(concept: str, context: str) -> str:
    concept = concept.strip()
    context = context.strip()
    if context:
        return f"{concept} {context}"
    return concept


def _tool_result_to_text(result: Any) -> str:
    parts: list[str] = []
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")

    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
            elif hasattr(block, "text") and getattr(block, "text"):
                parts.append(str(block.text))
            else:
                parts.append(_stringify(block))
    elif content:
        parts.append(_stringify(content))

    structured = getattr(result, "structuredContent", None) or getattr(
        result, "structured_content", None
    )
    if structured:
        parts.append(_stringify(structured))

    return "\n".join(p for p in parts if p and p.strip())


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)

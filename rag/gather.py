"""Gather MCP search hits and turn them into text chunks."""

from __future__ import annotations

import json
import uuid
from typing import Any

from strands.tools.mcp import MCPClient

from rag.chunking import TextChunk, chunk_text

GATHER_TOOLS = ("telecom_search", "patent_search", "web_text_search")


def gather_mcp_documents(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
) -> list[TextChunk]:
    """Call search tools and return chunked passages ready to index."""
    query = _build_search_query(concept, context)
    chunks: list[TextChunk] = []
    for tool_name in GATHER_TOOLS:
        try:
            result = mcp_client.call_tool_sync(
                tool_use_id=str(uuid.uuid4()),
                name=tool_name,
                arguments={"query": query},
            )
        except Exception:
            continue
        text = _tool_result_to_text(result)
        if not text:
            continue
        chunks.extend(
            chunk_text(text, source_tool=tool_name, query=query)
        )
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

"""Gather MCP search hits and turn them into text chunks."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from strands.tools.mcp import MCPClient

from db.repositories import source_documents as source_documents_repo
from models import models
from rag.chunking import TextChunk, chunk_text
from rag.constants import GATHER_TOOLS
ProgressCallback = Callable[[str], None]


async def iter_gather_mcp_documents(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
    *,
    query_key: str | None = None,
    force_refresh: bool = False,
) -> AsyncIterator[str | list[TextChunk]]:
    """
    Yield progress strings while gathering, then the chunk list.

    Final yield is always ``list[TextChunk]``.
    """
    query = _build_search_query(concept, context)
    use_cache = (
        models.MONGODB_ENABLED
        and query_key is not None
        and not force_refresh
    )
    chunks: list[TextChunk] = []

    for tool_name in GATHER_TOOLS:
        before = len(chunks)
        raw_text: str | None = None

        if use_cache:
            raw_text = await source_documents_repo.get_cached_raw_text(
                query_key, tool_name
            )
            if raw_text:
                yield f"Using cached `{tool_name}`…"

        if raw_text is None:
            yield f"Calling search tool: `{tool_name}`…"
            try:
                result = mcp_client.call_tool_sync(
                    tool_use_id=str(uuid.uuid4()),
                    name=tool_name,
                    arguments={"query": query},
                )
            except Exception as exc:
                yield f"`{tool_name}` failed ({exc}); skipping."
                continue
            raw_text = _tool_result_to_text(result)
            if not raw_text:
                yield f"`{tool_name}` returned no text."
                continue
            if models.MONGODB_ENABLED and query_key is not None:
                await source_documents_repo.upsert(
                    query_key=query_key,
                    concept=concept,
                    context=context,
                    tool_name=tool_name,
                    raw_text=raw_text,
                )

        chunks.extend(
            chunk_text(raw_text, source_tool=tool_name, query=query)
        )
        added = len(chunks) - before
        yield (
            f"`{tool_name}` → {added} chunks "
            f"({len(chunks)} total)."
        )

    yield chunks


def iter_gather_mcp_documents_sync(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
    *,
    query_key: str | None = None,
    force_refresh: bool = False,
) -> Iterator[str | list[TextChunk]]:
    """Sync wrapper for the async gather iterator."""
    import asyncio

    async def _collect() -> list[str | list[TextChunk]]:
        items: list[str | list[TextChunk]] = []
        async for item in iter_gather_mcp_documents(
            mcp_client,
            concept,
            context,
            query_key=query_key,
            force_refresh=force_refresh,
        ):
            items.append(item)
        return items

    for item in asyncio.run(_collect()):
        yield item


async def gather_mcp_documents(
    mcp_client: MCPClient,
    concept: str,
    context: str = "",
    *,
    query_key: str | None = None,
    force_refresh: bool = False,
    on_progress: ProgressCallback | None = None,
) -> list[TextChunk]:
    """Call search tools and return chunked passages ready to index."""
    chunks: list[TextChunk] = []
    async for item in iter_gather_mcp_documents(
        mcp_client,
        concept,
        context,
        query_key=query_key,
        force_refresh=force_refresh,
    ):
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

"""Gather MCP search hits and turn them into text chunks."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

from strands.tools.mcp import MCPClient

from db.repositories import source_documents as source_documents_repo
from models import models
from rag.chunking import TextChunk, chunk_text
from rag.constants import GATHER_TOOLS

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]

# Parallel gather often trips transient Tavily/MCP disconnects; retry those.
GATHER_MAX_ATTEMPTS = 3
GATHER_RETRY_BASE_DELAY_S = 1.0
_TRANSIENT_ERROR_MARKERS = (
    "connection aborted",
    "remotedisconnected",
    "unexpected_eof",
    "ssleoferror",
    "sslerror",
    "max retries exceeded",
    "connection reset",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "503",
    "502",
    "504",
)


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
        models.DB_ENABLED
        and query_key is not None
        and not force_refresh
    )

    raw_by_tool: dict[str, str | None] = {name: None for name in GATHER_TOOLS}

    if use_cache:
        cached = await asyncio.gather(
            *(
                source_documents_repo.get_cached_raw_text(query_key, tool_name)
                for tool_name in GATHER_TOOLS
            )
        )
        for tool_name, raw_text in zip(GATHER_TOOLS, cached, strict=True):
            if not raw_text:
                continue
            if _looks_like_tool_error_text(raw_text):
                logger.warning(
                    "Ignoring cached error payload for %s (query_key=%s)",
                    tool_name,
                    query_key,
                )
                yield f"Cached `{tool_name}` is an error payload; refetching…"
                continue
            raw_by_tool[tool_name] = raw_text
            yield f"Using cached `{tool_name}`…"

    miss_tools = [name for name in GATHER_TOOLS if raw_by_tool[name] is None]
    if miss_tools:
        tool_list = ", ".join(f"`{name}`" for name in miss_tools)
        yield f"Calling search tools in parallel: {tool_list}…"

        # First attempt in parallel; retry transient failures together.
        pending = list(miss_tools)
        to_upsert: list[tuple[str, str]] = []

        for attempt in range(1, GATHER_MAX_ATTEMPTS + 1):
            if attempt > 1:
                delay = GATHER_RETRY_BASE_DELAY_S * (2 ** (attempt - 2))
                names = ", ".join(f"`{n}`" for n in pending)
                yield (
                    f"Transient failure on {names}; "
                    f"retry {attempt}/{GATHER_MAX_ATTEMPTS} in {delay:.0f}s…"
                )
                await asyncio.sleep(delay)

            results = await asyncio.gather(
                *(
                    _call_search_tool(mcp_client, tool_name, query)
                    for tool_name in pending
                ),
                return_exceptions=True,
            )

            still_pending: list[str] = []
            for tool_name, result in zip(pending, results, strict=True):
                ok_text, err_msg, transient = _classify_tool_outcome(result)
                if ok_text is not None:
                    raw_by_tool[tool_name] = ok_text
                    to_upsert.append((tool_name, ok_text))
                    if attempt > 1:
                        yield f"`{tool_name}` succeeded on retry {attempt}."
                    continue

                if transient and attempt < GATHER_MAX_ATTEMPTS:
                    still_pending.append(tool_name)
                else:
                    suffix = (
                        f" after {attempt} attempts"
                        if attempt > 1
                        else ""
                    )
                    logger.warning(
                        "%s failed%s: %s",
                        tool_name,
                        suffix,
                        (err_msg or "")[:300],
                    )
                    yield f"`{tool_name}` failed{suffix} ({err_msg}); skipping."

            pending = still_pending
            if not pending:
                break

        if to_upsert and models.DB_ENABLED and query_key is not None:
            await asyncio.gather(
                *(
                    source_documents_repo.upsert(
                        query_key=query_key,
                        concept=concept,
                        context=context,
                        tool_name=tool_name,
                        raw_text=raw_text,
                    )
                    for tool_name, raw_text in to_upsert
                )
            )

    chunks: list[TextChunk] = []
    for tool_name in GATHER_TOOLS:
        raw_text = raw_by_tool[tool_name]
        if not raw_text:
            continue
        before = len(chunks)
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


async def _call_search_tool(
    mcp_client: MCPClient,
    tool_name: str,
    query: str,
) -> Any:
    return await mcp_client.call_tool_async(
        tool_use_id=str(uuid.uuid4()),
        name=tool_name,
        arguments={"query": query},
    )


def _classify_tool_outcome(
    result: Any,
) -> tuple[str | None, str, bool]:
    """
    Return (ok_text, error_message, is_transient).

    ``ok_text`` is set only for usable search content.
    """
    if isinstance(result, BaseException):
        msg = str(result) or result.__class__.__name__
        return None, msg, _is_transient_error_text(msg)

    if _mcp_result_is_error(result):
        detail = _tool_result_to_text(result) or "unknown MCP error"
        return None, detail, _is_transient_error_text(detail)

    raw_text = _tool_result_to_text(result)
    if not raw_text:
        return None, "returned no text", False

    if _looks_like_tool_error_text(raw_text):
        return None, raw_text, _is_transient_error_text(raw_text)

    return raw_text, "", False


def _is_transient_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _TRANSIENT_ERROR_MARKERS)


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


def _mcp_result_is_error(result: Any) -> bool:
    """True when the MCP CallToolResult is marked as an error."""
    if isinstance(result, dict):
        return bool(result.get("isError") or result.get("is_error"))
    return bool(
        getattr(result, "isError", False) or getattr(result, "is_error", False)
    )


def _looks_like_tool_error_text(text: str) -> bool:
    """
    Detect search-tool failure payloads returned as content.

    Tavily/MCP often returns JSON like ``[{"error": "Connection aborted..."}]``
    or ``{"error": "..."}`` instead of raising; those must not be cached.
    """
    stripped = text.strip()
    if not stripped:
        return False

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Doubled quotes from some serializers: [{""error"":""...""}]
        try:
            parsed = json.loads(stripped.replace('""', '"'))
        except json.JSONDecodeError:
            return False

    return _payload_is_error_only(parsed)


def _payload_is_error_only(value: Any) -> bool:
    if isinstance(value, dict):
        if "error" not in value:
            return False
        # Pure failure object, or error with no usable search body.
        useful = ("results", "documents", "content", "answer", "text", "hits")
        return not any(value.get(k) for k in useful)

    if isinstance(value, list):
        if not value:
            return False
        return all(
            isinstance(item, dict) and "error" in item and _payload_is_error_only(item)
            for item in value
        )

    return False


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)

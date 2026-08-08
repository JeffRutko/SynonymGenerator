"""Cohere MCP reranking for RAG shortlists."""

from __future__ import annotations

import uuid
from typing import Sequence

from strands.tools.mcp import MCPClient

from rag.embeddings import _tool_result_to_dict

RERANK_TOOL = "rerank_documents"


def rerank_documents(
    mcp_client: MCPClient,
    query: str,
    documents: Sequence[str],
    *,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """
    Rerank document strings via Cohere MCP ``rerank_documents``.

    Returns ordered (original_index, relevance_score) pairs.
    """
    query = query.strip()
    docs = [d.strip() for d in documents]
    if not query or not docs:
        return []

    arguments: dict = {"query": query, "documents": docs}
    if top_n is not None:
        arguments["top_n"] = top_n

    result = mcp_client.call_tool_sync(
        tool_use_id=str(uuid.uuid4()),
        name=RERANK_TOOL,
        arguments=arguments,
    )
    payload = _tool_result_to_dict(result)
    if "error" in payload:
        raise RuntimeError(f"rerank_documents failed: {payload['error']}")

    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected rerank results payload: {type(rows)!r}")

    ranked: list[tuple[int, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        score = row.get("relevance_score")
        if not isinstance(idx, int):
            continue
        ranked.append((idx, float(score) if score is not None else 0.0))
    return ranked

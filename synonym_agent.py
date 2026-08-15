import asyncio
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

from mcp.client.streamable_http import streamable_http_client
from strands import Agent
from strands.tools.mcp import MCPClient

from db import client as db_client
from db.query_key import make_query_key
from db.repositories import chunk_vectors as chunk_vectors_repo
from db.repositories import synonym_outputs as synonym_outputs_repo
from models import models
from rag import (
    EphemeralRagStore,
    build_rag_prompt,
    format_retrieved_passages,
)
from rag.chunking import TextChunk
from rag.gather import iter_gather_mcp_documents
from rag.mongo_store import MongoVectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a conceptual search helper for telecom and patent prior-art research.

Produce rich, structured markdown reports: component-wise synonym families, multiple Boolean search-string variants that include CPC subgroup symbols (e.g. H04W36/00, not only H04W), a Relevant CPC subgroups list, and brief notes on why terms work.

When retrieved passages are provided, treat them as supporting evidence and seeds for expansion — not a closed vocabulary. Prefer 3GPP, IEEE, and patent-style phrasing. Use tools (including CPC/IPC scheme lookups via web_text_search when needed) to identify precise subgroups.

Honor any CPC / domain hints the user supplies, and drill down to subgroup level in search strings."""

EMBED_TOOL = "embed_texts"
SEARCH_TOOL = "telecom_search"
AGENT_EXCLUDED_TOOLS = frozenset({"embed_texts", "rerank_documents"})


@dataclass(frozen=True)
class RagResult:
    note: str
    passages: str


async def _ensure_mongo() -> None:
    if models.MONGODB_ENABLED:
        await db_client.connect()


def create_mcp_clients() -> list[MCPClient]:
    """One MCPClient per URL in ``models.MCP_SERVER_URLS``."""
    clients: list[MCPClient] = []
    for url in models.MCP_SERVER_URLS:
        clients.append(
            MCPClient(
                lambda url=url: streamable_http_client(url),
                continue_on_error=True,
            )
        )
    return clients


@contextmanager
def mcp_session() -> Iterator[tuple[MCPClient, MCPClient]]:
    """
    Enter all MCP clients; yield (search_client, cohere_client).

    Raises if either role cannot be resolved after connect.
    """
    clients = create_mcp_clients()
    entered: list[MCPClient] = []
    try:
        for client in clients:
            client.__enter__()
            entered.append(client)

        search = _find_client_with_tool(entered, SEARCH_TOOL)
        cohere = _find_client_with_tool(entered, EMBED_TOOL)
        if search is None:
            raise RuntimeError(
                f"No MCP server exposed `{SEARCH_TOOL}` "
                f"(configured URLs: {models.MCP_SERVER_URLS})"
            )
        if cohere is None:
            raise RuntimeError(
                f"No MCP server exposed `{EMBED_TOOL}` "
                f"(configured URLs: {models.MCP_SERVER_URLS})"
            )
        yield search, cohere
    finally:
        for client in reversed(entered):
            try:
                client.__exit__(None, None, None)
            except Exception:
                pass


def _find_client_with_tool(
    clients: list[MCPClient], tool_name: str
) -> MCPClient | None:
    for client in clients:
        try:
            tools = client.list_tools_sync()
        except Exception:
            continue
        for tool in tools:
            name = getattr(tool, "tool_name", None) or getattr(tool, "name", None)
            if name == tool_name:
                return client
            spec = getattr(tool, "tool_spec", None) or getattr(tool, "mcp_tool", None)
            if isinstance(spec, dict) and spec.get("name") == tool_name:
                return client
            nested = getattr(spec, "name", None) if spec is not None else None
            if nested == tool_name:
                return client
    return None


def create_agent_with_tools(
    search_client: MCPClient,
    callback_handler: Callable[..., None] | None = None,
) -> Agent:
    tools = [
        tool
        for tool in search_client.list_tools_sync()
        if _tool_name(tool) not in AGENT_EXCLUDED_TOOLS
    ]
    return Agent(
        model=models.create_hf_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        callback_handler=callback_handler,
    )


def _tool_name(tool: object) -> str:
    name = getattr(tool, "tool_name", None) or getattr(tool, "name", None)
    if isinstance(name, str) and name:
        return name
    spec = getattr(tool, "tool_spec", None) or getattr(tool, "mcp_tool", None)
    if isinstance(spec, dict) and spec.get("name"):
        return str(spec["name"])
    nested = getattr(spec, "name", None) if spec is not None else None
    return str(nested) if nested else ""


def build_prompt(concept: str, context: str = "", passages: str = "") -> str:
    return build_rag_prompt(concept, context, passages)


async def iter_rag_pipeline(
    search_client: MCPClient,
    cohere_client: MCPClient,
    concept: str,
    context: str,
    *,
    query_key: str | None = None,
    force_refresh: bool = False,
) -> AsyncIterator[str | RagResult]:
    """
    Run gather → embed/index → retrieve → rerank.

    Yields progress strings in pipeline order, then a final ``RagResult``.
    """
    if query_key is None:
        query_key = make_query_key(concept, context)

    yield "Gathering search results…"
    chunks: list[TextChunk] = []
    try:
        async for item in iter_gather_mcp_documents(
            search_client,
            concept,
            context,
            query_key=query_key,
            force_refresh=force_refresh,
        ):
            if isinstance(item, list):
                chunks = item
            else:
                yield item
    except Exception as exc:
        yield RagResult(
            f"Gather failed ({exc}); continuing with tools only.",
            "",
        )
        return

    if not chunks:
        yield RagResult("No passages indexed; continuing with tools only.", "")
        return

    query = f"{concept.strip()} {context.strip()}".strip()
    vectors_cached = False
    if models.MONGODB_ENABLED and force_refresh:
        deleted = await chunk_vectors_repo.delete_by_query_key(query_key)
        if deleted:
            yield f"Cleared {deleted} cached vector(s) (force_refresh)."
    elif models.MONGODB_ENABLED:
        vectors_cached = await chunk_vectors_repo.has_vectors(query_key)

    try:
        if models.MONGODB_ENABLED:
            store: MongoVectorStore | EphemeralRagStore = MongoVectorStore(
                cohere_client,
                query_key,
                vectors_cached=vectors_cached,
            )
            if vectors_cached:
                yield f"Using {len(chunks)} cached chunks from MongoDB."
            else:
                yield (
                    f"Embedding {len(chunks)} chunks via `embed_texts` "
                    f"(input_type=search_document)…"
                )
            indexed = await store.add_chunks(chunks)
            yield f"Indexed {indexed} chunks in MongoDB."

            yield (
                "Embedding query via `embed_texts` "
                "(input_type=search_query)…"
            )
            candidates = await store.retrieve_shortlist(query, k=8)
        else:
            chroma = EphemeralRagStore(cohere_client)
            yield (
                f"Embedding {len(chunks)} chunks via `embed_texts` "
                f"(input_type=search_document)…"
            )
            chroma.add_chunks(chunks)
            store = chroma
            yield f"Indexed {len(chunks)} chunks in ephemeral store."
            yield (
                "Embedding query via `embed_texts` "
                "(input_type=search_query)…"
            )
            candidates = chroma.retrieve_shortlist(query, k=8)

        yield f"Retrieved {len(candidates)} vector shortlist candidates."

        if not candidates:
            yield RagResult(
                f"Indexed {len(chunks)} chunks but retrieval was empty; "
                "continuing with tools only.",
                "",
            )
            return

        yield (
            f"Reranking {len(candidates)} candidates via "
            f"`rerank_documents`…"
        )
        hits = store.rerank(query, candidates, k=8)
        yield f"Reranked; kept {len(hits)} passages."
    except Exception as exc:
        yield RagResult(
            f"Embedding/index failed ({exc}); continuing with tools only.",
            "",
        )
        return

    passages = format_retrieved_passages(hits)
    if not passages:
        yield RagResult(
            f"Indexed {len(chunks)} chunks but retrieval was empty; "
            "continuing with tools only.",
            "",
        )
        return

    yield RagResult(
        f"RAG ready: {len(chunks)} chunks → {len(hits)} passages.",
        passages,
    )


async def _index_and_retrieve(
    search_client: MCPClient,
    cohere_client: MCPClient,
    concept: str,
    context: str,
    *,
    query_key: str,
    force_refresh: bool,
) -> tuple[str, str]:
    """Consume the RAG pipeline; return (status_note, passages)."""
    result: RagResult | None = None
    async for item in iter_rag_pipeline(
        search_client,
        cohere_client,
        concept,
        context,
        query_key=query_key,
        force_refresh=force_refresh,
    ):
        if isinstance(item, RagResult):
            result = item
    if result is None:
        return "RAG pipeline produced no result; continuing with tools only.", ""
    return result.note, result.passages


def generate_synonyms(
    concept: str,
    context: str = "",
    *,
    force_refresh: bool = False,
) -> str:
    if not concept.strip():
        return "Please enter a concept or phrase."

    return asyncio.run(
        _generate_synonyms_async(concept, context, force_refresh=force_refresh)
    )


async def _generate_synonyms_async(
    concept: str,
    context: str,
    *,
    force_refresh: bool,
) -> str:
    await _ensure_mongo()
    query_key = make_query_key(concept, context)

    with mcp_session() as (search_client, cohere_client):
        _, passages = await _index_and_retrieve(
            search_client,
            cohere_client,
            concept,
            context,
            query_key=query_key,
            force_refresh=force_refresh,
        )
        prompt = build_prompt(concept, context, passages)
        agent = create_agent_with_tools(search_client, callback_handler=None)
        result = agent(prompt)
        answer = str(result)

    if models.MONGODB_ENABLED:
        try:
            await synonym_outputs_repo.upsert(
                query_key=query_key,
                concept=concept,
                context=context,
                answer=answer,
                progress="Done.",
                tools_used=[],
            )
        except Exception as exc:
            logger.warning("Failed to cache synonym output: %s", exc)

    return answer


async def generate_synonyms_stream(
    concept: str,
    context: str = "",
    *,
    force_refresh: bool = False,
) -> AsyncIterator[tuple[str, str]]:
    """Yield (progress_markdown, answer_markdown) pairs as the agent runs."""
    if not concept.strip():
        yield ("Please enter a concept or phrase.", "")
        return

    if not os.environ.get("HF_TOKEN"):
        yield (
            (
                "**Error:** `HF_TOKEN` is not set. "
                "Add it to your local `.env` file (or host environment)."
            ),
            "",
        )
        return

    await _ensure_mongo()
    query_key = make_query_key(concept, context)

    if models.MONGODB_ENABLED and not force_refresh:
        try:
            cached = await synonym_outputs_repo.get_cached(query_key)
        except Exception as exc:
            logger.warning("Output cache lookup failed: %s", exc)
            cached = None
        if cached is not None:
            progress = cached.progress or "Loaded cached report from MongoDB."
            if not progress.startswith("- "):
                progress = f"- {progress}\n- Done."
            yield (progress, cached.answer)
            return

    status_lines: list[str] = ["Connecting to MCP servers…"]
    answer = ""
    seen_tools: set[str] = set()

    def progress() -> str:
        return "\n".join(f"- {line}" for line in status_lines)

    yield progress(), answer

    try:
        with mcp_session() as (search_client, cohere_client):
            passages = ""
            async for item in iter_rag_pipeline(
                search_client,
                cohere_client,
                concept,
                context,
                query_key=query_key,
                force_refresh=force_refresh,
            ):
                if isinstance(item, RagResult):
                    status_lines.append(item.note)
                    passages = item.passages
                else:
                    status_lines.append(item)
                yield progress(), answer

            prompt = build_prompt(concept, context, passages)
            status_lines.append("Agent ready — researching…")
            yield progress(), answer

            agent = create_agent_with_tools(search_client, callback_handler=None)

            async for event in agent.stream_async(prompt):
                tool = event.get("current_tool_use") or {}
                tool_name = tool.get("name")
                if tool_name and tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    status_lines.append(f"Agent calling tool: `{tool_name}`")
                    yield progress(), answer

                if "data" in event and event["data"]:
                    answer += event["data"]
                    yield progress(), answer

            status_lines.append("Done.")
            yield progress(), answer

            if models.MONGODB_ENABLED:
                try:
                    await synonym_outputs_repo.upsert(
                        query_key=query_key,
                        concept=concept,
                        context=context,
                        answer=answer,
                        progress=progress(),
                        tools_used=sorted(seen_tools),
                    )
                except Exception as exc:
                    logger.warning("Failed to cache synonym output: %s", exc)
    except Exception as exc:
        status_lines.append(f"Failed: {exc}")
        yield progress(), answer

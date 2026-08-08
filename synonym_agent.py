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

from models import models
from rag import (
    EphemeralRagStore,
    build_rag_prompt,
    format_retrieved_passages,
)
from rag.chunking import TextChunk
from rag.gather import iter_gather_mcp_documents

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
            # AgentTool / MCPAgentTool may nest the MCP name
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


def iter_rag_pipeline(
    search_client: MCPClient,
    cohere_client: MCPClient,
    concept: str,
    context: str,
) -> Iterator[str | RagResult]:
    """
    Run gather → embed/index → retrieve → rerank.

    Yields progress strings in pipeline order, then a final ``RagResult``.
    """
    yield "Gathering search results…"
    chunks: list[TextChunk] = []
    try:
        for item in iter_gather_mcp_documents(search_client, concept, context):
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

    try:
        store = EphemeralRagStore(cohere_client)
        yield (
            f"Embedding {len(chunks)} chunks via `embed_texts` "
            f"(input_type=search_document)…"
        )
        store.add_chunks(chunks)
        yield f"Indexed {len(chunks)} chunks in ephemeral store."

        query = f"{concept.strip()} {context.strip()}".strip()
        yield (
            "Embedding query via `embed_texts` "
            "(input_type=search_query)…"
        )
        candidates = store.retrieve_shortlist(query, k=8)
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


def _index_and_retrieve(
    search_client: MCPClient,
    cohere_client: MCPClient,
    concept: str,
    context: str,
) -> tuple[str, str]:
    """Consume the RAG pipeline; return (status_note, passages)."""
    result: RagResult | None = None
    for item in iter_rag_pipeline(search_client, cohere_client, concept, context):
        if isinstance(item, RagResult):
            result = item
    if result is None:
        return "RAG pipeline produced no result; continuing with tools only.", ""
    return result.note, result.passages


def generate_synonyms(concept: str, context: str = "") -> str:
    if not concept.strip():
        return "Please enter a concept or phrase."

    with mcp_session() as (search_client, cohere_client):
        _, passages = _index_and_retrieve(
            search_client, cohere_client, concept, context
        )
        prompt = build_prompt(concept, context, passages)
        agent = create_agent_with_tools(search_client, callback_handler=None)
        result = agent(prompt)
        return str(result)


async def generate_synonyms_stream(
    concept: str, context: str = ""
) -> AsyncIterator[tuple[str, str]]:
    """Yield (progress_markdown, answer_markdown) pairs as the agent runs."""
    if not concept.strip():
        yield ("Please enter a concept or phrase.", "")
        return

    if not os.environ.get("HF_TOKEN"):
        yield (
            (
                "**Error:** `HF_TOKEN` is not set. "
                "Add it as a Space secret (or in a local `.env` file)."
            ),
            "",
        )
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
            for item in iter_rag_pipeline(
                search_client, cohere_client, concept, context
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
    except Exception as exc:
        status_lines.append(f"Failed: {exc}")
        yield progress(), answer

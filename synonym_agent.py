import os
from collections.abc import AsyncIterator
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
    gather_mcp_documents,
)

SYSTEM_PROMPT = """You are a conceptual search helper for telecom and patent prior-art research.

Produce rich, structured markdown reports: component-wise synonym families, multiple Boolean search-string variants that include CPC subgroup symbols (e.g. H04W36/00, not only H04W), a Relevant CPC subgroups list, and brief notes on why terms work.

When retrieved passages are provided, treat them as supporting evidence and seeds for expansion — not a closed vocabulary. Prefer 3GPP, IEEE, and patent-style phrasing. Use tools (including CPC/IPC scheme lookups via web_text_search when needed) to identify precise subgroups.

Honor any CPC / domain hints the user supplies, and drill down to subgroup level in search strings."""


def create_mcp_client() -> MCPClient:
    return MCPClient(lambda: streamable_http_client(models.MCP_SERVER_URL))


def create_agent_with_tools(
    mcp_client: MCPClient,
    callback_handler: Callable[..., None] | None = None,
) -> Agent:
    tools = mcp_client.list_tools_sync()
    return Agent(
        model=models.create_hf_model(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        callback_handler=callback_handler,
    )


def build_prompt(concept: str, context: str = "", passages: str = "") -> str:
    return build_rag_prompt(concept, context, passages)


def _index_and_retrieve(
    mcp_client: MCPClient,
    concept: str,
    context: str,
) -> tuple[str, str]:
    """
    Gather MCP hits into ephemeral Chroma and return (status_note, passages).

    On failure or empty gather, returns a fallback note and empty passages.
    """
    try:
        chunks = gather_mcp_documents(mcp_client, concept, context)
    except Exception as exc:
        return f"Gather failed ({exc}); continuing with tools only.", ""

    if not chunks:
        return "No passages indexed; continuing with tools only.", ""

    try:
        store = EphemeralRagStore()
        store.add_chunks(chunks)
        query = f"{concept.strip()} {context.strip()}".strip()
        hits = store.query(query, k=8)
    except Exception as exc:
        return f"Embedding/index failed ({exc}); continuing with tools only.", ""

    passages = format_retrieved_passages(hits)
    if not passages:
        return (
            f"Indexed {len(chunks)} chunks but retrieval was empty; continuing with tools only.",
            "",
        )
    return f"Indexed {len(chunks)} chunks; retrieved {len(hits)} passages.", passages


def generate_synonyms(concept: str, context: str = "") -> str:
    if not concept.strip():
        return "Please enter a concept or phrase."

    mcp_client = create_mcp_client()

    with mcp_client:
        _, passages = _index_and_retrieve(mcp_client, concept, context)
        prompt = build_prompt(concept, context, passages)
        agent = create_agent_with_tools(mcp_client, callback_handler=None)
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

    mcp_client = create_mcp_client()
    status_lines: list[str] = ["Connecting to search tools…"]
    answer = ""
    seen_tools: set[str] = set()

    def progress() -> str:
        return "\n".join(f"- {line}" for line in status_lines)

    yield progress(), answer

    try:
        with mcp_client:
            status_lines.append("Gathering search results…")
            yield progress(), answer

            note, passages = _index_and_retrieve(mcp_client, concept, context)
            status_lines.append(note)
            if passages:
                status_lines.append("Retrieving similar passages… done.")
            yield progress(), answer

            prompt = build_prompt(concept, context, passages)
            status_lines.append("Agent ready — researching…")
            yield progress(), answer

            agent = create_agent_with_tools(mcp_client, callback_handler=None)

            async for event in agent.stream_async(prompt):
                tool = event.get("current_tool_use") or {}
                tool_name = tool.get("name")
                if tool_name and tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    status_lines.append(f"Using tool: `{tool_name}`")
                    yield progress(), answer

                if "data" in event and event["data"]:
                    answer += event["data"]
                    yield progress(), answer

            status_lines.append("Done.")
            yield progress(), answer
    except Exception as exc:
        status_lines.append(f"Failed: {exc}")
        yield progress(), answer

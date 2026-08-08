"""Live RAG helpers for the synonym agent."""

from rag.chunking import TextChunk, chunk_text
from rag.embeddings import embed_texts
from rag.gather import gather_mcp_documents, iter_gather_mcp_documents
from rag.rerank import rerank_documents
from rag.store import EphemeralRagStore, RetrievedChunk

__all__ = [
    "EphemeralRagStore",
    "RetrievedChunk",
    "TextChunk",
    "chunk_text",
    "embed_texts",
    "rerank_documents",
    "gather_mcp_documents",
    "iter_gather_mcp_documents",
    "format_retrieved_passages",
    "build_rag_prompt",
]


def format_retrieved_passages(hits: list[RetrievedChunk]) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    for i, hit in enumerate(hits, start=1):
        source = hit.source_tool or "unknown"
        lines.append(f"{i}. [{source}] {hit.text}")
    return "\n".join(lines)


def build_rag_prompt(concept: str, context: str, passages: str) -> str:
    concept = concept.strip()
    context = context.strip()

    parts = [
        f"Help with conceptual search for this concept: {concept}.",
    ]
    if context:
        parts.append(f"Domain / CPC hint: {context}")

    if passages:
        parts.append(
            "Retrieved passages (supporting evidence — cite and expand from these; "
            "they are seeds, not an exclusive vocabulary):"
        )
        parts.append(passages)
        parts.append(
            "Use the passages plus available tools. Call tools when passages leave gaps "
            "(missing aliases, standards terms, or CPC subgroup symbols)."
        )
    else:
        parts.append(
            "No retrieved passages were available. Use the available tools to research "
            "the concept before writing the report."
        )

    parts.append(
        """Write a rich markdown report with this structure:

1. **Concept gloss** — 1–3 sentences clarifying the technical meaning.
2. **Synonyms by component** — Decompose the concept into entities, actions, and technical objects. For each component, list synonym families (tables or bullets). Include telecom / 3GPP / patent phrasing and generation-specific aliases when relevant (e.g. eNB, gNB, UE, MS).
3. **Relevant CPC subgroups** — List specific CPC subgroup symbols (e.g. H04W36/00, H04W28/06), not only section/class codes like H04W. Include a one-line title/meaning for each when known. Use the user's CPC hint and tools (e.g. web_text_search for CPC scheme) when unsure.
4. **Recommended Boolean search strings** — Provide three variants. Each variant must incorporate CPC subgroup symbols via AND/OR (e.g. (...terms...) AND (H04W36/00 OR H04W36/08)):
   - Full Boolean (high precision)
   - Compact practical string
   - Patent-style string
5. **Key terminology notes** — Brief “why these terms work,” citing passage `source_tool` labels and/or standards/patent language.

Do not limit yourself only to words that appear verbatim in the passages; expand with closely related domain terminology that would help literature and patent search. Prefer subgroup-level CPC over class-only codes in the search strings."""
    )

    return "\n\n".join(parts)

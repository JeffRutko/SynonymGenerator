"""Character chunking with a fixed overlap ratio."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHUNK_SIZE = 500
OVERLAP_RATIO = 0.15


@dataclass(frozen=True)
class TextChunk:
    text: str
    source_tool: str
    query: str
    index: int


def chunk_text(
    text: str,
    *,
    source_tool: str,
    query: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_ratio: float = OVERLAP_RATIO,
) -> list[TextChunk]:
    """Split ``text`` into windows with ``overlap_ratio`` overlap (e.g. 0.15 = 15%)."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [
            TextChunk(
                text=cleaned,
                source_tool=source_tool,
                query=query,
                index=0,
            )
        ]

    overlap = max(1, int(chunk_size * overlap_ratio))
    step = max(1, chunk_size - overlap)
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(
                TextChunk(
                    text=piece,
                    source_tool=source_tool,
                    query=query,
                    index=index,
                )
            )
            index += 1
        if end >= len(cleaned):
            break
        start += step
    return chunks

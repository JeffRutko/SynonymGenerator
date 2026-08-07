"""Per-request ephemeral Chroma store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import chromadb

from rag.chunking import TextChunk
from rag.embeddings import embed_texts


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_tool: str
    distance: float | None


class EphemeralRagStore:
    """In-memory Chroma collection for a single request."""

    def __init__(self) -> None:
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"synonyms_{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)
        ids = [f"{c.source_tool}-{c.index}-{uuid.uuid4().hex[:8]}" for c in chunks]
        metadatas = [
            {"source_tool": c.source_tool, "query": c.query, "index": c.index}
            for c in chunks
        ]
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(self, text: str, k: int = 8) -> list[RetrievedChunk]:
        text = text.strip()
        if not text:
            return []
        count = self._collection.count()
        if count == 0:
            return []
        n = min(k, count)
        query_embedding = embed_texts([text])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if not doc:
                continue
            source = ""
            if isinstance(meta, dict):
                source = str(meta.get("source_tool") or "")
            hits.append(
                RetrievedChunk(
                    text=doc,
                    source_tool=source,
                    distance=float(dist) if dist is not None else None,
                )
            )
        return hits

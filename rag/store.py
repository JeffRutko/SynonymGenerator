"""Per-request ephemeral Chroma store."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.chunking import TextChunk
from rag.embeddings import embed_texts
from rag.rerank import rerank_documents

if TYPE_CHECKING:
    from strands.tools.mcp import MCPClient

DEFAULT_SHORTLIST = 24


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_tool: str
    distance: float | None
    relevance_score: float | None = None


class EphemeralRagStore:
    """In-memory Chroma collection for a single request."""

    def __init__(self, mcp_client: MCPClient) -> None:
        # Lazy: chromadb/onnxruntime are heavy; skip import when Mongo RAG is used.
        import chromadb

        self._mcp = mcp_client
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"synonyms_{uuid.uuid4().hex}",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = embed_texts(
            self._mcp, texts, input_type="search_document"
        )
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

    def retrieve_shortlist(self, text: str, k: int = 8) -> list[RetrievedChunk]:
        """Embed the query and return a cosine shortlist (no rerank)."""
        text = text.strip()
        if not text:
            return []
        count = self._collection.count()
        if count == 0:
            return []

        shortlist_n = min(max(k, DEFAULT_SHORTLIST), count)
        query_embedding = embed_texts(
            self._mcp, [text], input_type="search_query"
        )[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=shortlist_n,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        candidates: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if not doc:
                continue
            source = ""
            if isinstance(meta, dict):
                source = str(meta.get("source_tool") or "")
            candidates.append(
                RetrievedChunk(
                    text=doc,
                    source_tool=source,
                    distance=float(dist) if dist is not None else None,
                )
            )
        return candidates

    def rerank(
        self,
        text: str,
        candidates: list[RetrievedChunk],
        k: int = 8,
    ) -> list[RetrievedChunk]:
        """Rerank shortlist candidates via Cohere MCP."""
        text = text.strip()
        if not text or not candidates:
            return []
        top_n = min(k, len(candidates))
        ranked = rerank_documents(
            self._mcp,
            text,
            [c.text for c in candidates],
            top_n=top_n,
        )
        hits: list[RetrievedChunk] = []
        for idx, score in ranked:
            if idx < 0 or idx >= len(candidates):
                continue
            base = candidates[idx]
            hits.append(
                RetrievedChunk(
                    text=base.text,
                    source_tool=base.source_tool,
                    distance=base.distance,
                    relevance_score=score,
                )
            )
        return hits

    def query(self, text: str, k: int = 8) -> list[RetrievedChunk]:
        candidates = self.retrieve_shortlist(text, k=k)
        return self.rerank(text, candidates, k=k)

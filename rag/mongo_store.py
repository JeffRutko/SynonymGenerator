"""Atlas-backed vector store with MCP embed/rerank."""

from __future__ import annotations

import logging

from strands.tools.mcp import MCPClient

from db.repositories import chunk_vectors as chunk_vectors_repo
from models import models
from rag.chunking import TextChunk
from rag.embeddings import embed_texts
from rag.rerank import rerank_documents
from rag.store import DEFAULT_SHORTLIST, RetrievedChunk

logger = logging.getLogger(__name__)


class MongoVectorStore:
    """Persist chunks in Atlas and retrieve via Vector Search."""

    def __init__(
        self,
        mcp_client: MCPClient,
        query_key: str,
        *,
        vectors_cached: bool = False,
    ) -> None:
        self._mcp = mcp_client
        self._query_key = query_key
        self._vectors_cached = vectors_cached

    async def add_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        if self._vectors_cached:
            return len(chunks)

        texts = [c.text for c in chunks]
        embeddings = embed_texts(
            self._mcp, texts, input_type="search_document"
        )
        if (
            models.MONGODB_ENABLED
            and embeddings
            and len(embeddings[0]) != models.MONGODB_VECTOR_DIMENSIONS
        ):
            logger.info(
                "Embedding dimension is %s (MONGODB_VECTOR_DIMENSIONS=%s)",
                len(embeddings[0]),
                models.MONGODB_VECTOR_DIMENSIONS,
            )

        return await chunk_vectors_repo.upsert_chunks(
            self._query_key, chunks, embeddings
        )

    async def retrieve_shortlist(self, text: str, k: int = 8) -> list[RetrievedChunk]:
        text = text.strip()
        if not text:
            return []

        query_embedding = embed_texts(
            self._mcp, [text], input_type="search_query"
        )[0]
        shortlist_n = max(k, DEFAULT_SHORTLIST)
        return await chunk_vectors_repo.vector_search(
            self._query_key,
            query_embedding,
            limit=shortlist_n,
        )

    def rerank(
        self,
        text: str,
        candidates: list[RetrievedChunk],
        k: int = 8,
    ) -> list[RetrievedChunk]:
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

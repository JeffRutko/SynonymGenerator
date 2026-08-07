"""Hugging Face Inference embeddings for RAG."""

from __future__ import annotations

import os
from typing import Sequence

from huggingface_hub import InferenceClient

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed texts via HF Inference feature-extraction. Raises on API failure."""
    if not texts:
        return []

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for embeddings")

    client = InferenceClient(provider="hf-inference", api_key=token)
    vectors: list[list[float]] = []
    for text in texts:
        raw = client.feature_extraction(text, model=EMBEDDING_MODEL_ID)
        vectors.append(_to_1d(raw))
    return vectors


def _to_1d(raw: object) -> list[float]:
    """Normalize nested feature-extraction output to a single vector."""
    if hasattr(raw, "tolist"):
        raw = raw.tolist()  # type: ignore[assignment]

    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"Unexpected embedding payload type: {type(raw)!r}")

    first = raw[0]
    if isinstance(first, (int, float)):
        return [float(x) for x in raw]

    # Token-level matrix → mean pool
    if isinstance(first, (list, tuple)):
        dim = len(first)
        acc = [0.0] * dim
        for row in raw:
            for i, v in enumerate(row):
                acc[i] += float(v)
        n = float(len(raw))
        return [v / n for v in acc]

    raise ValueError(f"Unexpected embedding row type: {type(first)!r}")

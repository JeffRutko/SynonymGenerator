"""Cache key helpers for concept + context lookups."""

from __future__ import annotations

import hashlib


def make_query_key(concept: str, context: str = "") -> str:
    normalized = f"{concept.strip().lower()}|{context.strip().lower()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

"""Shared cache expiry helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models import models


def cache_expires_at() -> datetime | None:
    hours = models.DB_CACHE_TTL_HOURS
    if hours <= 0:
        return None
    return datetime.now(UTC) + timedelta(hours=hours)


def not_expired_sql(*, prefix: str = "") -> str:
    """SQL fragment: row is within the soft cache window."""
    col = f"{prefix}expires_at" if prefix else "expires_at"
    return f"({col} IS NULL OR {col} > now())"

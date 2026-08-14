"""Shared cache expiry helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from models import models


def cache_expires_at() -> datetime | None:
    hours = models.MONGODB_CACHE_TTL_HOURS
    if hours <= 0:
        return None
    return datetime.now(UTC) + timedelta(hours=hours)


def not_expired_filter() -> dict:
    now = datetime.now(UTC)
    return {
        "$or": [
            {"expires_at": None},
            {"expires_at": {"$exists": False}},
            {"expires_at": {"$gt": now}},
        ]
    }

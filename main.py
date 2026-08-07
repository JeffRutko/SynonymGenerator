"""ASGI entrypoint for hosts that default to ``main:app`` (e.g. Railway)."""

from app import app

__all__ = ["app"]

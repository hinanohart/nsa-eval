"""Attention backends. Each backend conforms to `AttentionBackend` and is registered by name."""

from __future__ import annotations

from .protocol import AttentionBackend

__all__ = ["AttentionBackend"]

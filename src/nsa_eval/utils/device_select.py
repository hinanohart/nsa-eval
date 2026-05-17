"""Device selection with fallback chain."""

from __future__ import annotations

from collections.abc import Iterable


def select_device(preferred: str, fallback: Iterable[str] = ("cpu",)) -> str:
    """Return the first preferred device that is actually available; else first fallback."""
    try:
        import torch
    except ImportError:
        return next(iter(fallback))
    for d in (preferred, *fallback):
        if d == "cuda" and torch.cuda.is_available():
            return d
        if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return d
        if d == "cpu":
            return d
    return "cpu"

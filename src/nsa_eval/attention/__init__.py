"""Attention backends. Each backend conforms to `AttentionBackend` and is registered by name."""

from __future__ import annotations

from ..eval.runner import register_attention
from .baselines import H2OBackend, SnapKVBackend
from .mps_backend import NSAMPSBackend
from .protocol import AttentionBackend

register_attention("nsa_mps", NSAMPSBackend)
register_attention("h2o", H2OBackend)
register_attention("snapkv", SnapKVBackend)

__all__ = ["AttentionBackend", "NSAMPSBackend", "H2OBackend", "SnapKVBackend"]

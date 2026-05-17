"""Model loaders. v0.1 ships Qwen2.5 (0.5B / 1.5B) and Llama-3.2 (1B / 3B)."""

from __future__ import annotations

from .hf_adapter import load_hf

__all__ = ["load_hf"]

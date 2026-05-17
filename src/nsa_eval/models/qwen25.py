"""Qwen2.5 small-model loaders via the HF adapter."""

from __future__ import annotations

from .hf_adapter import load_hf


def load_qwen25_0_5b(**kwargs):
    return load_hf("Qwen/Qwen2.5-0.5B", **kwargs)


def load_qwen25_1_5b(**kwargs):
    return load_hf("Qwen/Qwen2.5-1.5B", **kwargs)

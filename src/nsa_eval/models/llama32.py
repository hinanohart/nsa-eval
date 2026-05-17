"""Llama-3.2 small-model loaders via the HF adapter."""

from __future__ import annotations

from .hf_adapter import load_hf


def load_llama32_1b(**kwargs):
    return load_hf("meta-llama/Llama-3.2-1B", **kwargs)


def load_llama32_3b(**kwargs):
    return load_hf("meta-llama/Llama-3.2-3B", **kwargs)

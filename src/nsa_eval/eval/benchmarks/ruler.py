"""RULER long-context benchmark adapter.

Reference: https://arxiv.org/abs/2404.06654 — synthetic long-context tasks at controlled
sequence lengths. v0.1 targets {4k, 8k, 16k, 32k} with the public RULER subset.
"""

from __future__ import annotations


class RULERBenchmark:
    name = "ruler"
    default_seqlens: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)

    def __init__(self, seqlens: tuple[int, ...] | None = None) -> None:
        self.seqlens = seqlens or self.default_seqlens

    def cases(self):
        raise NotImplementedError("RULER dataset loader lands in week 3")

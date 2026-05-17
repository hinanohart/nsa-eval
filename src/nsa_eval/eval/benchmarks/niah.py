"""Needle-in-a-Haystack adapter — synthetic recall benchmark, smallest and fastest of v0.1.

Reference: https://github.com/gkamradt/LLMTest_NeedleInAHaystack. Used as the smoke test
benchmark; if NIAH fails on a backend the matrix run for that backend is aborted.
"""

from __future__ import annotations

import random


class NIAHBenchmark:
    name = "niah"
    default_seqlens: tuple[int, ...] = (2048, 4096, 8192, 16384)
    default_depth_steps: int = 10

    def __init__(
        self,
        seqlens: tuple[int, ...] | None = None,
        depth_steps: int | None = None,
        seed: int = 0,
    ) -> None:
        self.seqlens = seqlens or self.default_seqlens
        self.depth_steps = depth_steps or self.default_depth_steps
        self.rng = random.Random(seed)

    def cases(self):
        raise NotImplementedError("NIAH generator lands in week 2 (gate for v0.1 smoke test)")

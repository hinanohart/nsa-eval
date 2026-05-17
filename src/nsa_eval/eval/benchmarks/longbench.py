"""LongBench v2 adapter — single-doc QA, multi-doc QA, summarisation, few-shot, synthetic.

Reference: https://arxiv.org/abs/2412.15204. v0.1 targets the en-subset of single-doc QA and
synthetic; full coverage lands in Phase 2.
"""

from __future__ import annotations


class LongBenchBenchmark:
    name = "longbench"
    default_subsets: tuple[str, ...] = ("narrativeqa", "qasper", "passage_retrieval_en")

    def __init__(self, subsets: tuple[str, ...] | None = None) -> None:
        self.subsets = subsets or self.default_subsets

    def cases(self):
        raise NotImplementedError("LongBench dataset loader lands in week 3")

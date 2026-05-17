"""AgentBench adapter — multi-turn agent tasks, included to probe path-dependent attention.

Reference: https://arxiv.org/abs/2308.03688. v0.1 targets the OS / DB / KG subsets only;
they have the longest context profiles, which is the relevant stress case for NSA's selection
branch.
"""

from __future__ import annotations


class AgentBenchBenchmark:
    name = "agentbench"
    default_subsets: tuple[str, ...] = ("os", "db", "kg")

    def __init__(self, subsets: tuple[str, ...] | None = None) -> None:
        self.subsets = subsets or self.default_subsets

    def cases(self):
        raise NotImplementedError("AgentBench harness lands in week 6")

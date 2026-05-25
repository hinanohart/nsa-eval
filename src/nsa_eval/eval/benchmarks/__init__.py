"""Benchmark adapters. Each adapter implements `cases()` returning iterable of prompts/labels."""

from __future__ import annotations

from ..runner import register_benchmark
from .agentbench import AgentBenchBenchmark
from .longbench import LongBenchBenchmark
from .niah import NIAHBenchmark
from .ruler import RULERBenchmark

register_benchmark("niah", NIAHBenchmark)
register_benchmark("ruler", RULERBenchmark)
register_benchmark("longbench", LongBenchBenchmark)
register_benchmark("agentbench", AgentBenchBenchmark)

# Aliases retained for backwards compatibility
AgentBench = AgentBenchBenchmark
LongBench = LongBenchBenchmark
NIAH = NIAHBenchmark
RULER = RULERBenchmark

__all__ = ["AgentBench", "LongBench", "NIAH", "RULER"]

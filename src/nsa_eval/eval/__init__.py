"""Evaluation harness: spec, runner, benchmarks."""

from __future__ import annotations

from .runner import BenchmarkRunner, register_attention, register_benchmark
from .spec import EvalSpec

__all__ = ["BenchmarkRunner", "EvalSpec", "register_attention", "register_benchmark"]

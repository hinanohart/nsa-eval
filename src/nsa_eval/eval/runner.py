"""Benchmark runner — turns an EvalSpec into a JSON result on disk.

The runner is intentionally registry-based: attention backends and benchmarks are looked up by
`name`, not by isinstance, so third-party plugins can register their own implementations and
be picked up via config without touching this file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from ..utils.manifest import build_manifest
from ..utils.seed import set_global_seed
from .spec import EvalSpec

_ATTENTION_REGISTRY: dict[str, Callable[[], object]] = {}
_BENCHMARK_REGISTRY: dict[str, Callable[[], object]] = {}


def register_attention(name: str, factory: Callable[[], object]) -> None:
    _ATTENTION_REGISTRY[name] = factory


def register_benchmark(name: str, factory: Callable[[], object]) -> None:
    _BENCHMARK_REGISTRY[name] = factory


def registered_attentions() -> list[str]:
    return sorted(_ATTENTION_REGISTRY)


def registered_benchmarks() -> list[str]:
    return sorted(_BENCHMARK_REGISTRY)


class BenchmarkRunner:
    def __init__(self, results_root: Path | str = "benchmarks/results") -> None:
        # Resolve relative paths against the caller's cwd at construction time so a later
        # `chdir` in user code doesn't move our output tree.
        self.results_root = Path(results_root).resolve()

    def run(self, spec: EvalSpec) -> Path:
        set_global_seed(spec.seed)
        attention_factory = _ATTENTION_REGISTRY.get(spec.attention)
        benchmark_factory = _BENCHMARK_REGISTRY.get(spec.benchmark)
        if attention_factory is None:
            raise KeyError(
                f"unknown attention backend: {spec.attention}; "
                f"registered: {registered_attentions()}"
            )
        if benchmark_factory is None:
            raise KeyError(
                f"unknown benchmark: {spec.benchmark}; registered: {registered_benchmarks()}"
            )
        attention = attention_factory()
        benchmark = benchmark_factory()
        result = {
            "spec": spec.model_dump(),
            "status": "scaffold_only",
            "attention_name": getattr(attention, "name", spec.attention),
            "benchmark_name": getattr(benchmark, "name", spec.benchmark),
        }
        # Scaffold-only results go to a sibling tree so they cannot pollute the real result
        # series that downstream tools (leaderboard, paper figures) will glob.
        bucket = (
            "_scaffold"
            if result["status"] == "scaffold_only"
            else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        out_dir = self.results_root / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{spec.slug()}.json"
        out.write_text(
            json.dumps({**result, "manifest": build_manifest()}, indent=2, sort_keys=True)
        )
        return out

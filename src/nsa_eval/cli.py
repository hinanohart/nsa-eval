"""nsa-eval CLI — `nsa-eval run|list-backends|version`."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

import click
import yaml

from .config.schema import RunConfig
from .eval.runner import BenchmarkRunner
from .eval.spec import EvalSpec


def _version() -> str:
    try:
        return pkg_version("nsa-eval")
    except PackageNotFoundError:
        return "0.0.1"


@click.group()
@click.version_option(_version(), prog_name="nsa-eval")
def main() -> None:
    """nsa-eval — MPS-native NSA + unified long-context evaluation suite."""


@main.command()
@click.option(
    "--config", "config_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--results",
    "results_root",
    default="benchmarks/results",
    type=click.Path(path_type=Path),
)
def run(config_path: Path, results_root: Path) -> None:
    """Run one eval cell described by a YAML config."""
    cfg = RunConfig.model_validate(yaml.safe_load(config_path.read_text()))
    spec = EvalSpec(
        model=cfg.model.name,
        benchmark=cfg.benchmark.name,
        attention=cfg.attention.name,
        device=cfg.device.name,
        seed=cfg.seed,
    )
    out = BenchmarkRunner(results_root).run(spec)
    click.echo(f"wrote {out}")


@main.command("list-backends")
def list_backends() -> None:
    """List registered attention backends and benchmarks."""
    from .eval.runner import _ATTENTION_REGISTRY, _BENCHMARK_REGISTRY

    click.echo("attention:")
    for name in sorted(_ATTENTION_REGISTRY):
        click.echo(f"  - {name}")
    click.echo("benchmark:")
    for name in sorted(_BENCHMARK_REGISTRY):
        click.echo(f"  - {name}")

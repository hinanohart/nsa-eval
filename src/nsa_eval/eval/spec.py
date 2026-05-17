"""Evaluation spec — declarative description of one benchmark run.

One `EvalSpec` is one row of the published evaluation matrix. The runner materialises it into
a deterministic file path under `benchmarks/results/<date>/<slug>.json` so two runs of the
same spec are diff-able.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(value: str) -> str:
    """Replace anything that would create a subdir or break a glob with `-`."""
    return _SAFE_SLUG_RE.sub("-", value)


class EvalSpec(BaseModel):
    model: str
    benchmark: str
    attention: str
    device: str
    seqlen: int = Field(default=8192, ge=128)
    batch_size: int = Field(default=1, ge=1)
    seed: int = 0
    dtype: str = "bfloat16"
    extra: dict[str, Any] = Field(default_factory=dict)

    def slug(self) -> str:
        return (
            f"{_safe(self.model)}_{_safe(self.benchmark)}_{_safe(self.device)}"
            f"_{_safe(self.attention)}_seq{self.seqlen}_seed{self.seed}"
        )

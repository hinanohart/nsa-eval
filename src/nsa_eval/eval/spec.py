"""Evaluation spec — declarative description of one benchmark run.

One `EvalSpec` is one row of the published evaluation matrix. The runner materialises it into
a deterministic file path under `benchmarks/results/<date>/<slug>.json` so two runs of the
same spec are diff-able.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalSpec(BaseModel):
    model: str
    benchmark: str
    attention: str
    device: str
    seqlen: int = Field(default=8192, ge=128)
    batch_size: int = Field(default=1, ge=1)
    seed: int = 0
    dtype: str = "bfloat16"
    extra: dict = Field(default_factory=dict)

    def slug(self) -> str:
        return (
            f"{self.model}_{self.benchmark}_{self.device}_{self.attention}"
            f"_seq{self.seqlen}_seed{self.seed}"
        )

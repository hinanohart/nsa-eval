"""Pydantic schema for the YAML configs under `configs/`.

Configs are layered: a top-level `config.yaml` references named sub-configs in
`configs/{model,benchmark,attention,device}/`. The CLI loads one top-level YAML, resolves
references, and validates against `RunConfig`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    hf_id: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False


class BenchmarkConfig(BaseModel):
    name: str
    seqlens: list[int] = Field(default_factory=list)
    subset: str | None = None


class AttentionConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)


class DeviceConfig(BaseModel):
    name: Literal["cuda", "mps", "cpu"]
    fallback: list[str] = Field(default_factory=lambda: ["cpu"])


class RunConfig(BaseModel):
    model: ModelConfig
    benchmark: BenchmarkConfig
    attention: AttentionConfig
    device: DeviceConfig
    seed: int = 0

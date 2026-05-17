"""H2O and SnapKV reference baselines.

These are not re-implementations of the underlying algorithms; they are thin adapters that
wire the published references into the evaluation harness. Real implementations are pulled in
by the runner via `third_party/` when the user opts into them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class H2OConfig:
    heavy_budget: int = 512
    recent_budget: int = 512


class H2OBackend:
    """H2O — heavy-hitter + recent window KV retention. arXiv 2306.14048."""

    name = "h2o"
    supports_devices: tuple[str, ...] = ("cuda", "mps", "cpu")

    def __init__(self, config: H2OConfig | None = None) -> None:
        self.config = config or H2OConfig()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("H2O baseline integration lands in week 4")

    def supports(self, device: torch.device) -> bool:
        return device.type in self.supports_devices


@dataclass
class SnapKVConfig:
    window_size: int = 32
    max_capacity: int = 1024


class SnapKVBackend:
    """SnapKV — observation-window guided KV selection. arXiv 2404.14469."""

    name = "snapkv"
    supports_devices: tuple[str, ...] = ("cuda", "mps", "cpu")

    def __init__(self, config: SnapKVConfig | None = None) -> None:
        self.config = config or SnapKVConfig()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("SnapKV baseline integration lands in week 4")

    def supports(self, device: torch.device) -> bool:
        return device.type in self.supports_devices

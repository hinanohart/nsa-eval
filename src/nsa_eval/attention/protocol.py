"""Attention backend protocol — the contract every NSA / baseline implementation honours.

Implementations own causal masking, dtype, and device placement. The runner does not pad,
transpose, or move tensors; backends must accept whatever the calling model hands them and
return a tensor with the same shape and dtype as `q`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AttentionBackend(Protocol):
    name: str
    supports_devices: tuple[str, ...]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def supports(self, device: torch.device) -> bool: ...

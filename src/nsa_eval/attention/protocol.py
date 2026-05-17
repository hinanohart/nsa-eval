"""Attention backend protocol — the contract every NSA / baseline implementation honours.

Implementations own causal masking, dtype, and device placement. The runner does not pad,
transpose, or move tensors; backends must accept whatever the calling model hands them and
return a tensor with the same shape and dtype as `q`.

`attention_mask` is reserved for future per-token masking (e.g. variable-length batches);
v0.1 backends either honour it or raise NotImplementedError. When both `is_causal=True` and
`attention_mask` are passed, the explicit mask wins.
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


def conforms(obj: object) -> bool:
    """Stronger check than `isinstance(_, AttentionBackend)`.

    `runtime_checkable` Protocol only verifies attribute presence (PEP-544); a class with the
    right attributes but wrong method signatures still passes `isinstance`. This helper at
    least asserts that `forward` and `supports` are callable on the candidate.
    """
    return (
        isinstance(obj, AttentionBackend)
        and callable(getattr(obj, "forward", None))
        and callable(getattr(obj, "supports", None))
    )

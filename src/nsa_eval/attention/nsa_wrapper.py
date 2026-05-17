"""CUDA NSA backend — thin wrapper around fla-org/native-sparse-attention.

The third-party package is loaded lazily so that `import nsa_eval` works on machines without
CUDA or Triton installed (e.g. the MPS-only path or the CI image).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NSAConfig:
    compression_block_size: int = 32
    compression_stride: int = 16
    selection_block_size: int = 64
    selection_top_n: int = 16
    sliding_window: int = 512
    gate_init: float = 0.0


class NSACudaBackend:
    """NSA forward delegating to fla-org's Triton kernel. CUDA-only."""

    name = "nsa_cuda"
    supports_devices: tuple[str, ...] = ("cuda",)

    def __init__(self, config: NSAConfig | None = None) -> None:
        self.config = config or NSAConfig()
        self._impl: object | None = None

    def _load(self) -> None:
        if self._impl is not None:
            return
        try:
            from native_sparse_attention import NativeSparseAttention
        except ImportError as e:
            raise RuntimeError(
                "fla-org/native-sparse-attention is not installed. "
                "Vendor it under third_party/ and `uv pip install -e .[nsa_cuda]`, "
                "or use a different backend (mps, h2o, snapkv, full)."
            ) from e
        self._impl = NativeSparseAttention(**self.config.__dict__)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q.device.type != "cuda":
            raise RuntimeError(f"NSACudaBackend requires CUDA, got {q.device}")
        self._load()
        assert self._impl is not None
        return self._impl(q, k, v, is_causal=is_causal, attention_mask=attention_mask)  # type: ignore[no-any-return,misc]

    def supports(self, device: torch.device) -> bool:
        return device.type == "cuda"

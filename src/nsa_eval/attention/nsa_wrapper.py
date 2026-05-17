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
        # fla-org exposes an HF-style module whose constructor takes a config object, not flat
        # kwargs. The wiring is intentionally deferred to issue #2 (P1-02) so the smoke test
        # path is honest about being unimplemented rather than crashing inside vendor code.
        raise NotImplementedError(
            "CUDA backend wiring lands in issue #2 (P1-02). The reference impl is vendored "
            "under third_party/native-sparse-attention; this wrapper still needs to bridge "
            "the fla-org config and hidden-dim arguments. Use a different backend until then."
        )

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
        self._load()  # raises NotImplementedError until issue #2 closes
        raise RuntimeError("unreachable — _load raises")

    def supports(self, device: torch.device) -> bool:
        return device.type == "cuda"

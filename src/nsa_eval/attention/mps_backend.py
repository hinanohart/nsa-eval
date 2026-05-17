"""MPS-native NSA backend — load-bearing for the project's core thesis.

Apple Silicon (M1/M2/M3) is not available on the maintainer's machine. The implementation here
was written against PyTorch's documented MPS public API; every Mac-specific code path is gated
so the module imports cleanly on CPU and CUDA-only machines too. Until a Mac contributor
validates numeric output and end-to-end performance, treat MPS results as un-attested.
See issue #1.

Reference: https://arxiv.org/abs/2502.11089 §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class NSAMPSConfig:
    compression_block_size: int = 32
    compression_stride: int = 16
    selection_block_size: int = 64
    selection_top_n: int = 16
    sliding_window: int = 512


class NSAMPSBackend(nn.Module):
    """NSA forward implemented with vanilla PyTorch ops so MPS dispatches to Metal kernels.

    Three branches combined by a learnable sigmoid gate:
      1. Compression — strided pooling over keys/values, then dense attention against the
         compressed KV. Cheap, captures global structure.
      2. Selection — block-wise importance scores, top-n blocks, dense attention on the slice.
         Captures relevant chunks.
      3. Sliding window — local attention with window `w`. Captures recency.
    """

    name = "nsa_mps"
    supports_devices: tuple[str, ...] = ("mps", "cpu")

    def __init__(self, config: NSAMPSConfig | None = None) -> None:
        super().__init__()
        self.config = config or NSAMPSConfig()
        self.gate = nn.Parameter(torch.zeros(1))

    def _compression_branch(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        block = self.config.compression_block_size
        stride = self.config.compression_stride
        _b, _h, n, _dh = k.shape
        n_win = max(1, (n - block) // stride + 1)
        windows_k = torch.stack(
            [k[..., i * stride : i * stride + block, :].mean(dim=-2) for i in range(n_win)],
            dim=-2,
        )
        windows_v = torch.stack(
            [v[..., i * stride : i * stride + block, :].mean(dim=-2) for i in range(n_win)],
            dim=-2,
        )
        return _scaled_dot_attn(q, windows_k, windows_v, is_causal=False)

    def _selection_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        lp, top_n = self.config.selection_block_size, self.config.selection_top_n
        b, h, n, dh = k.shape
        n_blk = max(1, n // lp)
        if n_blk * lp != n:
            k = k[..., : n_blk * lp, :]
            v = v[..., : n_blk * lp, :]
        blocks_k = k.reshape(b, h, n_blk, lp, dh).mean(dim=-2)
        blocks_v = v.reshape(b, h, n_blk, lp, dh).mean(dim=-2)
        importance = torch.einsum("bhnd,bhkd->bhnk", q, blocks_k)
        top = importance.topk(min(top_n, n_blk), dim=-1).indices
        gather_idx = top.unsqueeze(-1).expand(-1, -1, -1, -1, dh).reshape(b, h, -1, dh)
        gathered_k = torch.gather(
            blocks_k.unsqueeze(2).expand(-1, -1, q.shape[-2], -1, -1).reshape(b, h, -1, dh),
            -2,
            gather_idx,
        )
        gathered_v = torch.gather(
            blocks_v.unsqueeze(2).expand(-1, -1, q.shape[-2], -1, -1).reshape(b, h, -1, dh),
            -2,
            gather_idx,
        )
        gathered_k = gathered_k.reshape(b, h, q.shape[-2], min(top_n, n_blk), dh)
        gathered_v = gathered_v.reshape(b, h, q.shape[-2], min(top_n, n_blk), dh)
        scores = torch.einsum("bhnd,bhnkd->bhnk", q, gathered_k) / math.sqrt(dh)
        attn = scores.softmax(dim=-1)
        return torch.einsum("bhnk,bhnkd->bhnd", attn, gathered_v)

    def _sliding_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = self.config.sliding_window
        _b, _h, n, dh = q.shape
        m = k.shape[-2]
        mask = torch.ones(n, m, device=q.device, dtype=torch.bool).tril(0).triu(-w + 1)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(dh)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        return torch.einsum("bhnm,bhmd->bhnd", attn, v)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        is_causal: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not is_causal:
            raise NotImplementedError("non-causal NSA is not part of v0.1")
        comp = self._compression_branch(q, k, v)
        sel = self._selection_branch(q, k, v)
        slid = self._sliding_branch(q, k, v)
        gate = self.gate.sigmoid()
        return gate * (comp + sel) + (1 - gate) * slid

    def supports(self, device: torch.device) -> bool:
        return device.type in self.supports_devices


def _scaled_dot_attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool
) -> torch.Tensor:
    dh = q.shape[-1]
    scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(dh)
    if is_causal:
        n, m = scores.shape[-2:]
        mask = torch.ones(n, m, device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return torch.einsum("bhnm,bhmd->bhnd", attn, v)

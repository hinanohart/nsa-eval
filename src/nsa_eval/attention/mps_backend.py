"""MPS-native NSA backend — load-bearing for the project's core thesis.

Apple Silicon (M1/M2/M3) is not available on the maintainer's machine. The implementation
here was written against PyTorch's documented MPS public API and is validated on CPU; a Mac
contributor still needs to confirm Metal kernel correctness and perf. Until then, treat MPS
results as un-attested. See issue #1.

Reference: https://arxiv.org/abs/2502.11089 §3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class NSAMPSConfig:
    compression_block_size: int = 32
    compression_stride: int = 16
    selection_block_size: int = 64
    selection_top_n: int = 16
    sliding_window: int = 512
    # Per NSA paper §3.4 the gate is a sigmoid over three branch outputs; bias toward sliding
    # so the locally-correct branch dominates before any training step.
    gate_init_compression: float = -2.0
    gate_init_selection: float = -2.0
    gate_init_sliding: float = 2.0


class NSAMPSBackend(nn.Module):
    """NSA forward implemented with vanilla PyTorch ops so MPS dispatches to Metal kernels.

    Three branches combined by three learnable sigmoid gates:
      1. Compression — strided mean-pooled keys/values, dense attention with a causal mask
         that admits window j only when its right edge is <= q_pos.
      2. Selection — block-summary importance, top-n blocks per query restricted by the same
         causal rule, dense attention on the full-resolution slice.
      3. Sliding — local attention with window `w`; mask handles n != m (decode step).
    """

    name = "nsa_mps"
    supports_devices: tuple[str, ...] = ("mps", "cpu")

    def __init__(self, config: NSAMPSConfig | None = None) -> None:
        super().__init__()
        self.config = config or NSAMPSConfig()
        self.gate_comp = nn.Parameter(torch.tensor(self.config.gate_init_compression))
        self.gate_sel = nn.Parameter(torch.tensor(self.config.gate_init_selection))
        self.gate_slid = nn.Parameter(torch.tensor(self.config.gate_init_sliding))

    def _compression_branch(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        block = self.config.compression_block_size
        stride = self.config.compression_stride
        b, h, n, dh = k.shape
        if n < block:
            return torch.zeros_like(q)
        # unfold(-2, block, stride) -> (b, h, n_win, dh, block); mean over last dim collapses
        # the window into a single key/value summary per window.
        windows_k = k.unfold(-2, block, stride).mean(-1)
        windows_v = v.unfold(-2, block, stride).mean(-1)
        n_win = windows_k.shape[-2]
        scores = torch.einsum("bhnd,bhwd->bhnw", q, windows_k) / math.sqrt(dh)
        q_pos = torch.arange(n, device=q.device)
        win_right = torch.arange(n_win, device=q.device) * stride + block - 1
        allowed = win_right.unsqueeze(0) <= q_pos.unsqueeze(1)  # (n, n_win)
        scores = scores.masked_fill(~allowed.view(1, 1, n, n_win), float("-inf"))
        any_allowed = allowed.any(dim=-1).view(1, 1, n, 1)
        scores = torch.where(any_allowed.expand_as(scores), scores, torch.zeros_like(scores))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhnw,bhwd->bhnd", attn, windows_v)
        return out * any_allowed.to(out.dtype)

    def _selection_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        lp = self.config.selection_block_size
        top_n = self.config.selection_top_n
        b, h, n, dh = k.shape
        # Pad k, v along seq so n is divisible by lp; padded positions are masked out below.
        pad = (-n) % lp
        if pad:
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        n_padded = n + pad
        n_blk = n_padded // lp
        if n_blk == 0:
            return torch.zeros_like(q)
        k_blocks = k.reshape(b, h, n_blk, lp, dh)
        v_blocks = v.reshape(b, h, n_blk, lp, dh)
        block_summary = k_blocks.mean(dim=-2)  # (b, h, n_blk, dh)
        importance = torch.einsum("bhnd,bhkd->bhnk", q, block_summary) / math.sqrt(dh)
        q_pos = torch.arange(n, device=q.device)
        blk_right = torch.arange(n_blk, device=q.device) * lp + lp - 1
        # Causal AND not-padding: block j is admissible for query i iff its right edge is
        # within history (<= i) and within the real (un-padded) sequence (< n).
        allowed_block = (blk_right.unsqueeze(0) <= q_pos.unsqueeze(1)) & (
            blk_right.unsqueeze(0) < n
        )  # (n, n_blk)
        importance = importance.masked_fill(~allowed_block.view(1, 1, n, n_blk), float("-inf"))
        top_k = min(top_n, n_blk)
        top = importance.topk(top_k, dim=-1).indices  # (b, h, n, top_k)
        # Gather full-resolution k, v for selected blocks.
        k_exp = k_blocks.unsqueeze(2).expand(-1, -1, n, -1, -1, -1)
        v_exp = v_blocks.unsqueeze(2).expand(-1, -1, n, -1, -1, -1)
        idx = top.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, lp, dh)
        sel_k = torch.gather(k_exp, 3, idx).reshape(b, h, n, top_k * lp, dh)
        sel_v = torch.gather(v_exp, 3, idx).reshape(b, h, n, top_k * lp, dh)
        # Token-level mask: drop padding tokens, drop tokens at positions > q_pos, drop tokens
        # whose source block was never allowed in the first place.
        pos_in_block = torch.arange(lp, device=q.device).view(1, 1, 1, 1, lp)
        sel_block_start = top.unsqueeze(-1) * lp  # (b, h, n, top_k, 1)
        sel_abs_pos = (sel_block_start + pos_in_block).reshape(b, h, n, top_k * lp)
        token_valid = (sel_abs_pos < n) & (sel_abs_pos <= q_pos.view(1, 1, n, 1))
        allowed_per_top = torch.gather(
            allowed_block.view(1, 1, n, n_blk).expand(b, h, -1, -1), -1, top
        )  # (b, h, n, top_k)
        allowed_per_top_flat = (
            allowed_per_top.unsqueeze(-1).expand(b, h, n, top_k, lp).reshape(b, h, n, top_k * lp)
        )
        valid = token_valid & allowed_per_top_flat
        scores = torch.einsum("bhnd,bhnmd->bhnm", q, sel_k) / math.sqrt(dh)
        scores = scores.masked_fill(~valid, float("-inf"))
        any_valid = valid.any(dim=-1, keepdim=True)
        scores = torch.where(any_valid.expand_as(scores), scores, torch.zeros_like(scores))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhnm,bhnmd->bhnd", attn, sel_v)
        return out * any_valid.to(out.dtype)

    def _sliding_branch(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        w = self.config.sliding_window
        b, h, n, dh = q.shape
        m = k.shape[-2]
        offset = m - n
        i = torch.arange(n, device=q.device).unsqueeze(1)
        j = torch.arange(m, device=q.device).unsqueeze(0)
        mask = (j <= i + offset) & (j > i + offset - w)  # (n, m)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(dh)
        scores = scores.masked_fill(~mask.view(1, 1, n, m), float("-inf"))
        any_valid = mask.any(dim=-1).view(1, 1, n, 1)
        scores = torch.where(any_valid.expand_as(scores), scores, torch.zeros_like(scores))
        attn = scores.softmax(dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        return out * any_valid.to(out.dtype)

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
        if attention_mask is not None:
            raise NotImplementedError(
                "explicit attention_mask is not supported by v0.1; NSA enforces causality "
                "internally per branch (see issue #11)"
            )
        comp = self._compression_branch(q, k, v)
        sel = self._selection_branch(q, k, v)
        slid = self._sliding_branch(q, k, v)
        g_c = self.gate_comp.sigmoid()
        g_s = self.gate_sel.sigmoid()
        g_w = self.gate_slid.sigmoid()
        return g_c * comp + g_s * sel + g_w * slid

    def supports(self, device: torch.device) -> bool:
        return device.type in self.supports_devices

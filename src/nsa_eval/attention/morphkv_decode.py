"""MorphKV decode-time constant-budget adapter — arXiv 2503.00979 (ICML 2025).

Layers on top of any AttentionBackend at decode time to keep the KV cache bounded. v0.1 ships
the interface and a CPU reference loop; an MPS-fused path lands in Phase 2.

Reported numbers from the paper: average KV memory reduction is -52.9% across the evaluated
models. The -83% figure that circulates is the best case on Qwen2.5; do not quote it as the
average.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MorphKVConfig:
    budget: int = 1024
    recency_weight: float = 0.5


class MorphKVDecode:
    """Eviction policy applied per-layer between decode steps."""

    name = "morphkv"

    def __init__(self, config: MorphKVConfig | None = None) -> None:
        self.config = config or MorphKVConfig()

    def evict(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        importance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k_cache.shape != v_cache.shape:
            raise ValueError(
                f"k_cache and v_cache must share shape; got {k_cache.shape} vs {v_cache.shape}"
            )
        if importance.shape != k_cache.shape[:-1]:
            raise ValueError(
                "importance must broadcast to k_cache's leading dims (everything except the "
                f"head-dim axis): expected shape {tuple(k_cache.shape[:-1])}, "
                f"got {tuple(importance.shape)}"
            )
        cached_len = k_cache.shape[-2]
        if cached_len <= self.config.budget:
            return k_cache, v_cache
        recency = torch.linspace(0, 1, cached_len, device=k_cache.device)
        score = (1 - self.config.recency_weight) * importance + (
            self.config.recency_weight * recency.broadcast_to(importance.shape)
        )
        keep = score.topk(self.config.budget, dim=-1).indices.sort(dim=-1).values
        # Insert a trailing singleton dim for dh, then expand to whatever shape k_cache has.
        keep_idx = keep.unsqueeze(-1).expand(*keep.shape, k_cache.shape[-1])
        return torch.gather(k_cache, -2, keep_idx), torch.gather(v_cache, -2, keep_idx)

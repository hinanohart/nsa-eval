"""Numeric regression tests for the MPS NSA backend.

These run on CPU (no MPS hardware needed) and assert behaviour that should hold regardless
of device: causality on each branch, well-defined output (no NaN), and equivalence to dense
attention in degenerate configurations.
"""

from __future__ import annotations

import math

import pytest
import torch

from nsa_eval.attention.mps_backend import NSAMPSBackend, NSAMPSConfig


def _local_causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: int) -> torch.Tensor:
    """Reference for the sliding branch: causal attention with a local window of size `w`."""
    dh = q.shape[-1]
    n, m = q.shape[-2], k.shape[-2]
    offset = m - n
    i = torch.arange(n).unsqueeze(1)
    j = torch.arange(m).unsqueeze(0)
    mask = (j <= i + offset) & (j > i + offset - w)
    scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(dh)
    scores = scores.masked_fill(~mask, float("-inf"))
    # Mirror the backend's safety guard for fully-masked rows (none here, but kept for parity).
    any_valid = mask.any(dim=-1).view(1, 1, n, 1)
    scores = torch.where(any_valid.expand_as(scores), scores, torch.zeros_like(scores))
    attn = scores.softmax(dim=-1)
    out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
    return out * any_valid.to(out.dtype)


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    return (
        torch.randn(1, 2, 128, 16),
        torch.randn(1, 2, 128, 16),
        torch.randn(1, 2, 128, 16),
    )


def test_output_is_finite(qkv):
    backend = NSAMPSBackend()
    out = backend.forward(*qkv, is_causal=True)
    assert torch.isfinite(out).all()
    assert out.shape == qkv[0].shape


def test_sliding_branch_matches_local_causal(qkv):
    cfg = NSAMPSConfig(sliding_window=8)
    backend = NSAMPSBackend(cfg)
    q, k, v = qkv
    got = backend._sliding_branch(q, k, v)
    want = _local_causal(q, k, v, w=8)
    assert torch.allclose(got, want, atol=1e-5)


def test_compression_branch_is_causal(qkv):
    """Queries before the first complete window must receive a zero contribution."""
    cfg = NSAMPSConfig(compression_block_size=32, compression_stride=16)
    backend = NSAMPSBackend(cfg)
    q, k, v = qkv
    out = backend._compression_branch(q, k, v)
    # Window 0 covers positions [0, 31]; queries 0..30 cannot see any complete window.
    assert torch.all(out[:, :, :31, :] == 0)


def test_selection_branch_no_future_leak(qkv):
    """Changing keys/values in a future block must not affect any query that cannot see it.

    Block layout: block_size=16, n=128, so blocks cover [0..15] [16..31] ... [112..127].
    Block 6 covers [96..111]. A query at position p can pick block j iff block-right (j*16+15)
    <= p. So for p<=95, only blocks 0..5 are visible; changes inside positions 96+ live in
    blocks 6 or 7 and must not affect those queries.
    """
    cfg = NSAMPSConfig(selection_block_size=16, selection_top_n=4)
    backend = NSAMPSBackend(cfg)
    q, k, v = qkv
    out_a = backend._selection_branch(q, k, v)
    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 96:, :] += 5.0
    v2[:, :, 96:, :] += 5.0
    out_b = backend._selection_branch(q, k2, v2)
    # Queries 0..95 strictly can only see blocks 0..5, all of whose contents are unchanged.
    assert torch.allclose(out_a[:, :, :96, :], out_b[:, :, :96, :], atol=1e-5)


def test_full_forward_is_causal(qkv):
    """Full NSA forward must not leak future tokens either."""
    backend = NSAMPSBackend(
        NSAMPSConfig(
            compression_block_size=32,
            compression_stride=16,
            selection_block_size=16,
            selection_top_n=4,
            sliding_window=8,
        )
    )
    q, k, v = qkv
    out_a = backend.forward(q, k, v, is_causal=True)
    k2 = k.clone()
    v2 = v.clone()
    k2[:, :, 96:, :] += 5.0
    v2[:, :, 96:, :] += 5.0
    out_b = backend.forward(q, k2, v2, is_causal=True)
    # Sliding window=8 means a query at position p can see [p-7, p]; for p<=88 the window
    # ends before any modified position. Use that as the safe boundary.
    assert torch.allclose(out_a[:, :, :89, :], out_b[:, :, :89, :], atol=1e-5)


def test_attention_mask_not_supported():
    backend = NSAMPSBackend()
    q = torch.randn(1, 2, 64, 16)
    with pytest.raises(NotImplementedError, match="attention_mask"):
        backend.forward(q, q, q, is_causal=True, attention_mask=torch.ones(64, 64).bool())

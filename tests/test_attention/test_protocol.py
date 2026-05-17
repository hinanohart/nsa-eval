"""Protocol-level tests for attention backends — runs on CPU, no GPU needed."""

from __future__ import annotations

import torch

from nsa_eval.attention.mps_backend import NSAMPSBackend, NSAMPSConfig
from nsa_eval.attention.nsa_wrapper import NSAConfig, NSACudaBackend
from nsa_eval.attention.protocol import AttentionBackend


def test_nsa_cuda_implements_protocol():
    backend = NSACudaBackend()
    assert isinstance(backend, AttentionBackend)
    assert backend.name == "nsa_cuda"
    assert "cuda" in backend.supports_devices


def test_nsa_mps_implements_protocol():
    backend = NSAMPSBackend()
    assert isinstance(backend, AttentionBackend)
    assert backend.name == "nsa_mps"
    assert "mps" in backend.supports_devices


def test_nsa_config_defaults():
    cfg = NSAConfig()
    assert cfg.compression_block_size == 32
    assert cfg.selection_top_n == 16
    assert cfg.sliding_window == 512


def test_nsa_mps_config_defaults():
    cfg = NSAMPSConfig()
    assert cfg.compression_block_size == 32
    assert cfg.selection_block_size == 64


def test_nsa_mps_cpu_smoke(small_qkv):
    backend = NSAMPSBackend()
    q, k, v = small_qkv
    out = backend.forward(q, k, v, is_causal=True)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


def test_nsa_cuda_requires_cuda(small_qkv):
    backend = NSACudaBackend()
    q, k, v = small_qkv
    import pytest

    with pytest.raises(RuntimeError, match="requires CUDA"):
        backend.forward(q, k, v, is_causal=True)

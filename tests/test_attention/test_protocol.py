"""Protocol-level tests for attention backends — runs on CPU, no GPU needed."""

from __future__ import annotations

import inspect

import pytest
import torch

from nsa_eval.attention.mps_backend import NSAMPSBackend, NSAMPSConfig
from nsa_eval.attention.nsa_wrapper import NSAConfig, NSACudaBackend
from nsa_eval.attention.protocol import AttentionBackend, conforms


def _has_required_signature(backend: object) -> bool:
    """`runtime_checkable` Protocol only checks attribute names; verify the actual signature
    so a backend with the wrong `forward` arguments cannot pass silently."""
    sig = inspect.signature(backend.forward)
    params = sig.parameters
    return (
        "q" in params
        and "k" in params
        and "v" in params
        and "is_causal" in params
        and "attention_mask" in params
    )


def test_nsa_cuda_conforms_to_protocol():
    backend = NSACudaBackend()
    assert conforms(backend)
    assert _has_required_signature(backend)
    assert backend.name == "nsa_cuda"
    assert "cuda" in backend.supports_devices


def test_nsa_mps_conforms_to_protocol():
    backend = NSAMPSBackend()
    assert conforms(backend)
    assert _has_required_signature(backend)
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
    with pytest.raises(RuntimeError, match="requires CUDA"):
        backend.forward(q, k, v, is_causal=True)


def test_protocol_isinstance_is_known_weak():
    """Documenting Protocol isinstance limitations so future readers don't trust it.

    PEP-544 `runtime_checkable` checks attribute presence only. Add a regression for the
    common foot-gun: an object with the right attribute names but wrong types passes
    `isinstance` but is not a real backend.
    """

    class FakeBackend:
        name = "fake"
        supports_devices: tuple[str, ...] = ()

        def forward(self):  # wrong signature, no q/k/v
            return None

        def supports(self, device):
            return False

    backend = FakeBackend()
    assert isinstance(backend, AttentionBackend)  # the weak check still passes
    assert not _has_required_signature(backend)  # the strong check catches it

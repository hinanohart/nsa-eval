"""pytest configuration — shared fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def small_qkv():
    """Tiny (1, 2, 64, 16) tensor triple for protocol-level tests on CPU."""
    import torch

    torch.manual_seed(0)
    return (
        torch.randn(1, 2, 64, 16),
        torch.randn(1, 2, 64, 16),
        torch.randn(1, 2, 64, 16),
    )

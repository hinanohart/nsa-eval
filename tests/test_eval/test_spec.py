"""Tests for the EvalSpec contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nsa_eval.eval.spec import EvalSpec


def test_spec_slug_is_deterministic():
    s = EvalSpec(model="qwen25_0_5b", benchmark="niah", attention="nsa_cuda", device="cuda")
    assert s.slug() == "qwen25_0_5b_niah_cuda_nsa_cuda_seq8192_seed0"


def test_spec_seqlen_lower_bound():
    with pytest.raises(ValidationError):
        EvalSpec(model="x", benchmark="y", attention="z", device="cpu", seqlen=64)


def test_spec_defaults():
    s = EvalSpec(model="x", benchmark="y", attention="z", device="cpu")
    assert s.seqlen == 8192
    assert s.batch_size == 1
    assert s.seed == 0
    assert s.dtype == "bfloat16"

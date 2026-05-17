"""Hugging Face model loader with patched attention dispatch.

The adapter resolves the requested attention backend via the runner registry and monkey-patches
the attention modules of the loaded model. v0.1 ships the API; the patching logic lands in
week 2 once the protocol is stable.
"""

from __future__ import annotations

from typing import Any


def load_hf(
    model_id: str,
    *,
    attention: str = "full",
    device: str = "cpu",
    dtype: str = "bfloat16",
) -> Any:
    raise NotImplementedError(
        f"HF adapter for {model_id} (attention={attention} device={device} dtype={dtype}) "
        "lands in week 2; tracked in issue #3"
    )

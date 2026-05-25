"""nsa-eval — MPS-native Native Sparse Attention reference + unified long-context evaluation suite.

The public surface is intentionally small. Entry points live in `nsa_eval.cli`. The contract every
attention backend must honour is `nsa_eval.attention.protocol.AttentionBackend`.
"""

from __future__ import annotations

__version__ = "0.0.3"

__all__ = ["__version__"]

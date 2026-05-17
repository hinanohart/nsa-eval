"""Global seed setter — Python, NumPy, PyTorch (CPU + CUDA + MPS)."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # `torch.mps.manual_seed` may be absent on older wheels even when the MPS backend
        # itself is built; guard the attribute access so non-MPS Macs and old PyTorch don't
        # crash the seed setter.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_seed = getattr(getattr(torch, "mps", None), "manual_seed", None)
            if callable(mps_seed):
                mps_seed(seed)
    except ImportError:
        pass

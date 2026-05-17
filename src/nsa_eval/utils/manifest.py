"""Run manifest — records env state needed to reproduce a result."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone


def build_manifest() -> dict:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        pass
    return info

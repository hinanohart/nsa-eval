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
        info["cuda_version"] = torch.version.cuda
        # Record `is_built` separately from `is_available`: a wheel built without MPS but run
        # on a Mac will show built=False; a Linux box shows both False. Useful when triaging
        # "why did the MPS backend fail to allocate" tickets.
        mps = getattr(torch.backends, "mps", None)
        info["mps_built"] = bool(mps and mps.is_built())
        info["mps_available"] = bool(mps and mps.is_available())
    except ImportError:
        pass
    return info

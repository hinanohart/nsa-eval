"""Benchmark adapters. Each adapter implements `cases()` returning iterable of prompts/labels."""

from __future__ import annotations

from .agentbench import AgentBench
from .longbench import LongBench
from .niah import NIAH
from .ruler import RULER

__all__ = ["AgentBench", "LongBench", "NIAH", "RULER"]

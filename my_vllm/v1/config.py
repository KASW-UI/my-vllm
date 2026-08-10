"""Backward-compat re-export — see v1/config/ package for real definitions.

C069: config split into v1/config/vllm.py, model.py, cache.py, scheduler.py.
This module re-exports for backward compatibility (run.py, tests).
"""

from my_vllm.v1.config.vllm import EngineConfig  # noqa: F401

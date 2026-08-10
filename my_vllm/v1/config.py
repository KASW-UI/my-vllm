"""Backward-compat re-export — see config/ package for real definitions.

C069: config split into config/vllm.py, model.py, cache.py, scheduler.py.
Cxxx: config/ moved from v1/config/ to top-level my_vllm/config/ (Plan-B C1).
This module re-exports for backward compatibility.
"""

from my_vllm.config.vllm import EngineConfig, VllmConfig  # noqa: F401

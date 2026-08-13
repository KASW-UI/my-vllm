"""DeviceConfig — execution device configuration.

Mirrors vLLM V1 config/device.py:17.

C137: minimal subset — `device: Device | None = "auto"` plus
`device_type` derived in `__post_init__`. Mirrors vLLM's
single-class shape (no `IrOpPriorityConfig` here — that lives in
`kernel.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

Device = Literal["auto", "cuda", "cpu", "tpu", "xpu"]


@dataclass
class DeviceConfig:
    """Configuration for the device used by my-vllm.

    Mirrors vLLM V1 config/device.py:DeviceConfig.
    """

    device: Device | torch.device | None = "auto"
    """Device type for my-vllm execution. Defaults to auto-detection."""

    device_type: str = field(init=False)
    """Device type from the current platform. Set in __post_init__."""

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device_type = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        elif isinstance(self.device, str):
            self.device_type = self.device
        elif isinstance(self.device, torch.device):
            self.device_type = self.device.type
        else:
            self.device_type = "cpu"

        if self.device is not None and not isinstance(self.device, torch.device):
            self.device = torch.device(self.device_type)

    def compute_hash(self) -> str:
        """Stable hash for cache invalidation (no graph-affecting fields)."""
        return f"DeviceConfig({self.device_type})"
from .interface import CpuArchEnum, DeviceCapability, Platform, PlatformEnum

# Platform singleton — mirrors vllm/platforms/__init__.py:current_platform
current_platform = Platform()
current_platform._enum = PlatformEnum.CUDA
current_platform.device_name = "cuda"
current_platform.device_type = "cuda"

__all__ = [
    "CpuArchEnum",
    "DeviceCapability",
    "Platform",
    "PlatformEnum",
    "current_platform",
]

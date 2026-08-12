import contextlib
import enum
import functools
import os
import platform
import sys
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

_assigned_physical_gpu_ids: list[int] | None = None


def set_assigned_physical_gpu_ids(ids: list[int]) -> None:
    """Set the physical GPU IDs assigned to this worker process.

    Called during worker init so that device_id_to_physical_device_id()
    can map local_rank to the correct physical device without relying
    on CUDA_VISIBLE_DEVICES.

    Idempotent: a second call with the same value is a no-op.
    Raises RuntimeError if called again with a different value.
    """
    global _assigned_physical_gpu_ids
    if _assigned_physical_gpu_ids is not None:
        if _assigned_physical_gpu_ids != ids:
            raise RuntimeError(
                f"set_assigned_physical_gpu_ids called with conflicting values: "
                f"existing={_assigned_physical_gpu_ids}, new={ids}"
            )
        return
    _assigned_physical_gpu_ids = ids


def get_assigned_physical_gpu_ids() -> list[int] | None:
    return _assigned_physical_gpu_ids


@functools.cache
def in_wsl() -> bool:
    return "microsoft" in " ".join(platform.uname()).lower()


class PlatformEnum(enum.Enum):
    """Enumeration of supported hardware platforms."""

    CUDA = enum.auto()
    ROCM = enum.auto()
    TPU = enum.auto()
    XPU = enum.auto()
    CPU = enum.auto()
    OOT = enum.auto()
    UNSPECIFIED = enum.auto()


class CpuArchEnum(enum.Enum):
    X86 = enum.auto()
    ARM = enum.auto()
    POWERPC = enum.auto()
    S390X = enum.auto()
    RISCV = enum.auto()
    OTHER = enum.auto()
    UNKNOWN = enum.auto()


class DeviceCapability(NamedTuple):
    major: int
    minor: int

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    def __hash__(self) -> int:
        return hash((self.major, self.minor))

    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"

    def to_int(self) -> int:
        """Express device capability as an integer <major><minor>.

        It is assumed that the minor version is always a single digit.
        """
        assert 0 <= self.minor < 10
        return self.major * 10 + self.minor


class Platform:
    _enum: PlatformEnum
    device_name: str
    device_type: str

    dispatch_key: str = "CPU"
    ray_device_key: str = ""
    device_control_env_var: str = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"
    ray_noset_device_env_vars: list[str] = []
    simple_compile_backend: str = "inductor"
    dist_backend: str = ""
    supported_quantization: list[str] = []
    additional_env_vars: list[str] = []

    _global_graph_pool: Any | None = None

    @property
    def pass_key(self) -> str:
        """Inductor config key for the PassManager custom pass"""
        return "post_grad_custom_post_pass"

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        """Returns the supported dtypes for the current platform.

        The first dtype will be used as the default dtype fallback.
        """
        return [torch.bfloat16, torch.float16, torch.float32]

    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    def is_tpu(self) -> bool:
        return self._enum == PlatformEnum.TPU

    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    def uses_host_device_handling(self) -> bool:
        return self.is_tpu()

    def is_zen_cpu(self) -> bool:
        return False

    def is_out_of_tree(self) -> bool:
        return self._enum == PlatformEnum.OOT

    def is_unspecified(self) -> bool:
        return self._enum == PlatformEnum.UNSPECIFIED

    def get_max_output_tokens(self, prompt_len: int) -> int:
        return sys.maxsize

    def is_cuda_alike(self) -> bool:
        """Stateless version of torch.cuda.is_available()."""
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)

    def is_sleep_mode_available(self) -> bool:
        return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM, PlatformEnum.XPU)

    @classmethod
    def get_pass_manager_cls(cls) -> str:
        return "vllm.compilation.passes.pass_manager.PostGradPassManager"

    @classmethod
    def get_compile_backend(cls) -> str:
        return cls.simple_compile_backend

    @classmethod
    def import_ir_kernels(cls) -> None:
        """Import IR kernel implementations."""
        pass

    @classmethod
    def device_control_id_to_physical_device_id(cls, device_id: str) -> int:
        """Map one device-control env entry to an integer physical device ID."""
        try:
            return int(device_id)
        except ValueError as e:
            raise ValueError(
                f"Non-integer device ID {device_id!r} is not supported by "
                f"{cls.device_name}."
            ) from e

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int) -> int:
        """Map a vLLM-local logical device ID to a physical device ID.

        The input is a logical local ID (e.g. a local rank), NOT a visible
        device ordinal.
        """
        if _assigned_physical_gpu_ids is not None:
            if device_id >= len(_assigned_physical_gpu_ids):
                raise IndexError(
                    f"device_id {device_id} is out of range for "
                    f"assigned_physical_gpu_ids {_assigned_physical_gpu_ids} "
                    f"({len(_assigned_physical_gpu_ids)} devices assigned)"
                )
            return _assigned_physical_gpu_ids[device_id]
        if (
            cls.device_control_env_var in os.environ
            and os.environ[cls.device_control_env_var] != ""
        ):
            device_ids = os.environ[cls.device_control_env_var].split(",")
            physical_device_id = device_ids[device_id]
            return cls.device_control_id_to_physical_device_id(physical_device_id)
        else:
            return device_id

    @classmethod
    def logical_device_id_to_visible_device_id(cls, device_id: int) -> int:
        """Map a vLLM-local logical device ID to the current process's
        visible accelerator ordinal.
        """
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        device_control_env = os.environ.get(cls.device_control_env_var, "")
        if not device_control_env:
            return physical_device_id
        visible_physical_device_ids = [
            cls.device_control_id_to_physical_device_id(physical_id)
            for physical_id in device_control_env.split(",")
        ]
        if physical_device_id not in visible_physical_device_ids:
            raise RuntimeError(
                f"Physical device {physical_device_id} for logical device "
                f"{device_id} is not visible in {cls.device_control_env_var}="
                f"{device_control_env}"
            )
        return visible_physical_device_ids.index(physical_device_id)

    @classmethod
    def visible_device_id_to_physical_device_id(cls, device_id: int) -> int:
        """Map a visible accelerator ordinal to a physical device ID."""
        device_control_env = os.environ.get(cls.device_control_env_var, "")
        if not device_control_env:
            return device_id
        visible_device_ids = device_control_env.split(",")
        if device_id >= len(visible_device_ids):
            raise IndexError(
                f"visible device ordinal {device_id} is out of range for "
                f"{cls.device_control_env_var}={device_control_env}"
            )
        return cls.device_control_id_to_physical_device_id(visible_device_ids[device_id])

    @classmethod
    def import_kernels(cls) -> None:
        """Import platform-specific C kernels."""
        with contextlib.suppress(ImportError):
            import vllm._C  # noqa: F401

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        """Stateless version of torch.cuda.get_device_capability()."""
        return None

    @classmethod
    def has_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """Test whether this platform is compatible with a device capability.

        The capability argument can be a tuple (major, minor) or
        an integer <major><minor>.
        """
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        if isinstance(capability, tuple):
            return current_capability >= capability
        return current_capability.to_int() >= capability

    @classmethod
    def is_device_capability(
        cls,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        """Test whether this platform has exactly the specified device capability."""
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        if isinstance(capability, tuple):
            return current_capability == capability
        return current_capability.to_int() == capability

    @classmethod
    def is_device_capability_family(
        cls,
        capability: int,
        device_id: int = 0,
    ) -> bool:
        """Returns True if the device capability is any <major>.x."""
        current_capability = cls.get_device_capability(device_id=device_id)
        if current_capability is None:
            return False
        return (current_capability.to_int() // 10) == (capability // 10)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """Get the name of a device."""
        return cls.device_name

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """Get the total memory of device in bytes."""
        raise NotImplementedError

    @classmethod
    def get_current_memory_usage(cls, device_id: int = 0) -> int:
        """Get current memory usage in bytes."""
        raise NotImplementedError

    @classmethod
    def supports_fp8(cls, device_id: int = 0) -> bool:
        """Check if FP8 is supported on the device."""
        return cls.has_device_capability(89, device_id)

    @classmethod
    def supports_mx(cls, device_id: int = 0) -> bool:
        """Check if Micro-Exponent (MX) formats are supported."""
        return cls.has_device_capability(100, device_id)

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        """Check if FP8 uses the FNUZ variant."""
        return False

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """Return the FP8 dtype for this platform."""
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None) -> str:
        """Get the attention backend class of a device."""
        return ""

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list:
        return []

    @classmethod
    def get_vit_attn_backend(cls, head_size: int, dtype: torch.dtype, backend=None):
        """Get the vision attention backend class of a device."""
        return None

    def is_cumem_allocator_available(self) -> bool:
        return False

    def get_attn_backend_for_batch(
        self, attn_backend, common_attn_metadata
    ):
        """Get the attention backend for the current batch."""
        return attn_backend

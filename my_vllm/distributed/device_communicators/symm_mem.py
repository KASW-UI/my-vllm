"""my_vllm/distributed/device_communicators/symm_mem.py

Symmetric memory communicator — NVLS (NVIDIA-Link-Switch) accelerated
collective primitives.

C241: extracted from ``vllm/distributed/device_communicators/symm_mem.py``
(155 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``SymmMemCommunicator`` class — kept verbatim (the thin
    wrapper around ``torch.distributed._symmetric_memory``).
  * ``disabled`` graceful-fallback flag — kept.
  * ``force_multimem`` / ``max_size_override`` test-only params
    — kept (so the heuristics can be tested in isolation).
  * ``VLLM_BATCH_INVARIANT`` check — adapted to ``MY_VLLM_BATCH_INVARIANT``.
  * NVLS environment validation (``current_platform.is_cuda``
    + device-capability table) — kept.
  * P2P runtime probing (``gpu_p2p_access_check`` /
    ``can_actually_p2p``) — not vendored here (those live in
    ``all_reduce_utils.py`` in vLLM, but my-vllm omits them
    because the dev box is a single GPU).

Hardware requirements:
  * **NVLS** (NVIDIA-Link-Switch) hardware — Hopper (sm_90),
    Blackwell (sm_100, sm_100a, sm_120).  RTX 4060 (sm_89)
    does NOT support NVLS; ``SymmMemCommunicator`` is
    gracefully disabled (`self.disabled = True`) on such
    hardware.
  * **PyTorch >= 2.6** — has the ``torch.ops.symm_mem`` ops.
  * **Multi-GPU same-host** — rendezvous is over the NCCL
    process group.

References:
  * vLLM ``symm_mem.py`` — original source.
  * NCCL NVLS docs: ``ncclCommWindowRegister`` + NVLS hardware.
  * PyTorch symmetric memory docs:
    ``torch.distributed._symmetric_memory``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from my_vllm.distributed.device_communicators.all_reduce_utils import (
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)

logger = logging.getLogger(__name__)


try:
    import torch.distributed._symmetric_memory as _symmetric_memory

    _SYMM_MEM_AVAILABLE = True
except ImportError:
    _symmetric_memory = None
    _SYMM_MEM_AVAILABLE = False


class SymmMemCommunicator:
    """NVLS-accelerated all-reduce via PyTorch's symmetric-memory runtime.

    C241: mirrors ``vllm.distributed.device_communicators.symm_mem
    .SymmMemCommunicator``.  When ``self.disabled is True``, all
    collective operations fall back to ``torch.distributed`` (the
    caller is responsible for that — see
    ``CudaCommunicator.all_reduce``).

    The communicator allocates a single ``max_size`` symmetric-memory
    buffer at construction time and reuses it across all-reduce calls
    in the same step sequence.

    Args:
        group: NCCL process group.
        device: target CUDA device (int, str, or ``torch.device``).
        force_multimem: test-only override — forces / disables the
            multimem-path (vs the two-shot fall-back).  ``None``
            lets the runtime pick based on world_size.
        max_size_override: test-only — overrides the size
            selected from ``SYMM_MEM_ALL_REDUCE_MAX_SIZES``.
    """

    _WORLD_SIZES_MULTIMEM: dict[str, list[int]] = {
        "9.0": [4, 6, 8],
        "10.0": [6, 8],
        "10.3": [6, 8],
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        force_multimem: bool | None = None,
        max_size_override: int | None = None,
    ) -> None:
        self.group = group
        self.device = (
            torch.device(device)
            if not isinstance(device, torch.device)
            else device
        )
        self.dtype = torch.bfloat16
        self.force_multimem = force_multimem
        self.max_size_override = max_size_override
        self.disabled = True

        if not _SYMM_MEM_AVAILABLE:
            return
        if not torch.cuda.is_available():
            return
        if self.device.type != "cuda":
            return
        try:
            torch.cuda.set_device(self.device)
        except Exception:
            return

        try:
            self.world_size = dist.get_world_size(self.group)
        except Exception:
            return

        capability = self._safe_get_device_capability()
        if capability is None:
            return
        device_capability_str = "{}.{}".format(capability[0], capability[1])
        self.device_capability = device_capability_str
        if device_capability_str not in SYMM_MEM_ALL_REDUCE_MAX_SIZES:
            return
        if self.world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[
            device_capability_str
        ]:
            return

        if max_size_override is not None:
            self.max_size = max_size_override
        else:
            self.max_size = SYMM_MEM_ALL_REDUCE_MAX_SIZES[
                device_capability_str
            ][self.world_size]

        try:
            self.buffer = _symmetric_memory.empty(
                self.max_size // self.dtype.itemsize,
                device=self.device,
                dtype=self.dtype,
            )
            handle = _symmetric_memory.rendezvous(
                self.buffer, self.group.group_name
            )
        except (RuntimeError, ValueError):
            return
        if getattr(handle, "multicast_ptr", 0) == 0:
            return
        if os.environ.get("MY_VLLM_BATCH_INVARIANT", "0") in ("1", "true"):
            return
        self.disabled = False

    @staticmethod
    def _safe_get_device_capability() -> tuple[int, int] | None:
        """Return ``(major, minor)`` capability, or ``None`` on failure."""
        try:
            major, minor = torch.cuda.get_device_capability()
            return (major, minor)
        except Exception:
            return None

    def should_use_symm_mem(self, inp: torch.Tensor) -> bool:
        """Return True iff ``inp`` is eligible for the symm-mem path."""
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 4 != 0:
            return False
        return inp_size <= self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """NVLS all-reduce on ``inp``, writing into ``out``.

        Returns ``None`` when symm-mem is not eligible (the caller
        falls back to ``torch.distributed.all_reduce``); returns
        ``out`` (or a fresh allocation) on success.

        Algorithm:
          1. Copy ``inp`` into the symmetric memory buffer.
          2. Run ``torch.ops.symm_mem.multimem_all_reduce_`` (Hopper+
             multicast hardware) when ``world_size`` is supported;
             otherwise fall back to ``two_shot_all_reduce_``.
          3. Copy the buffer back into ``out`` (or a fresh tensor
             of the same shape / dtype / device as ``inp``).
        """
        if not self.should_use_symm_mem(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.reshape(-1))

        use_multimem = self._should_use_multimem()
        if use_multimem:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out

    def _should_use_multimem(self) -> bool:
        """Decide whether to use the multimem path."""
        if self.force_multimem is not None:
            return self.force_multimem
        if self.device_capability not in self._WORLD_SIZES_MULTIMEM:
            return False
        return (
            self.world_size
            in self._WORLD_SIZES_MULTIMEM[self.device_capability]
        )


__all__ = ["SymmMemCommunicator", "_SYMM_MEM_AVAILABLE"]


def is_symmetric_memory_available() -> bool:
    """Module-level helper for callers that want to gate on runtime support."""
    return _SYMM_MEM_AVAILABLE

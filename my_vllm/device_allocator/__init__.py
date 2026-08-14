"""Device allocator package — singleton memory pool + sleep/wake_up.

Mirrors vLLM ``vllm/device_allocator/__init__.py`` (50 lines,
2025-04 snapshot).

This package provides:

  * ``HandleType`` / ``AllocationData`` — the per-allocation
    bookkeeping record (matches vLLM's structure verbatim).
  * ``MemAllocator`` — the Protocol interface that any
    platform-specific allocator must implement.
  * ``get_mem_allocator_instance()`` — platform-routed factory;
    CUDA platforms return a ``CumemAllocator`` singleton
    (see ``my_vllm.device_allocator.cumem``).

VMM in my-vllm
---------------

Real vLLM's cumem path uses a **C extension** (loaded as
``libcumem_allocator.so``) that calls ``cuMemCreate`` /
``cuMemAddressReserve`` / ``cuMemMap`` / ``cuMemUnmap`` directly.
That extension handles the per-allocation physical chunk
mapping and is what enables the >50% KV-cache memory savings
on Hopper-class GPUs.

my-vllm **does not vendor the C extension**.  Building it
would require ~500 lines of C++/CUDA, CMake wiring into
``setup.py``, and a runtime-loaded library dependency — all
out of scope for the engineering subset (JD-aligned; CPU
inference is out of scope per the cleanup-plan, and a single-
GPU RTX 4060 doesn't benefit from VMM the way a Hopper H100
with many small sequences does).

What C231 ships instead:

  1. The **architectural surface** — singleton, ``use_memory_pool``
     context manager, ``sleep`` / ``wake_up`` API, tag-based
     bookkeeping (``AllocationData``).  Production code that
     imports ``get_mem_allocator_instance()`` and calls these
     methods will work unchanged against a future real C
     extension — the API is drop-in compatible with vLLM's.

  2. A **Python-level implementation** of ``sleep`` / ``wake_up``
     that uses PyTorch's standard caching allocator APIs
     (``torch.cuda.synchronize`` + ``torch.cuda.empty_cache``).
     This frees as much cached memory as possible without the
     per-tensor unmapping that the C extension provides.  It's
     correct (no crashes, no leaks) but doesn't recover the
     physical-memory savings of real VMM.

  3. ``Platform.is_cumem_allocator_available()`` returns True
     on CUDA platforms with SM >= 7.0 (the VMM-capable
     threshold).  my-vllm's CUDA path uses this to route to
     ``CumemAllocator``; CPU / no-CUDA platforms get a
     no-op allocator.

The sleep-mode use case (offload KV cache to CPU between
scaling events) requires the C extension.  Production vLLM
ships this; my-vllm's subset skips it.  See
``docs/snapshots/C231.md`` for the full scope decision.
"""

from __future__ import annotations

import dataclasses
from contextlib import AbstractContextManager
from typing import Protocol, TypeAlias

# HandleType matches vLLM's tuple shape:
#   (py_device, py_size_or_aligned_size, py_ptr, py_handle)
# On CUDA the trailing ``py_handle`` is a ``list[int]`` of
# physical chunk IDs returned by the C extension; on ROCm it's
# just an int.  my-vllm ships CUDA-only, so we type it as
# ``list[int]``.
HandleType: TypeAlias = tuple[int, int, int, list[int]]


@dataclasses.dataclass
class AllocationData:
    """Per-allocation bookkeeping record.

    Mirrors ``vllm/device_allocator/__init__.py:AllocationData``.

    Attributes:
        handle: the underlying allocation handle (py_device, size,
            ptr, physical-chunk-list).  For the Python-only path
            ``handle[3]`` is always ``[]``.
        tag: user-provided tag identifying which memory pool this
            allocation belongs to (``"default"`` / ``"kv_cache"``
            / etc.).  Used by ``sleep`` to decide which allocations
            to offload vs discard.
        cpu_backup_tensor: optional pinned CPU tensor holding the
            offloaded contents during sleep.  ``None`` while
            awake.  In the real C-extension path this is a
            pinned byte tensor of size ``handle[1]``.
        is_asleep: True between ``sleep`` and the matching
            ``wake_up``.
    """

    handle: HandleType
    tag: str
    cpu_backup_tensor: "object | None" = None
    is_asleep: bool = False


class MemAllocator(Protocol):
    """Allocator Protocol — every backend must implement this.

    Mirrors ``vllm/device_allocator/__init__.py:MemAllocator``.
    All methods are no-ops in the CPU / no-CUDA path; the CUDA
    path delegates to ``CumemAllocator``.
    """

    def use_memory_pool(
        self, tag: str | None = None
    ) -> AbstractContextManager:
        """Context manager that routes subsequent allocations to
        the memory pool tagged ``tag`` (default: ``"default"``).

        Mirrors vLLM's ``CuMemAllocator.use_memory_pool``."""

    def sleep(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
    ) -> None:
        """Free the physical memory backing the pool.

        Allocations whose tag is in ``offload_tags`` are copied
        to pinned CPU memory before being freed (so they can be
        restored on ``wake_up``).  The rest are discarded.

        Default (``offload_tags=None``) is to offload the
        ``"default"`` tag — i.e., the standard path."""

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Restore the offloaded allocations to GPU memory.

        ``tags=None`` restores everything; a list filters."""

    def get_current_usage(self) -> int:
        """Total bytes currently allocated in the pool."""


def get_mem_allocator_instance() -> MemAllocator:
    """Platform-routed factory.

    Mirrors ``vllm/device_allocator/__init__.py:
    get_mem_allocator_instance``.

    CUDA (with VMM-capable SM) → ``CumemAllocator`` singleton.
    CPU / no-CUDA → raises ``RuntimeError`` (sleep mode is
    not a meaningful operation without a CUDA device).
    """
    # Import here to avoid circular import at module load
    # (platforms import from device_allocator in some setups).
    from my_vllm.platforms import current_platform

    if current_platform.is_cuda_alike():
        from my_vllm.device_allocator.cumem import CumemAllocator

        return CumemAllocator.get_instance()

    raise RuntimeError(
        "Sleep mode allocator is not available on platform "
        f"{type(current_platform).__name__} "
        f"(device_type={current_platform.device_type}). "
        "CumemAllocator requires CUDA."
    )


__all__ = [
    "AllocationData",
    "HandleType",
    "MemAllocator",
    "get_mem_allocator_instance",
]
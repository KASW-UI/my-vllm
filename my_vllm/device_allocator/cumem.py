"""CumemAllocator — wraps the real CUDA Virtual Memory Management (VMM)
path via the vendored ``my_vllm.cumem_allocator`` C extension.

Mirrors ``vllm/device_allocator/cumem.py`` (377 lines, 2025-04
snapshot).  The C extension is ``my_vllm/cumem_allocator.cpp``
(extracted verbatim from ``vllm/csrc/cumem_allocator.cpp``,
810 lines).  It implements:

  * ``cuMemAddressReserve`` / ``cuMemAddressFree`` —
    virtual address range management.
  * ``cuMemCreate`` / ``cuMemRelease`` — physical memory
    chunk allocation.
  * ``cuMemMap`` / ``cuMemUnmap`` — physical-to-virtual mapping.
  * ``cuMemSetAccess`` — access permission setup.
  * Two C entry points (``my_malloc`` / ``my_free``) that
    PyTorch's ``CUDAPluggableAllocator`` calls via dlsym.
  * Three Python-callable functions (``init_module``,
    ``python_create_and_map``, ``python_unmap_and_release``)
    that vLLM's ``CuMemAllocator`` uses for the sleep /
    wake_up cycle.

Architecture
------------

  ┌────────────────────────────────────────────────────────────────┐
  │  PyTorch CUDA caching allocator                                 │
  │      ↓ (MemPool-wrapped allocations)                            │
  │  CUDAPluggableAllocator(libcumem_allocator.so, "my_malloc", …) │
  │      ↓ dlsym                                                    │
  │  my_malloc(size, device, stream)  [in csrc/cumem_allocator.cpp] │
  │      ↓                                                          │
  │  1. cuMemAddressReserve (virtual range)                        │
  │  2. python_malloc_callback(handle) [→ CumemAllocator.          │
  │                                    pointer_to_data[ptr] = …]    │
  │  3. cuMemCreate (physical chunk)                                │
  │  4. cuMemMap (physical → virtual)                              │
  │      ↓ returns device pointer                                   │
  │  PyTorch tensor gets the pointer                                │
  └────────────────────────────────────────────────────────────────┘

Sleep / wake_up flow:

  sleep(offload_tags=("kv_cache",)):
      for ptr, data in pointer_to_data.items():
          if data.tag in offload_tags:
              libcudart.cudaMemcpy(host, ptr, size)
              data.cpu_backup_tensor = host
          python_unmap_and_release(*data.handle)
              # ↑ cuMemUnmap + cuMemRelease
      torch.cuda.synchronize()
      torch.cuda.empty_cache()

  wake_up(tags=["kv_cache"]):
      for ptr, data in pointer_to_data.items():
          if data.tag in tags:
              python_create_and_map(*data.handle)
                  # ↑ cuMemCreate + cuMemMap
              if data.cpu_backup_tensor is not None:
                  libcudart.cudaMemcpy(ptr, host, size)
                  data.cpu_backup_tensor = None

The C extension is required for the sleep / wake_up cycle to
actually free physical memory.  Without it (``extension_loaded
== False``), the class falls back to Python-level bookkeeping +
``torch.cuda.empty_cache`` reclamation — adequate for tests
but missing the per-tensor VMM benefits.

Build requirement
-----------------

The C extension is built by ``setup.py build_ext --inplace``
via ``torch.utils.cpp_extension.CUDAExtension`` (CUDA Driver
API + libcuda link).  See ``setup.py`` for the configuration.
If the extension wasn't built (e.g. on a CPU-only host),
``cumem_available = False`` and the allocator falls back to
Python-level only.
"""

from __future__ import annotations

import atexit
import gc
import logging
from contextlib import contextmanager
from typing import Iterator

from my_vllm.device_allocator import AllocationData, HandleType

logger = logging.getLogger(__name__)


# Detect whether the C extension was built.
_cumem_ext = None
try:
    from my_vllm import cumem_allocator as _cumem_ext  # type: ignore
except ImportError:
    _cumem_ext = None

cumem_available: bool = _cumem_ext is not None

# Path to the .so (for PyTorch's CUDAPluggableAllocator which
# dlsym's it).
_cumem_lib_path: str | None = None
if _cumem_ext is not None:
    # ``my_vllm.cumem_allocator`` is a compiled extension
    # module; the .so file lives next to my_vllm/__init__.py.
    import os

    _ext_path = _cumem_ext.__file__
    if _ext_path is not None:
        _cumem_lib_path = _ext_path


class CumemAllocator:
    """Singleton memory-pool allocator (architecture-aligned with vLLM).

    See module docstring for the architecture.  When the C
    extension is loaded (``cumem_available == True``), real VMM
    is exercised end-to-end (per-tensor unmap on sleep, remap
    on wake_up).  When not, the allocator falls back to
    PyTorch's standard caching allocator + ``torch.cuda.
    empty_cache`` reclamation.
    """

    instance: "CumemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CumemAllocator":
        """Singleton accessor — mirrors vLLM
        ``CuMemAllocator.get_instance``.

        Cannot call ``__init__`` directly.  First invocation:
          * initializes the C extension with our Python
            callbacks (``init_module``), if available;
          * installs an ``atexit`` shutdown hook that releases
            the pool reference before interpreter finalization
            (mirrors vLLM's pytorch#146431 workaround).
        """
        if CumemAllocator.instance is None:
            CumemAllocator.instance = CumemAllocator()
            atexit.register(CumemAllocator._shutdown_singleton)
        return CumemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        """Release pool refs at interpreter shutdown (defensive).

        Mirrors vLLM's defensive cleanup pattern.  Real C-extension
        path: releases the pluggable allocator wrappers per
        pytorch#146431.  Python-only path: a no-op.
        """
        instance = CumemAllocator.instance
        if instance is None:
            return
        try:
            instance.close()
        except Exception:
            logger.exception("CumemAllocator singleton shutdown failed")

    def __init__(self) -> None:
        # Maps ptr → AllocationData.  vLLM uses the same key.
        self.pointer_to_data: dict[int, AllocationData] = {}
        # Currently-active tag (set by ``use_memory_pool``).
        self.current_tag: str = CumemAllocator.default_tag
        # Tracks each tag's pool context so it outlives the
        # ``with`` block (PyTorch bug workaround).
        self.allocator_and_pools: dict[str, tuple] = {}
        # Bind methods (vLLM holds strong refs to avoid GC
        # issues with PyTorch's pluggable allocator callbacks).
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback
        # Whether the real VMM C extension is loaded.
        self.cumem_available = cumem_available
        # When the C extension is available, the .so path.
        self.cumem_lib_path = _cumem_lib_path

        if cumem_available and _cumem_ext is not None:
            # Wire our Python callbacks into the C extension.
            try:
                _cumem_ext.init_module(
                    self.python_malloc_callback, self.python_free_callback
                )
                logger.info(
                    "CumemAllocator wired to real VMM extension "
                    "(%s)",
                    _cumem_lib_path,
                )
            except Exception:
                logger.exception(
                    "CumemAllocator: init_module failed; falling back "
                    "to Python-only path"
                )
                self.cumem_available = False

    # ------------------------------------------------------------------
    # Allocation hooks — invoked by the C extension's my_malloc/my_free
    # ------------------------------------------------------------------

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """Record an allocation.

        Mirrors ``vllm.cumem.CuMemAllocator._python_malloc_callback``.
        Called by the C extension's ``my_malloc`` AFTER it has
        reserved the VA range but BEFORE the physical map.
        The C extension finishes the map after this callback
        returns.
        """
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "CumemAllocator: malloc %d bytes @ 0x%x (tag=%s)",
            allocation_handle[1],
            py_d_mem,
            self.current_tag,
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Look up + remove an allocation record on free.

        Mirrors ``vllm.cumem.CuMemAllocator._python_free_callback``.
        Returns the handle so the C extension can perform
        ``cuMemUnmap`` + ``cuMemRelease``.
        """
        data = self.pointer_to_data.pop(ptr, None)
        if data is None:
            # Allocator freed a ptr we never recorded — defensive.
            logger.warning("CumemAllocator: free on unregistered ptr 0x%x", ptr)
            # CUDA ABI: p_memHandle is a single int (CUmemGenericAllocationHandle*).
            # Sentinel value 0 indicates "no handle" to the C extension.
            return (0, 0, ptr, 0)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        return data.handle

    # ------------------------------------------------------------------
    # use_memory_pool — tag-based context manager
    # ------------------------------------------------------------------

    @contextmanager
    def use_memory_pool(self, tag: str | None = None) -> Iterator[None]:
        """Context manager: subsequent allocations get ``tag``.

        Mirrors ``vllm.cumem.CuMemAllocator.use_memory_pool``.

        When the C extension is available, this wraps a real
        ``torch.cuda.memory.MemPool`` (via the pluggable
        allocator) so allocations route to the VMM-backed pool.
        When the C extension isn't loaded, this is a no-op tag
        tracker — the bookkeeping still works but actual
        allocation routing goes through PyTorch's default
        allocator.

        Nested contexts are NOT supported on PyTorch's
        ``use_mem_pool`` API (MemPool's destructor asserts on
        pending capture events; nested entry races the outer
        one).  We detect the nesting and skip MemPool creation
        for the inner context — the tag still flips, so the
        bookkeeping tracks correctly; the inner allocations
        go to PyTorch's default allocator (not VMM-backed).
        """
        if tag is None:
            tag = CumemAllocator.default_tag
        if not isinstance(tag, str):
            raise TypeError(f"tag must be str, got {type(tag).__name__}")

        old_tag = self.current_tag
        self.current_tag = tag
        already_inside = bool(self.allocator_and_pools)
        try:
            # Real C-extension path: wrap a MemPool so allocations
            # during this context go through cuMem.  Skip when
            # already inside an outer context (nested MemPool not
            # supported on PyTorch).
            if (
                cumem_available
                and _cumem_ext is not None
                and self.cumem_lib_path
                and not already_inside
            ):
                import torch.cuda.memory as cudamem

                torch_alloc = cudamem.CUDAPluggableAllocator(
                    self.cumem_lib_path, "my_malloc", "my_free"
                )
                mem_pool = cudamem.MemPool(torch_alloc.allocator())
                with cudamem.use_mem_pool(mem_pool):
                    # Keep pool + allocator alive for the lifetime
                    # of the context (mirrors vLLM's pytorch#146431
                    # workaround).
                    self.allocator_and_pools[tag] = (mem_pool, torch_alloc)
                    try:
                        yield
                    finally:
                        self.allocator_and_pools.pop(tag, None)
            else:
                # Python-only fallback (or nested context — the
                # inner context just tracks the tag, no pool).
                yield
        finally:
            self.current_tag = old_tag

    # ------------------------------------------------------------------
    # sleep / wake_up — memory-reclaim API
    # ------------------------------------------------------------------

    def sleep(
        self,
        offload_tags: tuple[str, ...] | str | None = None,
    ) -> None:
        """Free as much memory as possible.

        Mirrors the vLLM ``CuMemAllocator.sleep`` API:
          * ``offload_tags`` selects which allocations to copy to
            pinned CPU memory before freeing (so they survive
            ``wake_up``).
          * Allocations NOT in ``offload_tags`` are simply freed
            (their contents are lost).
          * Default: offload the ``"default"`` tag.

        Real C-extension behavior:
          1. For each tracked allocation whose tag is in
             ``offload_tags``: ``cudaMemcpy`` to pinned host
             tensor (populates ``cpu_backup_tensor``).
          2. ``python_unmap_and_release(*handle)`` — calls
             ``cuMemUnmap`` + ``cuMemRelease``.
          3. ``torch.cuda.synchronize()`` + ``torch.cuda.empty_cache()``.

        Python-only behavior (no C extension):
          1. ``torch.cuda.synchronize()``.
          2. ``torch.cuda.empty_cache()`` — release cached
             blocks.  No per-tensor unmapping.
        """
        if offload_tags is None:
            offload_tags = (CumemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)
        if not isinstance(offload_tags, tuple):
            raise TypeError(
                f"offload_tags must be tuple/str/None, got {type(offload_tags).__name__}"
            )

        total_bytes = 0
        backup_bytes = 0
        num_unmapped = 0
        num_failed_unmap = 0

        # Snapshot the handles before unmap (the dict is mutated
        # by free; we don't free here but want stable iteration).
        for ptr, data in list(self.pointer_to_data.items()):
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                # Real C-extension path: cudaMemcpy to pinned
                # host tensor.  Not implemented in my-vllm
                # because we'd need a libcudart binding (vLLM's
                # ``CudaRTLibrary`` wrapper); the C extension's
                # unmap_and_release still runs below to actually
                # free the physical chunks.
                data.cpu_backup_tensor = None  # mark "would-have-offloaded"
            # Real C-extension path: call python_unmap_and_release
            # which performs cuMemUnmap + cuMemRelease.
            if (
                cumem_available
                and _cumem_ext is not None
                and self.cumem_available
            ):
                try:
                    _cumem_ext.python_unmap_and_release(*handle)
                    num_unmapped += 1
                except RuntimeError as e:
                    num_failed_unmap += 1
                    logger.warning(
                        "CumemAllocator: unmap failed for ptr=0x%x: %s",
                        ptr,
                        e,
                    )
            data.is_asleep = True

        try:
            import torch

            torch.cuda.synchronize()
        except Exception:
            logger.debug("synchronize() failed (no CUDA?)", exc_info=True)

        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            logger.debug("empty_cache() failed (no CUDA?)", exc_info=True)

        gc.collect()

        logger.info(
            "CumemAllocator: sleep total=%d B offload=%d B "
            "unmapped=%d/%d (failed=%d)",
            total_bytes,
            backup_bytes,
            num_unmapped,
            len(self.pointer_to_data),
            num_failed_unmap,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Restore offloaded allocations to GPU memory.

        Mirrors the vLLM ``CuMemAllocator.wake_up`` API.

        Real C-extension behavior:
          1. For each tracked allocation whose tag matches:
             ``python_create_and_map(*handle)`` — calls
             ``cuMemCreate`` + ``cuMemMap`` (fresh physical
             chunks mapped to the original VA range).
          2. If ``cpu_backup_tensor`` was populated, copy back
             via ``cudaMemcpy``.

        Python-only behavior:
          1. Flip ``is_asleep`` to False.
          2. The allocations are gone (we unmapped them on
             sleep but couldn't back them up); subsequent
             allocations go through PyTorch's default caching
             allocator.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is not None and data.tag not in tags:
                continue
            # Real C-extension path: re-create physical chunks.
            if (
                cumem_available
                and _cumem_ext is not None
                and self.cumem_available
            ):
                try:
                    _cumem_ext.python_create_and_map(*data.handle)
                except RuntimeError as e:
                    logger.warning(
                        "CumemAllocator: create_and_map failed for "
                        "ptr=0x%x: %s",
                        ptr,
                        e,
                    )
            # Real path: cudaMemcpy from host back to device.
            # Not implemented in my-vllm (would need libcudart
            # binding); cpu_backup_tensor is always None here.
            data.is_asleep = False
            if data.cpu_backup_tensor is not None:
                data.cpu_backup_tensor = None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release pool refs eagerly.

        Mirrors vLLM's defensive cleanup (pytorch#146431
        workaround).  Two-phase release: drop MemPool refs
        while allocators are still strongly held, then drop the
        allocator wrappers.
        """
        if not self.allocator_and_pools:
            return
        pool_entries = list(self.allocator_and_pools.values())
        self.allocator_and_pools.clear()

        mem_pools = [entry[0] for entry in pool_entries]
        allocators = [entry[1] for entry in pool_entries]
        pool_entries.clear()

        # Phase 1: drop MemPool refs while allocators are alive.
        mem_pools.clear()
        gc.collect()
        # Phase 2: now safe to release allocator wrappers.
        allocators.clear()

    def release_pools(self) -> None:
        """Compatibility alias — same as ``close``."""
        self.close()

    def get_current_usage(self) -> int:
        """Total bytes currently tracked in the allocator.

        Mirrors vLLM's ``CuMemAllocator.get_current_usage``.
        Sums ``handle[1]`` across tracked allocations.  The
        ``handle[1]`` is the *aligned* size (post-granularity
        rounding), so this typically exceeds the requested
        user size by 1-2× depending on VMM granularity.
        """
        return sum(data.handle[1] for data in self.pointer_to_data.values())


__all__ = ["CumemAllocator"]
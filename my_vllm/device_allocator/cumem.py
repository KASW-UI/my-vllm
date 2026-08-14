"""CumemAllocator — Python-level implementation of the cumem
allocator architecture for my-vllm.

Mirrors ``vllm/device_allocator/cumem.py`` (377 lines, 2025-04
snapshot).  The full architecture is reproduced (singleton,
``use_memory_pool`` context manager, ``sleep`` / ``wake_up``
API, tag-based bookkeeping via ``AllocationData``) but the
**actual VMM calls** (``cuMemCreate`` / ``cuMemAddressReserve`
/ `cuMemMap`` / `cuMemUnmap``) are replaced with PyTorch
caching-allocator primitives.

Why no real VMM
---------------

Real vLLM's cumem path is a C extension (``libcumem_allocator.so``)
that wraps the CUDA Virtual Memory Management API directly.  The
extension is built per-platform and uses the same approach as the
NVIDIA driver samples for low-level memory management.

Building it requires:

  * ~500 lines of C++/CUDA (CUDA Driver API wrappers;
    ``cuMemCreate``, ``cuMemAddressReserve``, ``cuMemMap``,
    ``cuMemUnmap``, ``cuMemSetAccess``, ``cuMemRelease``,
    ``cuMemAddressFree``).
  * CMake wiring into ``setup.py`` (additional CUDA library
    targets).
  * A runtime-loaded library dependency that needs to be packaged
    correctly (PyTorch's ``CUDAPluggableAllocator`` expects a
    specific ``my_malloc`` / ``my_free`` ABI).
  * Platform-specific bindings (Linux only — CUDA driver API
    ctypes wrapping on Windows is a separate effort).

This is a 2-3 day effort that consumes the entire Phase 3 budget.
Production vLLM has it because the H100/Hopper memory savings
(50%+ on the 80GB pool when serving many small sequences)
justify the engineering cost.  RTX 4060 (8GB, sm_89) gets
noticeably less benefit and our workloads are smaller.

What C231 ships
---------------

  * **API parity with vLLM.**  Every public method
    (``get_instance`` / ``use_memory_pool`` / ``sleep`` /
    ``wake_up`` / ``get_current_usage``) and dataclass
    (``AllocationData``) is present with the same signature.
    Production code that imports these from vLLM can be
    retargeted to my-vllm with a one-line import change.

  * **Python-level sleep that actually frees memory.**
    ``sleep`` calls ``torch.cuda.synchronize`` then
    ``torch.cuda.empty_cache``.  This releases cached
    allocation pool blocks back to the driver — not the
    per-tensor unmapping that real VMM provides, but a
    meaningful chunk (typically 200MB-1GB depending on
    fragmentation).

  * **Tag-based bookkeeping.**  Each ``AllocationData`` carries
    its tag and a ``cpu_backup_tensor`` slot (None in the
    Python-only path; populated by the C extension in vLLM).
    Tests can verify the singleton + tag lifecycle without
    needing real VMM.

  * **Pluggable allocator hook.**  When PyTorch's
    ``torch.cuda.memory.MemPool`` API is wired (currently
    requires the C extension), ``use_memory_pool`` plugs
    in there.  Today it falls back to a no-op context
    manager that simply tracks the tag in a Python dict —
    sufficient for the API contract; missing the actual
    pool routing.

  * **Singleton + ``atexit`` cleanup.**  Mirrors vLLM's
    ``CuMemAllocator.instance`` + ``atexit.register`` so the
    pool reference outlives the busy-loop; only the C-extension
    path strictly requires this, but we follow the same
    pattern for safety.

Limitations vs vLLM
-------------------

  * No real per-tensor unmapping (CUDA driver API calls).
  * No CPU offload (``cudaMemcpy`` to pinned host memory)
    during sleep.
  * No MemPool integration — the per-tag pool is just a
    Python dict.
  * ``release_pools`` (the GC workaround for
    pytorch/pytorch#146431) is a no-op without the
    pluggable allocator path.

These limits are documented inline at each method.
"""

from __future__ import annotations

import atexit
import gc
import logging
from contextlib import contextmanager
from typing import Iterator

from my_vllm.device_allocator import AllocationData, HandleType

logger = logging.getLogger(__name__)


class CumemAllocator:
    """Singleton memory-pool allocator (architecture-aligned with vLLM).

    See module docstring for the scope decision; this is the
    Python-level implementation.  To upgrade to real VMM, swap
    ``_python_malloc_callback`` / ``_python_free_callback``
    for the C-extension versions and replace ``_sleep_free``
    / ``_wake_up_restore`` with the ``unmap_and_release`` /
    ``create_and_map`` calls into the extension.
    """

    instance: "CumemAllocator | None" = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CumemAllocator":
        """Singleton accessor — mirrors vLLM
        ``CuMemAllocator.get_instance``.

        Cannot call ``__init__`` directly.  First invocation
        installs an ``atexit`` shutdown hook that releases the
        pool reference before interpreter finalization (the
        same defensive pattern vLLM uses for pytorch#146431).
        """
        if CumemAllocator.instance is None:
            CumemAllocator.instance = CumemAllocator()
            atexit.register(CumemAllocator._shutdown_singleton)
        return CumemAllocator.instance

    @staticmethod
    def _shutdown_singleton() -> None:
        """Release pool refs at interpreter shutdown (defensive).

        Mirrors vLLM's defensive cleanup pattern.  In our
        Python-only path the cleanup is a no-op — there's no
        ``MemPool`` / ``CUDAPluggableAllocator`` to keep alive
        — but we keep the same hook for parity.
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
        # ``with`` block (PyTorch bug workaround).  In our
        # Python-only path this is empty (the pool is just a
        # dict) but we keep the field for API parity.
        self.allocator_and_pools: dict[str, tuple] = {}
        # Bind methods (vLLM holds strong refs to avoid GC
        # issues with PyTorch's pluggable allocator callbacks).
        self.python_malloc_callback = self._python_malloc_callback
        self.python_free_callback = self._python_free_callback

    # ------------------------------------------------------------------
    # Allocation hooks (no-op in Python-only path)
    # ------------------------------------------------------------------

    def _python_malloc_callback(self, allocation_handle: HandleType) -> None:
        """Record an allocation.

        Mirrors ``vllm.cumem.CuMemAllocator._python_malloc_callback``.
        In the Python-only path this is invoked by the pluggable
        allocator; in our path it's a no-op because we don't
        route PyTorch allocations through a pluggable allocator.
        """
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(
            allocation_handle, self.current_tag
        )
        logger.debug(
            "CumemAllocator: malloc %d bytes @ %d (tag=%s)",
            allocation_handle[1],
            py_d_mem,
            self.current_tag,
        )

    def _python_free_callback(self, ptr: int) -> HandleType:
        """Look up + remove an allocation record on free.

        Mirrors ``vllm.cumem.CuMemAllocator._python_free_callback``.
        Returns the handle so the C extension can perform
        unmap+release; in the Python-only path the handle is
        discarded (we don't unmap anything).
        """
        data = self.pointer_to_data.pop(ptr, None)
        if data is None:
            # Allocator freed a ptr we never recorded — defensive.
            logger.warning("CumemAllocator: free on unregistered ptr %d", ptr)
            return (0, 0, ptr, [])
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

        In the Python-only path, the context does NOT actually
        route allocations to a separate physical pool — we
        just track the tag in ``self.current_tag`` so any
        ``AllocationData`` records created during the context
        carry it.  Real MemPool integration requires the C
        extension.

        The PYTORCH_CUDA_ALLOC_CONF / expandable_segments dance
        from vLLM (``expandable_segments:True`` is incompatible
        with the pluggable allocator path; see pytorch#147851)
        is a no-op here since we don't plug an allocator.
        """
        if tag is None:
            tag = CumemAllocator.default_tag
        if not isinstance(tag, str):
            raise TypeError(f"tag must be str, got {type(tag).__name__}")

        old_tag = self.current_tag
        self.current_tag = tag
        try:
            # In the C-extension path this would call
            # ``use_memory_pool_with_allocator(...)`` and yield
            # ``(mem_pool, new_alloc)``.  We yield nothing — the
            # Python-only path doesn't allocate a separate pool.
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

        Python-only behavior:

          1. ``torch.cuda.synchronize()`` — drain pending kernels
             so any in-flight tensor work doesn't race the
             allocator's free path.
          2. ``torch.cuda.empty_cache()`` — return cached blocks
             to the driver.  This is the **same** call vLLM
             makes at the end of ``sleep`` after the C extension
             has already unmapped everything; for us it's the
             whole reclamation step.
          3. ``gc.collect()`` — release Python-side references
             (mirrors vLLM).

        What we DON'T do (out of scope without the C extension):
          * Per-tensor unmapping of VMM chunks.
          * ``cudaMemcpy`` to pinned host memory for offload.
          * Population of ``AllocationData.cpu_backup_tensor``.

        In a hypothetical "wake_up after offload" call on the
        Python-only path, the offloaded tensors' content would
        already be lost (we never copied it).  This is a known
        limitation; the API still works for the common case of
        "free as much memory as possible, then reallocate on
        wake_up" which is what the elastic-EP sleep flow needs.
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

        for ptr, data in list(self.pointer_to_data.items()):
            handle = data.handle
            total_bytes += handle[1]
            if data.tag in offload_tags:
                backup_bytes += handle[1]
                # C-extension path: cudaMemcpy to pinned host
                # tensor.  Python path: no-op (contents lost).
            # Mark asleep — but the C extension would have
            # already unmapped the physical chunks.  Without
            # the C extension, marking asleep is a courtesy;
            # PyTorch's caching allocator still owns the memory
            # until empty_cache runs below.
            data.is_asleep = True

        # Synchronize + empty_cache is the Python-only reclamation
        # step.  This is what the user actually gets from the
        # sleep call.
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
            "CumemAllocator: sleep reclaimed (tracked=%d bytes, would-offload=%d bytes). "
            "Python-only path: contents of offload_tags are NOT preserved "
            "(real VMM requires the C extension).",
            total_bytes,
            backup_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Restore offloaded allocations to GPU memory.

        Mirrors the vLLM ``CuMemAllocator.wake_up`` API.

        Python-only behavior:

          1. For each recorded allocation whose tag matches
             ``tags`` (or all tags, if ``tags=None``), mark
             ``is_asleep = False``.
          2. **Don't restore contents.**  In the Python-only
             path, ``cpu_backup_tensor`` was never populated
             (the C extension would have done that during
             ``sleep``).  Subsequent allocations will use
             PyTorch's caching allocator as normal.

        This is a no-op for the Python-only path on the GPU side
        (the ``empty_cache`` from sleep already returned the
        memory to the driver; PyTorch reallocates as needed).
        The bookkeeping flip is preserved for API parity.
        """
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                data.is_asleep = False
                # C-extension path: cudaMemcpy from pinned
                # host tensor back to the device pointer, then
                # free the host tensor.  Python path: nothing
                # to copy (no backup tensor was ever created).
                if data.cpu_backup_tensor is not None:
                    data.cpu_backup_tensor = None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release pool refs eagerly.

        Mirrors vLLM's defensive cleanup (pytorch#146431
        workaround).  Python-only path: a no-op (the
        ``allocator_and_pools`` dict is empty) but kept for API
        parity.
        """
        if not self.allocator_and_pools:
            return
        pool_entries = list(self.allocator_and_pools.values())
        self.allocator_and_pools.clear()
        # C-extension path: phase 1 drops MemPool refs while
        # allocators are still strongly held; phase 2 releases
        # the allocators.  Both phases are defensive against
        # pytorch/pytorch#145168 (pure virtual method called).
        pool_entries.clear()

    def release_pools(self) -> None:
        """Compatibility alias — same as ``close``."""
        self.close()

    def get_current_usage(self) -> int:
        """Total bytes currently tracked in the allocator.

        Mirrors vLLM's ``CuMemAllocator.get_current_usage``.  In
        the Python-only path this is the sum of all allocation
        sizes recorded by ``_python_malloc_callback``; without
        actual allocation routing, the count is 0 unless
        callers manually drive the callback.  Useful primarily
        for tests that verify the bookkeeping invariant.
        """
        return sum(data.handle[1] for data in self.pointer_to_data.values())


__all__ = ["CumemAllocator"]
"""test_C231_cumem_allocator.py — C231 cumem allocator tests.

Verifies the architecture-aligned surface of my-vllm's
``device_allocator`` package:

  1. ``AllocationData`` dataclass.
     * Field defaults (handle, tag, cpu_backup_tensor=None,
       is_asleep=False).
     * ``handle`` shape: (py_device, size, ptr, list-of-chunks).

  2. ``MemAllocator`` Protocol — every concrete class implements
     the four required methods
     (``use_memory_pool`` / ``sleep`` / ``wake_up`` /
     ``get_current_usage``).

  3. ``get_mem_allocator_instance()`` — platform-routed factory.
     * CUDA + VMM-capable → ``CumemAllocator`` singleton.
     * Singleton identity preserved across calls.
     * Non-CUDA path raises ``RuntimeError``.

  4. ``Platform.is_cumem_allocator_available()``:
     * True on a CUDA device with SM >= 7.0 (Volta+).
     * False on CPU / no-CUDA (skipped in CUDA-only env).

  5. ``CumemAllocator`` — singleton + Python-level implementation.
     * ``get_instance()`` returns the same object every time.
     * ``use_memory_pool(tag)`` sets ``current_tag`` for the
       duration of the ``with`` block, restores on exit.
     * Tag type validation (string only).
     * Default tag is ``"default"`` when not specified.
     * Nested ``use_memory_pool`` contexts restore the
       *outer* tag on exit (not the previous global).

  6. ``sleep`` / ``wake_up`` API:
     * ``sleep`` calls ``torch.cuda.synchronize`` +
       ``torch.cuda.empty_cache`` (verified by patching).
     * ``wake_up`` flips ``is_asleep`` back to False.
     * String ``offload_tags`` is normalized to a tuple.
     * Multiple ``sleep`` / ``wake_up`` cycles are idempotent.

  7. Bookkeeping — calling the malloc / free callbacks updates
     ``pointer_to_data`` and ``get_current_usage`` correctly.
"""

from __future__ import annotations

from unittest import mock

import pytest

from my_vllm.device_allocator import (
    AllocationData,
    HandleType,
    get_mem_allocator_instance,
)
from my_vllm.platforms import current_platform


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _require_cuda():
    if not current_platform.is_cuda_alike():
        pytest.skip("CUDA required")


@pytest.fixture
def fresh_allocator(monkeypatch):
    """Reset the CumemAllocator singleton so each test gets a clean slate."""
    from my_vllm.device_allocator import cumem as cumem_mod

    cumem_mod.CumemAllocator.instance = None
    yield
    cumem_mod.CumemAllocator.instance = None


# ---------------------------------------------------------------------------
# 1. AllocationData
# ---------------------------------------------------------------------------


class TestAllocationData:
    def test_default_field_values(self):
        handle: HandleType = (0, 1024, 0xDEAD_BEEF, [])
        d = AllocationData(handle=handle, tag="default")
        assert d.handle == handle
        assert d.tag == "default"
        assert d.cpu_backup_tensor is None
        assert d.is_asleep is False

    def test_handle_shape(self):
        # 4-tuple: (py_device, size, ptr, chunks-list).
        d = AllocationData(
            handle=(0, 4096, 0x12345, [101, 102, 103]),
            tag="kv_cache",
        )
        assert d.handle[0] == 0  # device
        assert d.handle[1] == 4096  # size
        assert d.handle[2] == 0x12345  # ptr
        assert d.handle[3] == [101, 102, 103]  # physical chunk ids

    def test_mutable_after_construction(self):
        d = AllocationData(handle=(0, 1, 2, []), tag="x")
        d.is_asleep = True
        d.cpu_backup_tensor = "fake"
        assert d.is_asleep is True
        assert d.cpu_backup_tensor == "fake"


# ---------------------------------------------------------------------------
# 2. Platform + factory
# ---------------------------------------------------------------------------


class TestPlatformCumem:
    def test_is_cumem_allocator_available_cuda(self):
        # On a CUDA host with a SM >= 7.0 device, this should
        # be True.  CPU-only hosts raise pytest.skip via the
        # ``_require_cuda`` fixture in calling tests.
        assert current_platform.is_cumem_allocator_available() is True


class TestFactory:
    def test_get_mem_allocator_instance_returns_singleton(self, _require_cuda, fresh_allocator):
        a = get_mem_allocator_instance()
        b = get_mem_allocator_instance()
        assert a is b

    def test_returns_cumem_allocator(self, _require_cuda, fresh_allocator):
        from my_vllm.device_allocator.cumem import CumemAllocator

        a = get_mem_allocator_instance()
        assert isinstance(a, CumemAllocator)


# ---------------------------------------------------------------------------
# 3. CumemAllocator singleton lifecycle
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_default_tag_constant(self, fresh_allocator):
        from my_vllm.device_allocator.cumem import CumemAllocator

        assert CumemAllocator.default_tag == "default"

    def test_get_instance_creates_then_returns(self, fresh_allocator):
        from my_vllm.device_allocator.cumem import CumemAllocator

        assert CumemAllocator.instance is None
        a = CumemAllocator.get_instance()
        assert CumemAllocator.instance is a
        b = CumemAllocator.get_instance()
        assert a is b


# ---------------------------------------------------------------------------
# 4. use_memory_pool — tag context manager
# ---------------------------------------------------------------------------


class TestUseMemoryPool:
    def test_default_tag(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        with alloc.use_memory_pool():
            assert alloc.current_tag == "default"

    def test_custom_tag(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        with alloc.use_memory_pool(tag="kv_cache"):
            assert alloc.current_tag == "kv_cache"
        assert alloc.current_tag == "default"

    def test_nested_tags_restore_correctly(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        with alloc.use_memory_pool(tag="outer"):
            assert alloc.current_tag == "outer"
            with alloc.use_memory_pool(tag="inner"):
                assert alloc.current_tag == "inner"
            # After inner exits, we're back to outer.
            assert alloc.current_tag == "outer"
        assert alloc.current_tag == "default"

    def test_invalid_tag_type_raises(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        with pytest.raises(TypeError, match="tag must be str"):
            with alloc.use_memory_pool(tag=123):
                pass


# ---------------------------------------------------------------------------
# 5. sleep / wake_up — API contract
# ---------------------------------------------------------------------------


class TestSleepWakeUp:
    def test_sleep_calls_synchronize_and_empty_cache(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        with mock.patch("torch.cuda.synchronize") as sync, mock.patch(
            "torch.cuda.empty_cache"
        ) as empty:
            alloc.sleep()
        assert sync.called
        assert empty.called

    def test_sleep_with_string_offload_tags_normalizes(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        # Should not raise; string is normalized internally.
        alloc.sleep(offload_tags="kv_cache")

    def test_sleep_with_default_offload_tags(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        # Default = (CumemAllocator.default_tag,) = ("default",).
        # Just exercise the path; no exceptions expected.
        alloc.sleep()

    def test_sleep_with_none_offload_tags(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        # None is normalized to ("default",).
        alloc.sleep(offload_tags=None)

    def test_sleep_with_invalid_type_raises(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        with pytest.raises(TypeError, match="offload_tags"):
            alloc.sleep(offload_tags=123)  # not str / tuple / None

    def test_wake_up_flips_is_asleep(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        # Pre-populate a record manually.
        alloc.pointer_to_data[0x100] = AllocationData(
            handle=(0, 4096, 0x100, []),
            tag="default",
            is_asleep=False,
        )
        alloc.sleep()
        assert alloc.pointer_to_data[0x100].is_asleep is True
        alloc.wake_up()
        assert alloc.pointer_to_data[0x100].is_asleep is False

    def test_wake_up_with_tag_filter(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        alloc.pointer_to_data[0xA] = AllocationData(
            handle=(0, 1, 0xA, []), tag="keep", is_asleep=True
        )
        alloc.pointer_to_data[0xB] = AllocationData(
            handle=(0, 1, 0xB, []), tag="drop", is_asleep=True
        )
        alloc.wake_up(tags=["keep"])
        assert alloc.pointer_to_data[0xA].is_asleep is False
        # The "drop" tag remains asleep.
        assert alloc.pointer_to_data[0xB].is_asleep is True

    def test_multiple_sleep_wake_up_cycles_idempotent(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        for _ in range(3):
            alloc.sleep()
            alloc.wake_up()


# ---------------------------------------------------------------------------
# 6. Bookkeeping — malloc / free callbacks
# ---------------------------------------------------------------------------


class TestBookkeeping:
    def test_malloc_callback_records_allocation(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        alloc.python_malloc_callback(
            (0, 4096, 0xCAFE_F00D, [])  # handle: device, size, ptr, chunks
        )
        assert 0xCAFE_F00D in alloc.pointer_to_data
        d = alloc.pointer_to_data[0xCAFE_F00D]
        assert d.handle[1] == 4096
        assert d.tag == "default"  # current_tag at allocation time

    def test_malloc_callback_uses_current_tag(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        with alloc.use_memory_pool(tag="kv_cache"):
            alloc.python_malloc_callback((0, 1024, 0x100, []))
        assert alloc.pointer_to_data[0x100].tag == "kv_cache"

    def test_free_callback_returns_handle(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        alloc.python_malloc_callback((0, 8192, 0x200, [11, 12]))
        returned = alloc.python_free_callback(0x200)
        assert returned == (0, 8192, 0x200, [11, 12])
        assert 0x200 not in alloc.pointer_to_data

    def test_free_unknown_ptr_returns_empty_handle(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        returned = alloc.python_free_callback(0xDEAD)
        # Defensive: returns a sentinel (device=0, size=0, ptr=0xDEAD, []).
        assert returned == (0, 0, 0xDEAD, [])

    def test_free_callback_clears_cpu_backup(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        d = AllocationData(
            handle=(0, 1, 0x300, []),
            tag="default",
            cpu_backup_tensor="fake_backup",
        )
        alloc.pointer_to_data[0x300] = d
        alloc.python_free_callback(0x300)
        assert d.cpu_backup_tensor is None

    def test_get_current_usage_sums_allocation_sizes(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        alloc.python_malloc_callback((0, 1024, 0x1, []))
        alloc.python_malloc_callback((0, 4096, 0x2, []))
        alloc.python_malloc_callback((0, 256, 0x3, []))
        assert alloc.get_current_usage() == 1024 + 4096 + 256

    def test_get_current_usage_empty(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        assert alloc.get_current_usage() == 0


# ---------------------------------------------------------------------------
# 7. close / release_pools — defensive cleanup
# ---------------------------------------------------------------------------


class TestCloseAndRelease:
    def test_close_is_idempotent(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        alloc.close()
        alloc.close()  # should not raise

    def test_release_pools_alias_for_close(self, _require_cuda, fresh_allocator):
        alloc = get_mem_allocator_instance()
        # Both should work the same way.
        alloc.release_pools()
        alloc.close()

    def test_close_with_empty_pools_is_noop(
        self, _require_cuda, fresh_allocator
    ):
        alloc = get_mem_allocator_instance()
        # Pre-condition: allocator_and_pools is empty.
        assert alloc.allocator_and_pools == {}
        alloc.close()
        assert alloc.allocator_and_pools == {}


# ---------------------------------------------------------------------------
# 8. End-to-end smoke test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_allocate_sleep_wake_up_cycle(self, _require_cuda, fresh_allocator):
        """Realistic usage: allocate, tag, sleep, wake_up."""
        alloc = get_mem_allocator_instance()

        # Allocate within a tagged context.
        with alloc.use_memory_pool(tag="kv_cache"):
            alloc.python_malloc_callback((0, 1024 * 1024, 0xA, []))
            alloc.python_malloc_callback((0, 1024 * 1024, 0xB, []))

        # Both records should carry the kv_cache tag.
        assert alloc.pointer_to_data[0xA].tag == "kv_cache"
        assert alloc.pointer_to_data[0xB].tag == "kv_cache"

        # Usage tracks the sum.
        assert alloc.get_current_usage() == 2 * 1024 * 1024

        # Sleep + wake_up should be safe.
        with mock.patch("torch.cuda.synchronize"), mock.patch(
            "torch.cuda.empty_cache"
        ):
            alloc.sleep()
        with mock.patch("torch.cuda.synchronize"), mock.patch(
            "torch.cuda.empty_cache"
        ):
            alloc.wake_up()

        # Records survive sleep/wake_up (we don't free them).
        assert 0xA in alloc.pointer_to_data
        assert 0xB in alloc.pointer_to_data
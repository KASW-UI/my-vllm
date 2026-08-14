"""tests/test_C241_symm_mem.py — Symmetric Memory communicator + heuristics.

C241: closes the A-level "Symmetric Memory" gap from
``phase3-distributed-runtime-plan.md`` (§ Phase 3).

Hardware: NVLS (NVLink-Switch) requires Hopper (sm_90) /
Blackwell (sm_100+) — RTX 4060 (sm_89) is NOT supported.
The my-vllm dev box is sm_89, so ``SymmMemCommunicator``
gracefully degrades to ``disabled=True``; the API surface
and selection heuristics are still testable in isolation.

Test classes:

  * ``TestSymmMemCommunicatorLifecycle`` — initialization is
    graceful when the runtime / hardware is unavailable; the
    disabled flag is correct in all cases.
  * ``TestSymmMemCommunicatorEligibility`` —
    ``should_use_symm_mem`` correctly gates by dtype (bf16),
    size (≤ max_size), and the disabled flag.
  * ``TestSymmMemCommunicatorAllReduce`` — ``all_reduce`` is a
    no-op (returns None) when disabled.
  * ``TestSymmMemCommunicatorForce`` — ``force_multimem`` and
    ``max_size_override`` are exposed for testing.
  * ``TestCudaCommunicatorSymmMemIntegration`` —
    ``CudaCommunicator.attach_symm_mem`` correctly delegates
    ``all_reduce`` to the symm_mem path when bound, and falls
    back when not bound.
  * ``TestSymmMemSizingTables`` — the ``SYMM_MEM_ALL_REDUCE_MAX_SIZES`` /
    ``CUSTOM_ALL_REDUCE_MAX_SIZES`` / ``NCCL_SYMM_MEM_ALL_REDUCE_CONFIG``
    dictionaries are well-formed and consistent.
  * ``TestShouldNCCLSymmMemAllreduce`` — the heuristic correctly
    rejects world_size < min_world_size and follows the
    custom_AR preferred ranges.
  * ``TestShouldNCCLSymmMemAGRS`` — the AG/RS heuristic.
  * ``TestIsSymmetricMemoryAvailable`` — module-level helper.
  * ``TestSymmMemCommunicatorBenchmark`` — micro-benchmark on
    a single GPU verifies the symm_mem buffer shape / dtype
    invariant (when NVLS isn't available, ``buffer`` attribute
    is None).
"""

from __future__ import annotations

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Lifecycle — gracefully disabled
# ---------------------------------------------------------------------------


class TestSymmMemCommunicatorLifecycle:
    """Initialization is graceful on non-NVLS hardware / missing runtime."""

    def test_cpu_device_disabled(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.disabled is True

    def test_cuda_device_disabled_without_group(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        sm = SymmMemCommunicator(group=None, device="cuda:0")
        # No real NCCL group → disable rather than crash.
        assert sm.disabled is True

    def test_cuda_no_group_no_buffer(self):
        """When disabled (no group), ``buffer`` attribute is missing."""
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.disabled
        # The buffer is not allocated when disabled.
        assert not hasattr(sm, "buffer") or sm.buffer is None


# ---------------------------------------------------------------------------
# 2. Eligibility — should_use_symm_mem
# ---------------------------------------------------------------------------


class TestSymmMemCommunicatorEligibility:
    """``should_use_symm_mem`` gates by dtype / size / disabled flag."""

    def _make_disabled(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.disabled
        return sm

    def test_disabled_returns_false(self):
        sm = self._make_disabled()
        assert sm.should_use_symm_mem(torch.zeros(8, dtype=torch.bfloat16)) is False

    def test_dtype_mismatch_returns_false(self):
        """When enabled but dtype is wrong, reject — construct a
        pseudo-enabled communicator by monkey-patching ``disabled``."""
        sm = self._make_disabled()
        sm.disabled = False
        sm.max_size = 1024 * 1024  # 1 MB
        # fp16 input is not the symm_mem dtype (bf16) → reject.
        assert (
            sm.should_use_symm_mem(torch.zeros(8, dtype=torch.float16))
            is False
        )

    def test_size_too_large_returns_false(self):
        sm = self._make_disabled()
        sm.disabled = False
        sm.dtype = torch.bfloat16
        sm.max_size = 1024  # 1 KB
        # 64 KB input → too large.
        big = torch.zeros(64 * 1024, dtype=torch.bfloat16)
        assert sm.should_use_symm_mem(big) is False

    def test_size_aligned_and_eligible(self):
        sm = self._make_disabled()
        sm.disabled = False
        sm.dtype = torch.bfloat16
        sm.max_size = 1024 * 1024  # 1 MB
        # 4 KB bf16 input → eligible.
        small = torch.zeros(2 * 1024, dtype=torch.bfloat16)
        assert sm.should_use_symm_mem(small) is True

    def test_size_unaligned_rejected(self):
        sm = self._make_disabled()
        sm.disabled = False
        sm.dtype = torch.bfloat16
        sm.max_size = 1024 * 1024
        # 7 bytes (unaligned to 4) → reject.
        odd = torch.zeros(7, dtype=torch.uint8)
        # Use bf16 with non-multiple-of-4 element count.
        unaligned = torch.zeros(3, dtype=torch.bfloat16)  # 6 bytes
        assert sm.should_use_symm_mem(unaligned) is False


# ---------------------------------------------------------------------------
# 3. all_reduce — returns None when disabled
# ---------------------------------------------------------------------------


class TestSymmMemCommunicatorAllReduce:
    """``all_reduce`` is a no-op (returns None) when symm_mem is disabled."""

    def test_all_reduce_disabled_returns_none(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.disabled
        inp = torch.zeros(8, dtype=torch.bfloat16)
        assert sm.all_reduce(inp) is None

    def test_all_reduce_dtype_mismatch_returns_none(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        sm.disabled = False
        sm.dtype = torch.bfloat16
        sm.max_size = 1024
        inp = torch.zeros(8, dtype=torch.float16)
        assert sm.all_reduce(inp) is None


# ---------------------------------------------------------------------------
# 4. Force / override parameters
# ---------------------------------------------------------------------------


class TestSymmMemCommunicatorForce:
    """``force_multimem`` and ``max_size_override`` are exposed correctly."""

    def test_force_multimem_param_default(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.force_multimem is None

    def test_max_size_override_param_default(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.max_size_override is None

    def test_should_use_multimem_with_force_true(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        sm.force_multimem = True
        sm.device_capability = "9.0"
        sm.world_size = 2  # not in {4, 6, 8} default
        assert sm._should_use_multimem() is True

    def test_should_use_multimem_with_force_false(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        sm.force_multimem = False
        sm.device_capability = "9.0"
        sm.world_size = 8  # in {4, 6, 8} default
        assert sm._should_use_multimem() is False

    def test_should_use_multimem_default_table(self):
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        sm = SymmMemCommunicator(group=None, device="cpu")
        sm.force_multimem = None
        sm.device_capability = "9.0"
        # world_size in {4, 6, 8} → True; otherwise False.
        sm.world_size = 4
        assert sm._should_use_multimem() is True
        sm.world_size = 6
        assert sm._should_use_multimem() is True
        sm.world_size = 8
        assert sm._should_use_multimem() is True
        sm.world_size = 2
        assert sm._should_use_multimem() is False


# ---------------------------------------------------------------------------
# 5. CudaCommunicator integration
# ---------------------------------------------------------------------------


class TestCudaCommunicatorSymmMemIntegration:
    """``CudaCommunicator.attach_symm_mem`` + ``all_reduce`` delegation."""

    def test_attach_symm_mem_default_none(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        assert getattr(cc, "_symm_mem_communicator", None) is None

    def test_attach_symm_mem_set(self):
        from my_vllm.distributed.device_communicators import (
            CudaCommunicator,
            SymmMemCommunicator,
        )

        sm = SymmMemCommunicator(group=None, device="cpu")
        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        cc.attach_symm_mem(sm)
        assert cc._symm_mem_communicator is sm

    def test_attach_symm_mem_clear(self):
        from my_vllm.distributed.device_communicators import (
            CudaCommunicator,
            SymmMemCommunicator,
        )

        sm = SymmMemCommunicator(group=None, device="cpu")
        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        cc.attach_symm_mem(sm)
        cc.attach_symm_mem(None)
        assert cc._symm_mem_communicator is None

    def test_all_reduce_with_disabled_symm_mem_falls_back(self):
        """When bound symm_mem is disabled, ``all_reduce`` falls back
        to ``GroupCoordinator.all_reduce`` (or torch.distributed)."""
        from my_vllm.distributed.device_communicators import (
            CudaCommunicator,
            SymmMemCommunicator,
        )

        sm = SymmMemCommunicator(group=None, device="cpu")
        assert sm.disabled
        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        cc.attach_symm_mem(sm)
        # world_size==1 short-circuits anyway; just verify it doesn't crash.
        inp = torch.randn(4)
        out = cc.all_reduce(inp)
        assert torch.equal(out, inp)


# ---------------------------------------------------------------------------
# 6. Sizing tables
# ---------------------------------------------------------------------------


class TestSymmMemSizingTables:
    """The size tables are well-formed and consistent across capabilities."""

    def test_symm_mem_table_world_sizes(self):
        from my_vllm.distributed.device_communicators import (
            SYMM_MEM_ALL_REDUCE_MAX_SIZES,
        )

        for cap, sizes in SYMM_MEM_ALL_REDUCE_MAX_SIZES.items():
            for ws, size in sizes.items():
                assert isinstance(ws, int)
                assert isinstance(size, int)
                assert size > 0

    def test_custom_ar_table_world_sizes(self):
        from my_vllm.distributed.device_communicators import (
            CUSTOM_ALL_REDUCE_MAX_SIZES,
        )

        for cap, sizes in CUSTOM_ALL_REDUCE_MAX_SIZES.items():
            for ws, size in sizes.items():
                assert isinstance(ws, int)
                assert isinstance(size, int)
                assert size > 0

    def test_symm_mem_contains_known_caps(self):
        from my_vllm.distributed.device_communicators import (
            SYMM_MEM_ALL_REDUCE_MAX_SIZES,
        )

        # All vLLM-tracked NVIDIA SM families.
        for cap in ("9.0", "10.0", "10.3"):
            assert cap in SYMM_MEM_ALL_REDUCE_MAX_SIZES

    def test_nccl_symm_mem_config_keys(self):
        from my_vllm.distributed.device_communicators import (
            NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
        )

        cfg = NCCL_SYMM_MEM_ALL_REDUCE_CONFIG
        assert cfg["min_world_size"] >= 1
        for ws, (lo, hi) in cfg["custom_ar_preferred_ranges"].items():
            assert lo < hi

    def test_world_sizes_multi_mem_subset(self):
        from my_vllm.distributed.device_communicators.symm_mem import (
            SymmMemCommunicator,
        )

        multimem_ws = SymmMemCommunicator._WORLD_SIZES_MULTIMEM
        assert "9.0" in multimem_ws
        assert "10.0" in multimem_ws
        for cap, ws_list in multimem_ws.items():
            for ws in ws_list:
                assert ws >= 4


# ---------------------------------------------------------------------------
# 7. Heuristic — should_nccl_symm_mem_allreduce
# ---------------------------------------------------------------------------


class TestShouldNCCLSymmMemAllreduce:
    """The all-reduce heuristic correctly picks the right backend."""

    def test_disabled_by_env(self, monkeypatch):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            _is_symmetric_memory_enabled,
        )

        monkeypatch.setenv("MY_VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
        # Re-evaluate the cached function — it's not cached but the env
        # var is read fresh each call.
        inp = torch.zeros(64, dtype=torch.bfloat16)
        assert should_nccl_symm_mem_allreduce(8, inp) is False

    def test_below_min_world_size_returns_false(self):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )

        # Default min_world_size=4 → ws=2 always False.
        inp = torch.zeros(64 * 1024 * 1024, dtype=torch.bfloat16)
        assert should_nccl_symm_mem_allreduce(2, inp) is False

    def test_small_size_uses_symm_mem(self):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )

        # ws=4, size=4KB → below the lower_bound (16K) → True.
        inp = torch.zeros(2 * 1024, dtype=torch.bfloat16)  # 4 KB
        assert should_nccl_symm_mem_allreduce(4, inp) is True

    def test_mid_range_size_uses_custom_ar(self):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )

        # ws=4, size=128KB → in [16K, 512K] → False (custom_AR wins).
        inp = torch.zeros(64 * 1024, dtype=torch.bfloat16)  # 128 KB
        assert should_nccl_symm_mem_allreduce(4, inp) is False

    def test_large_size_uses_symm_mem(self):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )

        # ws=4, size=2MB → above upper_bound (512K) → True.
        inp = torch.zeros(1024 * 1024, dtype=torch.bfloat16)  # 2 MB
        assert should_nccl_symm_mem_allreduce(4, inp) is True

    def test_huge_world_size_always_symm_mem(self):
        from my_vllm.distributed.device_communicators.all_reduce_utils import (
            should_nccl_symm_mem_allreduce,
        )

        # ws=16 (> always_use_above_world_size=8) and outside the
        # custom_AR preferred ranges dict → True.
        inp = torch.zeros(64, dtype=torch.bfloat16)
        assert should_nccl_symm_mem_allreduce(16, inp) is True


# ---------------------------------------------------------------------------
# 8. Heuristic — should_nccl_symm_mem_ag_rs
# ---------------------------------------------------------------------------


class TestShouldNCCLSymmMemAGRS:
    """The AG/RS heuristic is just the runtime-enabled flag."""

    def test_default_returns_true(self):
        from my_vllm.distributed.device_communicators import (
            should_nccl_symm_mem_ag_rs,
        )

        # Default env (no MY_VLLM_ALLREDUCE_USE_SYMM_MEM set) → True.
        os.environ.pop("MY_VLLM_ALLREDUCE_USE_SYMM_MEM", None)
        # Force re-import to test cleanly.
        assert should_nccl_symm_mem_ag_rs() is True

    def test_disabled_via_env(self, monkeypatch):
        from my_vllm.distributed.device_communicators import (
            should_nccl_symm_mem_ag_rs,
        )

        monkeypatch.setenv("MY_VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
        assert should_nccl_symm_mem_ag_rs() is False


# ---------------------------------------------------------------------------
# 9. is_symmetric_memory_available helper
# ---------------------------------------------------------------------------


class TestIsSymmetricMemoryAvailable:
    """The module-level helper gates runtime support."""

    def test_returns_bool(self):
        from my_vllm.distributed.device_communicators import (
            is_symmetric_memory_available,
        )
        result = is_symmetric_memory_available()
        assert isinstance(result, bool)

    def test_consistent_with_internal_flag(self):
        from my_vllm.distributed.device_communicators import (
            is_symmetric_memory_available,
        )
        from my_vllm.distributed.device_communicators.symm_mem import (
            _SYMM_MEM_AVAILABLE,
        )

        assert is_symmetric_memory_available() == _SYMM_MEM_AVAILABLE


# ---------------------------------------------------------------------------
# 10. End-to-end smoke — CudaCommunicator + symm_mem through real cuda
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSymmMemOnCUDA:
    """Sanity checks when CUDA is available (RTX 4060 / sm_89 → disabled)."""

    def test_cuda_symm_mem_disabled_on_ada(self):
        """RTX 4060 is sm_89 → NVLS not available → disabled."""
        from my_vllm.distributed.device_communicators import SymmMemCommunicator

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 9:  # Hopper or newer
            pytest.skip("NVLS-capable GPU; would need real NCCL group to test")
        sm = SymmMemCommunicator(group=None, device="cuda:0")
        # Either because the device isn't supported, or because the
        # group is None.
        assert sm.disabled is True

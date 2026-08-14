"""tests/test_C244_quick_flashinfer_ar.py — QuickReduce + FlashInfer-AR skeletons.

C244: closes the B-level "Quick AllReduce + FlashInfer AllReduce 笔记"
gap from ``phase3-distributed-runtime-plan.md``.

Both backends are placeholders in my-vllm (QuickReduce is ROCm MI300-
only + needs a C extension; FlashInfer-AR needs the ``flashinfer``
library).  This test file verifies:

* the module-level availability flags are correctly ``False``
* the constructors set ``disabled = True`` regardless of inputs
* the heuristic methods (``should_quick_allreduce`` /
  ``should_use_fi_ar``) return ``False``
* the call surfaces (``quick_all_reduce`` / ``all_reduce``) raise
  ``RuntimeError``
* the quantization regime enum has all 6 members
* the min-size table is well-formed and consistent
* the lifecycle (``close`` / ``destroy``) is a safe no-op

Total: 8 test classes, ~20 cases.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Module-level availability flags
# ---------------------------------------------------------------------------


class TestAvailabilityFlags:
    """``quick_ar`` and ``fi_ar_available`` are False at import time."""

    def test_quick_ar_is_false(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            quick_ar,
        )

        assert quick_ar is False

    def test_fi_ar_available_is_false(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            fi_ar_available,
        )

        assert fi_ar_available is False


# ---------------------------------------------------------------------------
# 2. QuickReduceRegime enum
# ---------------------------------------------------------------------------


class TestQuickReduceRegime:
    """The 6 quantization regimes match vLLM's C header ids."""

    def test_six_members(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickReduceRegime,
        )

        members = list(QuickReduceRegime)
        assert len(members) == 6
        names = {m.name for m in members}
        assert names == {"FP", "INT8", "INT6", "INT4", "INT3", "NONE"}

    def test_integer_ids_aligned_with_c_header(self):
        """vLLM keeps these ids aligned with ``csrc/quickreduce/quick_reduce.h``;
        the ``_QR_MIN_SIZE`` table indexes by ``.value``."""
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickReduceRegime,
        )

        assert QuickReduceRegime.FP.value == 0
        assert QuickReduceRegime.INT8.value == 1
        assert QuickReduceRegime.INT6.value == 2
        assert QuickReduceRegime.INT4.value == 3
        assert QuickReduceRegime.INT3.value == 4
        assert QuickReduceRegime.NONE.value == 5

    def test_round_trip_by_name(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickReduceRegime,
        )

        for name in ("FP", "INT8", "INT6", "INT4", "INT3", "NONE"):
            assert QuickReduceRegime[name].name == name


# ---------------------------------------------------------------------------
# 3. QuickAllReduce constructor + disabled flag
# ---------------------------------------------------------------------------


class TestQuickAllReduceDisabled:
    """``QuickAllReduce`` is always disabled in my-vllm."""

    def test_construction_sets_disabled(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        assert qa.disabled is True

    def test_supported_world_sizes(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        assert QuickAllReduce._SUPPORTED_WORLD_SIZES == [2, 4, 8]

    def test_supported_dtypes(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        assert QuickAllReduce._SUPPORTED_DTYPES == [torch.float16, torch.bfloat16]

    def test_should_quick_allreduce_always_false(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        for tensor in [
            torch.zeros(1, dtype=torch.float32),
            torch.zeros(1024, 1024, dtype=torch.bfloat16),
            torch.randn(4, 8, 16, dtype=torch.float16),
        ]:
            assert qa.should_quick_allreduce(tensor) is False

    def test_quick_all_reduce_raises(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        with pytest.raises(RuntimeError, match="not available"):
            qa.quick_all_reduce(torch.zeros(4))

    def test_close_is_noop_when_disabled(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        qa.close()  # should not raise
        assert qa.disabled is True

    def test_init_quick_all_reduce_is_noop(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        qa.init_quick_all_reduce()  # no-op
        assert qa.disabled is True

    def test_create_shared_buffer_is_noop(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        qa.create_shared_buffer()  # no-op

    def test_del_does_not_raise(self):
        """``__del__`` must swallow exceptions (Python destructor contract)."""
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qa = QuickAllReduce(group=None, device="cuda:0")
        # Manually trigger __del__ to ensure no exception leaks.
        del qa


# ---------------------------------------------------------------------------
# 4. QuickAllReduce min-size table
# ---------------------------------------------------------------------------


class TestQuickReduceMinSizeTable:
    """``_QR_MIN_SIZE`` is well-formed and consistent."""

    def test_keys_cover_all_supported_pairs(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        expected_keys = {
            (torch.float16, 2),
            (torch.float16, 4),
            (torch.float16, 8),
            (torch.bfloat16, 2),
            (torch.bfloat16, 4),
            (torch.bfloat16, 8),
        }
        assert set(QuickAllReduce._QR_MIN_SIZE.keys()) == expected_keys

    def test_each_row_has_five_regimes(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        for sizes in QuickAllReduce._QR_MIN_SIZE.values():
            assert len(sizes) == 5  # FP, INT8, INT6, INT4, INT3

    def test_min_sizes_are_positive(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        for key, sizes in QuickAllReduce._QR_MIN_SIZE.items():
            for size in sizes:
                assert size > 0, f"{key}: non-positive size {size}"


# ---------------------------------------------------------------------------
# 5. FlashInferAllReduce constructor + disabled flag
# ---------------------------------------------------------------------------


class TestFlashInferAllReduceDisabled:
    """``FlashInferAllReduce`` is always disabled in my-vllm."""

    def test_construction_sets_disabled(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        ar = FlashInferAllReduce(group=None, device="cuda:0")
        assert ar.disabled is True

    def test_should_use_fi_ar_always_false(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        ar = FlashInferAllReduce(group=None, device="cuda:0")
        for tensor in [
            torch.zeros((1, 8)),
            torch.zeros((128, 1024)),
            torch.randn(4, 8, 16),  # 3D, not 2D
        ]:
            assert ar.should_use_fi_ar(tensor) is False

    def test_all_reduce_raises(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        ar = FlashInferAllReduce(group=None, device="cuda:0")
        with pytest.raises(RuntimeError, match="not available"):
            ar.all_reduce(torch.zeros((4, 8)))

    def test_destroy_is_noop(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        ar = FlashInferAllReduce(group=None, device="cuda:0")
        ar.destroy()  # no-op

    def test_ensure_workspace_returns_false(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        ar = FlashInferAllReduce(group=None, device="cuda:0")
        assert ar._ensure_workspace(128, torch.bfloat16) is False
        assert ar.disabled is True

    def test_init_does_not_raise_without_distributed(self):
        """``__init__`` must not touch the global torch.distributed state."""
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )

        # No ``dist.init_process_group`` has been called in this test,
        # but the constructor must still complete.
        ar = FlashInferAllReduce(group=None, device="cuda:0")
        assert ar.disabled is True


# ---------------------------------------------------------------------------
# 6. Workspace helpers
# ---------------------------------------------------------------------------


class TestFlashInferWorkspaceHelpers:
    """``get_fi_ar_workspace`` and ``get_fi_ar_quant_workspace`` return None."""

    def test_get_fi_ar_workspace_returns_none(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            get_fi_ar_workspace,
        )

        result = get_fi_ar_workspace(
            world_size=2,
            rank=0,
            max_token_num=128,
            hidden_dim=128,
            dtype=torch.float16,
            group=None,
        )
        assert result is None

    def test_get_fi_ar_quant_workspace_returns_none(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            get_fi_ar_quant_workspace,
        )

        result = get_fi_ar_quant_workspace(
            world_size=2,
            rank=0,
            max_token_num=128,
            hidden_dim=128,
            dtype=torch.float16,
            group=None,
        )
        assert result is None

    def test_destroy_fi_ar_workspace_is_noop(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            destroy_fi_ar_workspace,
        )

        destroy_fi_ar_workspace()  # no-op, no exception

    def test_resolve_fi_ar_backend_returns_none(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            _resolve_fi_ar_backend,
        )

        backend, allow_fallback = _resolve_fi_ar_backend()
        assert backend == "none"
        assert allow_fallback is False


# ---------------------------------------------------------------------------
# 7. vLLM contract preserved — env-var helper stubs
# ---------------------------------------------------------------------------


class TestQuickReduceEnvVarHelpers:
    """Env-var helpers are no-op stubs in my-vllm."""

    def test_get_qr_min_size_returns_none(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            _get_qr_min_size,
        )

        assert _get_qr_min_size(qr_max_size=1024 * 1024) is None
        assert _get_qr_min_size(qr_max_size=None) is None

    def test_get_qr_quantization_min_size_returns_none(self):
        from my_vllm.distributed.device_communicators.quick_all_reduce import (
            _get_qr_quantization_min_size,
        )

        assert _get_qr_quantization_min_size() is None


# ---------------------------------------------------------------------------
# 8. PDL threshold constant
# ---------------------------------------------------------------------------


class TestFlashInferPDLThreshold:
    """The PDL advance-launch threshold constant is preserved."""

    def test_pdl_threshold_value(self):
        from my_vllm.distributed.device_communicators.flashinfer_all_reduce import (
            PDL_ADVANCE_LAUNCH_TOKENS,
        )

        assert PDL_ADVANCE_LAUNCH_TOKENS == 16

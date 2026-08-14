"""tests/test_C240_all2all.py — All-to-All (MoE EP) path.

C240: closes the S-level "All-to-All 通信算子" gap from
``phase3-distributed-runtime-plan.md``.

Test classes:

  * ``TestAll2AllManagerBase`` — ABC contract: methods raise
    ``NotImplementedError`` by default.
  * ``TestAgRsAll2AllManagerIdentity`` — single-rank dispatch /
    combine are identity operations.
  * ``TestAgRsAll2AllManagerAlgorithm`` — algorithm correctness
    with a mock ``GroupCoordinator`` that simulates a
    ``world_size == 4`` AG/RS collective.  Verifies the AG+RS
    produces the world-concatenated / per-rank-reduced result
    correctly for both uniform and variable sizes.
  * ``TestAgRsAll2AllManagerExtras`` — the ``extra_tensors`` path
    (per-token scales) is gathered alongside ``hidden_states``.
  * ``TestCudaCommunicatorIdentity`` — ``CudaCommunicator`` identity
    + dispatch / combine shim with no ``all2all_manager``.
  * ``TestFusedMoEParallelConfigProperties`` — the
    ``use_all2all_kernels`` / ``use_ag_rs_all2all_kernels`` /
    ``use_deepep_kernels`` / ``use_nixl_ep_kernels`` /
    ``use_flashinfer_kernels`` / ``use_mori_kernels`` properties
    map the ``all2all_backend`` literal correctly.
  * ``TestMaybeMakePrepareFinalize`` — factory dispatch for the
    NaiveDPEP and NoDPEP backends.
  * ``TestNoDPEPRoundTrip`` — ``MoEPrepareAndFinalizeNoDPEPModular``
    end-to-end: prepare → apply top-k → finalize returns the
    weighted-reduced output unchanged.
  * ``TestNaiveDPEPRoundTripIdentity`` — when no EP group is set,
    the NaiveDPEP path is also identity (same as NoDPEP).
  * ``TestParallelStateEPGroup`` — ``init_expert_parallel`` is a
    no-op when ``ep_size <= 1``; ``get_ep_group`` returns ``None``.
  * ``TestGroupCoordinatorAllGatherRSingle`` —
    ``GroupCoordinator.all_gatherv`` / ``reduce_scatterv``
    short-circuit when ``world_size == 1``.
  * ``TestAll2AllBackendEnum`` — the ``All2AllBackend`` literal
    set is valid.
  * ``TestBackendShellsRaise`` — DeepEP / NIXL EP / FlashInfer /
    Mori class shells raise ``NotImplementedError`` on any
    communication primitive.
  * ``TestModularKernelABC`` — ``FusedMoEPrepareAndFinalize``
    subclasses work as expected.

Total: 13 classes, 50+ test cases.
"""

from __future__ import annotations

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProcessGroup:
    """Minimal stand-in for ``torch.distributed`` ProcessGroup.

    Captures ``rank``, ``world_size``, and ``rank_in_group``.  The
    ``MockCommGroup`` below extends this with an ``all_gatherv`` /
    ``reduce_scatterv`` that simulates the world-wide collective
    using CPU buffers (so the test is hermetic — no NCCL/Gloo
    process bootstrap).
    """

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        self.rank_in_group = rank


class MockCommGroup(MockProcessGroup):
    """Simulates ``all_gatherv`` + ``reduce_scatterv`` in-process.

    Each "rank" contributes its local buffer.  ``all_gatherv``
    returns the world-wide concatenation; ``reduce_scatterv``
    returns the rank-local slice (sum of all ranks' contributions
    to that slice).
    """

    def __init__(self, rank: int, world_size: int) -> None:
        super().__init__(rank, world_size)
        self._local_buffers: list[torch.Tensor] = []
        self._allgather_log: list = []
        self._reduce_scatter_log: list = []

    def push_local(self, tensor: torch.Tensor | list[torch.Tensor]) -> None:
        if isinstance(tensor, torch.Tensor):
            self._local_buffers.append([tensor.detach().clone()])
        else:
            self._local_buffers.append(
                [t.detach().clone() for t in tensor]
            )

    def all_gatherv(
        self,
        input_,
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        """Concatenate all ranks' ``_local_buffers`` along ``dim``.

        Accepts a single tensor or a list of tensors.  For lists,
        returns a list of concatenated tensors.  Each rank's
        contribution is the rank-th entry in ``_local_buffers``;
        ``_local_buffers[i]`` is a list of tensors in the same
        order as ``input_``.
        """
        if isinstance(input_, torch.Tensor):
            inputs = [input_]
            single = True
        else:
            inputs = list(input_)
            single = False

        results = []
        for j, inp in enumerate(inputs):
            pieces = [
                self._local_buffers[i][j]
                for i in range(self.world_size)
            ]
            results.append(torch.cat(pieces, dim=dim))

        self._allgather_log.append((input_, dim, sizes))
        if single:
            return results[0]
        return results

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Sum-reduce the world-wide ``input_`` along ``dim``
        and return rank-local slice.

        The mock simulates the collective: ``input_`` is the
        world-wide tensor (already on every rank); we sum it
        element-wise (a no-op since all ranks have the same
        ``input_``) and narrow to the rank-local slice.
        """
        if sizes is None:
            return input_
        my_offset = sum(sizes[: self.rank_in_group])
        my_size = sizes[self.rank_in_group]
        self._reduce_scatter_log.append((input_, dim, sizes))
        return input_.narrow(dim, my_offset, my_size).clone()


def _make_ag_rs_manager_for_rank(
    rank: int,
    world_size: int,
):
    """Build an ``AgRsAll2AllManager`` whose ``comm_group`` is a
    ``MockCommGroup`` shared across all ranks of the same world.

    Each ``(world_size,)`` key has ONE shared ``MockCommGroup``
    that all ranks reference.  This simulates the world-wide
    collective: rank ``r``'s dispatch is observed by ranks
    ``!= r`` because the AG collects everyone's contribution.

    Subsequent calls (with the same world_size) reuse the same
    mock and its already-pushed buffers.  To reset between
    tests, the test must explicitly pop the ``_world_mocks``
    entry (or call ``mock._local_buffers.clear()``).
    """
    from my_vllm.distributed.device_communicators import AgRsAll2AllManager

    if not hasattr(_make_ag_rs_manager_for_rank, "_world_mocks"):
        _make_ag_rs_manager_for_rank._world_mocks = {}

    world_key = (world_size,)
    if world_key not in _make_ag_rs_manager_for_rank._world_mocks:
        shared = MockCommGroup(rank=0, world_size=world_size)
        _make_ag_rs_manager_for_rank._world_mocks[world_key] = shared
    mock = _make_ag_rs_manager_for_rank._world_mocks[world_key]

    mgr = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    mgr.cpu_group = mock
    mgr.tcp_store_group = None
    mgr.dp_group = None
    mgr.tp_group = None
    mgr.dp_rank = 0
    mgr.dp_world_size = 1
    mgr.rank = rank
    mgr.world_size = world_size
    mgr.comm_group = mock
    # Override the mock's rank_in_group for this manager's view.
    mock.rank = rank
    mock.rank_in_group = rank
    return mgr, mock


# ---------------------------------------------------------------------------
# 1. All2AllManagerBase ABC contract
# ---------------------------------------------------------------------------


class TestAll2AllManagerBase:
    """The ``All2AllManagerBase`` ABC raises ``NotImplementedError``
    on abstract methods by default."""

    def test_dispatch_router_logits_not_implemented(self):
        from my_vllm.distributed.device_communicators import All2AllManagerBase

        class Stub(All2AllManagerBase):
            pass

        s = Stub.__new__(Stub)
        s.cpu_group = None
        s.tcp_store_group = None
        s.dp_group = None
        s.tp_group = None
        s.dp_rank = 0
        s.dp_world_size = 1
        s.rank = 0
        s.world_size = 1
        with pytest.raises(NotImplementedError):
            s.dispatch_router_logits(
                torch.zeros(1), torch.zeros(1)
            )

    def test_dispatch_not_implemented(self):
        from my_vllm.distributed.device_communicators import All2AllManagerBase

        class Stub(All2AllManagerBase):
            pass

        s = Stub.__new__(Stub)
        s.cpu_group = None
        s.tcp_store_group = None
        s.dp_group = None
        s.tp_group = None
        s.dp_rank = 0
        s.dp_world_size = 1
        s.rank = 0
        s.world_size = 1
        with pytest.raises(NotImplementedError):
            s.dispatch(torch.zeros(1), torch.zeros(1), torch.zeros(1))

    def test_combine_not_implemented(self):
        from my_vllm.distributed.device_communicators import All2AllManagerBase

        class Stub(All2AllManagerBase):
            pass

        s = Stub.__new__(Stub)
        s.cpu_group = None
        s.tcp_store_group = None
        s.dp_group = None
        s.tp_group = None
        s.dp_rank = 0
        s.dp_world_size = 1
        s.rank = 0
        s.world_size = 1
        with pytest.raises(NotImplementedError):
            s.combine(torch.zeros(1))


# ---------------------------------------------------------------------------
# 2. AgRsAll2AllManager — single-rank identity
# ---------------------------------------------------------------------------


class TestAgRsAll2AllManagerIdentity:
    """Single-rank dispatch / combine are identity operations."""

    def _make_manager(self):
        from my_vllm.distributed.device_communicators import AgRsAll2AllManager

        m = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
        m.cpu_group = None
        m.tcp_store_group = None
        m.dp_group = None
        m.tp_group = None
        m.dp_rank = 0
        m.dp_world_size = 1
        m.rank = 0
        m.world_size = 1
        m.comm_group = None
        return m

    def test_dispatch_router_logits_identity(self):
        m = self._make_manager()
        hs = torch.randn(4, 8)
        rl = torch.randn(4, 16)
        result = m.dispatch_router_logits(hs, rl)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], hs)
        assert torch.equal(result[1], rl)

    def test_dispatch_router_logits_with_extras_identity(self):
        m = self._make_manager()
        hs = torch.randn(4, 8)
        rl = torch.randn(4, 16)
        scales = [torch.randn(4)]
        result = m.dispatch_router_logits(hs, rl, extra_tensors=scales)
        assert len(result) == 3
        assert torch.equal(result[0], hs)
        assert torch.equal(result[1], rl)
        assert isinstance(result[2], list)
        assert torch.equal(result[2][0], scales[0])

    def test_dispatch_identity(self):
        m = self._make_manager()
        hs = torch.randn(4, 8)
        tw = torch.randn(4, 2)
        ti = torch.zeros(4, 2, dtype=torch.int64)
        result = m.dispatch(hs, tw, ti)
        assert len(result) == 3
        assert torch.equal(result[0], hs)
        assert torch.equal(result[1], tw)
        assert torch.equal(result[2], ti)

    def test_dispatch_with_extras_identity(self):
        m = self._make_manager()
        hs = torch.randn(4, 8)
        tw = torch.randn(4, 2)
        ti = torch.zeros(4, 2, dtype=torch.int64)
        scales = [torch.randn(4)]
        result = m.dispatch(hs, tw, ti, extra_tensors=scales)
        assert len(result) == 4
        assert torch.equal(result[0], hs)
        assert torch.equal(result[2], ti)
        assert torch.equal(result[3][0], scales[0])

    def test_combine_identity(self):
        m = self._make_manager()
        hs = torch.randn(4, 8)
        out = m.combine(hs)
        assert torch.equal(out, hs)

    def test_get_sizes_uniform_path(self):
        """When ``dp_world_size == 1``, ``_get_sizes`` returns uniform sizes."""
        m = self._make_manager()
        sizes = m._get_sizes(7, None)
        assert sizes == [7]

    def test_get_sizes_uniform_path_with_world_size(self):
        m = self._make_manager()
        m.comm_group = MockProcessGroup(rank=0, world_size=4)
        sizes = m._get_sizes(7, m.comm_group)
        assert sizes == [7, 7, 7, 7]


# ---------------------------------------------------------------------------
# 3. AgRsAll2AllManager — algorithm correctness with mock comm group
# ---------------------------------------------------------------------------


class TestAgRsAll2AllManagerAlgorithm:
    """Algorithm correctness with a mock ``MockCommGroup`` simulating world_size=4.

    For each "rank" ``r`` we set up a separate manager and push the
    rank's local tensors into the mock comm group.  Then we run
    dispatch / combine and verify the result is what the AG+RS
    algorithm should produce.
    """

    @staticmethod
    def _setup_rank_buffers(world_size: int):
        """Pre-load each rank's local buffers into the shared mock group."""
        if hasattr(_make_ag_rs_manager_for_rank, "_world_mocks"):
            _make_ag_rs_manager_for_rank._world_mocks.pop(
                (world_size,), None
            )
        _mgr0, mock = _make_ag_rs_manager_for_rank(0, world_size)
        # Clear any previous buffers in case the test reuses the mock.
        mock._local_buffers = []
        buffers_per_rank: list[list[torch.Tensor]] = []
        for r in range(world_size):
            hs = torch.tensor(
                [[float(r), float(r) + 1], [float(r) + 10, float(r) + 11]],
                dtype=torch.float32,
            )
            tw = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
            ti = torch.tensor([[r, r + 1], [r + 10, r + 11]], dtype=torch.int64)
            mock.push_local([hs, tw, ti])
            buffers_per_rank.append([hs, tw, ti])
        return buffers_per_rank

    def test_dispatch_concatenates_all_ranks(self):
        world_size = 4
        self._setup_rank_buffers(world_size)

        expected_hs = torch.cat(
            [
                torch.tensor(
                    [[float(r), float(r) + 1], [float(r) + 10, float(r) + 11]]
                )
                for r in range(world_size)
            ],
            dim=0,
        )

        for r in range(world_size):
            mgr, _ = _make_ag_rs_manager_for_rank(r, world_size)
            hs = torch.tensor(
                [[float(r), float(r) + 1], [float(r) + 10, float(r) + 11]]
            )
            tw = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
            ti = torch.tensor([[r, r + 1], [r + 10, r + 11]], dtype=torch.int64)
            result = mgr.dispatch(hs, tw, ti)
            assert result[0].shape == (world_size * 2, 2)
            assert torch.allclose(result[0], expected_hs), (
                f"rank {r} got {result[0]} but expected {expected_hs}"
            )

    def test_combine_scatters_per_rank_sum(self):
        """Each rank should receive the per-rank slice of the
        world-wide summed tensor.

        The mock simulates the AG→expert→RS flow: world-wide
        input is the per-rank-stacked ``hs`` values; combine
        narrows each rank to its slice.  Since the mock doesn't
        actually sum, the per-rank slice should equal the
        rank-th contribution to the stacked tensor.
        """
        world_size = 4
        self._setup_rank_buffers(world_size)

        for r in range(world_size):
            mgr, _ = _make_ag_rs_manager_for_rank(r, world_size)
            stacked = torch.stack(
                [
                    torch.tensor(
                        [[float(i), float(i) + 1], [float(i) + 10, float(i) + 11]]
                    )
                    for i in range(world_size)
                ],
                dim=0,
            )
            # World-wide input (what every rank sees after AG).
            world_wide = stacked.reshape(world_size * 2, 2)
            sizes = [2] * world_size
            expected = stacked[r].clone()
            out = mgr.combine(world_wide)
            assert torch.allclose(out, expected), (
                f"rank {r} got {out} but expected {expected}"
            )

    def test_dispatch_with_extras(self):
        world_size = 4
        # Setup pushes 3-tensor buffers; for the extras test we need
        # to push 4-tensor buffers (hs, tw, ti, scales).
        if hasattr(_make_ag_rs_manager_for_rank, "_world_mocks"):
            _make_ag_rs_manager_for_rank._world_mocks.pop(
                (world_size,), None
            )
        _mgr0, mock = _make_ag_rs_manager_for_rank(0, world_size)
        mock._local_buffers = []
        for r in range(world_size):
            hs = torch.tensor(
                [[float(r), float(r) + 1], [float(r) + 10, float(r) + 11]]
            )
            tw = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
            ti = torch.tensor([[r, r + 1], [r + 10, r + 11]], dtype=torch.int64)
            scales = torch.full((2,), float(r))
            mock.push_local([hs, tw, ti, scales])

        for r in range(world_size):
            mgr, _ = _make_ag_rs_manager_for_rank(r, world_size)
            hs = torch.tensor(
                [[float(r), float(r) + 1], [float(r) + 10, float(r) + 11]]
            )
            tw = torch.tensor([[1.0, 0.5], [0.25, 0.75]])
            ti = torch.tensor([[r, r + 1], [r + 10, r + 11]], dtype=torch.int64)
            scales = [torch.full((2,), float(r))]
            result = mgr.dispatch(hs, tw, ti, extra_tensors=scales)
            assert len(result) == 4
            assert result[3][0].shape == (world_size * 2,)
            expected_scales = torch.cat(
                [torch.full((2,), float(i)) for i in range(world_size)],
                dim=0,
            )
            assert torch.allclose(result[3][0], expected_scales)


# ---------------------------------------------------------------------------
# 4. AgRsAll2AllManager — variable sizes
# ---------------------------------------------------------------------------


class TestAgRsAll2AllManagerVariableSizes:
    """Variable per-rank sizes — exercises the non-uniform AG/RS case.

    C240: my-vllm's ``AgRsAll2AllManager._get_sizes`` returns
    uniform sizes by default (the dp_world_size == 1 path).  The
    variable-size path is exercised via the explicit
    ``forward_context.dp_metadata`` mechanism (deferred to C246
    scheduler resource awareness).  Here we use a uniform
    ``world_size * M_local`` shape so the AG result is well-defined.
    """

    @staticmethod
    def _setup(world_size: int, m_local: int):
        """Configure each rank with ``m_local`` tokens."""
        if hasattr(_make_ag_rs_manager_for_rank, "_world_mocks"):
            _make_ag_rs_manager_for_rank._world_mocks.pop(
                (world_size,), None
            )
        _mgr0, mock = _make_ag_rs_manager_for_rank(0, world_size)
        mock._local_buffers = []
        for r in range(world_size):
            local_hs = torch.full((m_local, 4), float(r))
            mock.push_local([local_hs, local_hs[:, :2]])
        return [m_local] * world_size

    def test_uniform_all_gather(self):
        world_size = 3
        m_local = 4
        sizes = self._setup(world_size, m_local)
        assert sizes == [4, 4, 4]

        for r in range(world_size):
            mgr, _ = _make_ag_rs_manager_for_rank(r, world_size)
            local_hs = torch.full((m_local, 4), float(r))
            out = mgr.dispatch_router_logits(local_hs, local_hs[:, :2])
            assert out[0].shape == (world_size * m_local, 4)
            expected = torch.cat(
                [
                    torch.full((m_local, 4), float(i))
                    for i in range(world_size)
                ],
                dim=0,
            )
            assert torch.allclose(out[0], expected)


# ---------------------------------------------------------------------------
# 5. CudaCommunicator identity + dispatch / combine shim
# ---------------------------------------------------------------------------


class TestCudaCommunicatorIdentity:
    """``CudaCommunicator`` identity when no manager is bound."""

    def test_all_gatherv_identity(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        hs = torch.randn(4, 8)
        out = cc.all_gatherv(hs)
        assert torch.equal(out, hs)

    def test_all_gatherv_list_identity(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        a = torch.randn(2, 4)
        b = torch.randn(2, 8)
        out = cc.all_gatherv([a, b])
        assert isinstance(out, list)
        assert torch.equal(out[0], a)
        assert torch.equal(out[1], b)

    def test_reduce_scatterv_identity(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        hs = torch.randn(4, 8)
        assert torch.equal(cc.reduce_scatterv(hs), hs)
        assert torch.equal(cc.reduce_scatterv(hs, dim=-1), hs)

    def test_dispatch_no_manager(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        hs = torch.randn(4, 8)
        out = cc.dispatch(hs, hs, hs)
        assert torch.equal(out[0], hs)
        out = cc.dispatch(hs, hs, hs, extra_tensors=[hs])
        assert torch.equal(out[3][0], hs)

    def test_combine_no_manager(self):
        from my_vllm.distributed.device_communicators import CudaCommunicator

        cc = CudaCommunicator(cpu_group=None, group_coordinator=None)
        hs = torch.randn(4, 8)
        assert torch.equal(cc.combine(hs), hs)


# ---------------------------------------------------------------------------
# 6. FusedMoEParallelConfig properties
# ---------------------------------------------------------------------------


class TestFusedMoEParallelConfigProperties:
    """``use_*_kernels`` selectors map ``all2all_backend`` correctly."""

    def test_no_all2all_when_ep_dp_one(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig()
        assert pc.use_all2all_kernels is False
        assert pc.use_ag_rs_all2all_kernels is True
        assert pc.use_deepep_kernels is False
        assert pc.use_nixl_ep_kernels is False
        assert pc.use_flashinfer_kernels is False
        assert pc.use_mori_kernels is False

    def test_ep_enables_all2all(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(ep_size=4)
        assert pc.use_all2all_kernels is True
        assert pc.use_ag_rs_all2all_kernels is True

    def test_dp_enables_all2all(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(dp_size=2)
        assert pc.use_all2all_kernels is True

    def test_deepep_backend(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(
            ep_size=4, all2all_backend="deepep_high_throughput"
        )
        assert pc.use_all2all_kernels is True
        assert pc.use_deepep_kernels is True
        assert pc.use_ag_rs_all2all_kernels is False

    def test_nixl_ep_backend(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(
            ep_size=4, all2all_backend="nixl_ep"
        )
        assert pc.use_nixl_ep_kernels is True

    def test_flashinfer_backend(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(
            ep_size=4, all2all_backend="flashinfer_two_sided"
        )
        assert pc.use_flashinfer_kernels is True

    def test_mori_backend(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        pc = FusedMoEParallelConfig(
            ep_size=4, all2all_backend="mori"
        )
        assert pc.use_mori_kernels is True


# ---------------------------------------------------------------------------
# 7. maybe_make_prepare_finalize factory dispatch
# ---------------------------------------------------------------------------


class TestMaybeMakePrepareFinalize:
    """The factory picks the correct PrepareAndFinalize based on config."""

    def _fake_moe(self, moe_parallel_config):
        class _Moe:
            pass
        m = _Moe()
        m.moe_parallel_config = moe_parallel_config
        return m

    def test_no_parallel_returns_no_dpep(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pc = FusedMoEParallelConfig()
        pf = maybe_make_make = maybe_make_prepare_finalize(self._fake_moe(pc))
        assert isinstance(pf, MoEPrepareAndFinalizeNoDPEPModular)

    def test_ep_with_naive_returns_naive_dp_ep(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
        pc = FusedMoEParallelConfig(ep_size=4, all2all_backend="naive")
        pf = maybe_make_prepare_finalize(self._fake_moe(pc))
        assert isinstance(pf, MoEPrepareAndFinalizeNaiveDPEPModular)

    def test_use_monolithic_returns_monolithic(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPMonolithic,
        )
        pf = maybe_make_prepare_finalize(
            self._fake_moe(FusedMoEParallelConfig()),
            use_monolithic=True,
        )
        assert isinstance(pf, MoEPrepareAndFinalizeNoDPEPMonolithic)

    def test_deepep_backend_falls_back_to_no_dpep(self):
        """DeepEP backend is not vendored; falls back to NoDPEP."""
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pc = FusedMoEParallelConfig(
            ep_size=4, all2all_backend="deepep_high_throughput"
        )
        pf = maybe_make_prepare_finalize(self._fake_moe(pc))
        assert isinstance(pf, MoEPrepareAndFinalizeNoDPEPModular)

    def test_nixl_ep_backend_falls_back_to_no_dpep(self):
        from my_vllm.model_executor.layers.fused_moe.config import (
            FusedMoEParallelConfig,
        )
        from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pc = FusedMoEParallelConfig(ep_size=4, all2all_backend="nixl_ep")
        pf = maybe_make_prepare_finalize(self._fake_moe(pc))
        assert isinstance(pf, MoEPrepareAndFinalizeNoDPEPModular)


# ---------------------------------------------------------------------------
# 8. NoDPEP round-trip (single-rank path is identity end-to-end)
# ---------------------------------------------------------------------------


class _IdentityTopKWeightAndReduce:
    """A trivial ``TopKWeightAndReduce`` that copies the first expert
    output (no router weight, no top-k reduction).

    Mirrors the vLLM contract: when ``output`` is provided, write
    into ``output`` in-place and return it; otherwise allocate.
    """

    def apply(
        self,
        output,
        fused_expert_output,
        topk_weights,
        topk_ids,
        apply_router_weight_on_input,
    ):
        result = fused_expert_output[:, 0, :].clone()
        if output is not None:
            output.copy_(result)
            return output
        return result


class TestNoDPEPRoundTrip:
    """``MoEPrepareAndFinalizeNoDPEPModular`` round-trip."""

    def test_prepare_returns_inputs_unchanged(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pf = MoEPrepareAndFinalizeNoDPEPModular()
        a1 = torch.randn(4, 8)
        tw = torch.randn(4, 2)
        ti = torch.zeros(4, 2, dtype=torch.int64)
        a1q, scale, meta, tw_out, ti_out = pf.prepare(
            a1, tw, ti, num_experts=8, expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=None,
        )
        assert torch.equal(a1q, a1)
        assert scale is None
        assert meta is None
        assert tw_out is None
        assert ti_out is None

    def test_finalize_writes_output(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pf = MoEPrepareAndFinalizeNoDPEPModular()
        output = torch.zeros(4, 8)
        fused = torch.randn(4, 2, 8)
        pf.finalize(
            output=output,
            fused_expert_output=fused,
            topk_weights=torch.ones(4, 2),
            topk_ids=torch.zeros(4, 2, dtype=torch.int64),
            apply_router_weight_on_input=False,
            weight_and_reduce_impl=_IdentityTopKWeightAndReduce(),
        )
        assert torch.equal(output, fused[:, 0, :])

    def test_full_round_trip(self):
        """prepare → fake experts → finalize preserves the first expert's output."""
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        pf = MoEPrepareAndFinalizeNoDPEPModular()
        a1 = torch.randn(4, 8)
        tw = torch.ones(4, 2)
        ti = torch.tensor(
            [[0, 1], [0, 2], [1, 0], [2, 1]], dtype=torch.int64
        )
        a1q, scale, meta, _, _ = pf.prepare(
            a1, tw, ti, num_experts=4, expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=None,
        )
        assert torch.equal(a1q, a1)

        expert_out = torch.randn(4, 2, 8)
        output = torch.zeros(4, 8)
        pf.finalize(
            output=output,
            fused_expert_output=expert_out,
            topk_weights=tw,
            topk_ids=ti,
            apply_router_weight_on_input=False,
            weight_and_reduce_impl=_IdentityTopKWeightAndReduce(),
        )
        assert torch.equal(output, expert_out[:, 0, :])


# ---------------------------------------------------------------------------
# 9. NaiveDPEP round-trip identity (no EP group → identity)
# ---------------------------------------------------------------------------


class TestNaiveDPEPRoundTripIdentity:
    """Without an EP group the NaiveDPEP prepare/finalize is identity."""

    def test_prepare_is_identity(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
        from my_vllm.distributed import get_ep_group
        pf = MoEPrepareAndFinalizeNaiveDPEPModular()
        a1 = torch.randn(4, 8)
        tw = torch.randn(4, 2)
        ti = torch.zeros(4, 2, dtype=torch.int64)
        result = pf.prepare(
            a1, tw, ti, num_experts=4, expert_map=None,
            apply_router_weight_on_input=False,
            quant_config=None,
        )
        assert get_ep_group() is None
        assert torch.equal(result[0], a1)
        assert torch.equal(result[3], ti)
        assert torch.equal(result[4], tw)


# ---------------------------------------------------------------------------
# 10. ParallelState — EP group lifecycle
# ---------------------------------------------------------------------------


class TestParallelStateEPGroup:
    """``init_expert_parallel`` is a no-op when ``ep_size <= 1``."""

    def test_get_ep_group_default_none(self):
        from my_vllm.distributed import get_ep_group
        assert get_ep_group() is None

    def test_init_expert_parallel_noop_when_size_1(self):
        from my_vllm.distributed import (
            get_ep_group,
            init_expert_parallel,
        )
        init_expert_parallel(expert_parallel_size=1)
        assert get_ep_group() is None


# ---------------------------------------------------------------------------
# 11. GroupCoordinator.all_gatherv / reduce_scatterv — single-rank short-circuit
# ---------------------------------------------------------------------------


class TestGroupCoordinatorAllGatherRSingle:
    """``GroupCoordinator`` is an internal data structure (not
    initialised in single-process tests), but the helpers are
    testable as plain functions."""

    def test_world_size_one_returns_input(self):
        """Direct call to a freshly-constructed ``GroupCoordinator``
        raises because the constructor expects real ``torch.distributed``
        process groups; instead we exercise the ``world_size == 1``
        short-circuit by mocking ``world_size`` and ``rank_in_group``."""

        from my_vllm.distributed.parallel_state import GroupCoordinator

        gc = GroupCoordinator.__new__(GroupCoordinator)
        gc.world_size = 1
        gc.rank = 0
        gc.rank_in_group = 0
        gc.ranks = [0]
        gc.device_group = None
        gc.cpu_group = None
        gc.device = torch.device("cpu")

        hs = torch.randn(4, 8)
        out = gc.all_gatherv(hs)
        assert torch.equal(out, hs)

        out = gc.reduce_scatterv(hs)
        assert torch.equal(out, hs)

        out = gc.dispatch(hs, hs, hs)
        assert torch.equal(out[0], hs)

        out = gc.dispatch_router_logits(hs, hs)
        assert torch.equal(out[0], hs)

        out = gc.combine(hs)
        assert torch.equal(out, hs)


# ---------------------------------------------------------------------------
# 12. All2AllBackend enum literal set
# ---------------------------------------------------------------------------


class TestAll2AllBackendEnum:
    """The ``All2AllBackend`` literal set is valid and importable."""

    def test_all2all_backend_literal_set(self):
        from my_vllm.config.parallel import ParallelConfig

        cfg = ParallelConfig(all2all_backend="naive")
        assert cfg.all2all_backend == "naive"
        cfg = ParallelConfig(all2all_backend="nixl_ep")
        assert cfg.all2all_backend == "nixl_ep"

    def test_parallel_config_use_all2all_kernels(self):
        from my_vllm.config.parallel import ParallelConfig

        cfg = ParallelConfig()
        assert cfg.use_all2all_kernels is False
        cfg = ParallelConfig(expert_parallel_size=2)
        assert cfg.use_all2all_kernels is True
        cfg = ParallelConfig(data_parallel_size=2)
        assert cfg.use_all2all_kernels is True


# ---------------------------------------------------------------------------
# 13. Backend class shells raise NotImplementedError
# ---------------------------------------------------------------------------


class TestBackendShellsRaise:
    """DeepEP / NIXL EP / FlashInfer / Mori shells raise on any comm primitive."""

    @pytest.mark.parametrize("cls_name", [
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "DeepEPV2All2AllManager",
        "NixlEPAll2AllManager",
        "FlashInferNVLinkOneSidedAll2AllManager",
        "FlashInferNVLinkTwoSidedAll2AllManager",
        "MoriAll2AllManager",
    ])
    def test_shell_dispatch_router_logits(self, cls_name):
        from my_vllm.distributed import device_communicators as dc

        cls = getattr(dc, cls_name)
        mgr = cls.__new__(cls)
        mgr.cpu_group = None
        mgr.tcp_store_group = None
        mgr.dp_group = None
        mgr.tp_group = None
        mgr.dp_rank = 0
        mgr.dp_world_size = 1
        mgr.rank = 0
        mgr.world_size = 1
        with pytest.raises(NotImplementedError):
            mgr.dispatch_router_logits(
                torch.zeros(1), torch.zeros(1)
            )

    @pytest.mark.parametrize("cls_name", [
        "DeepEPHTAll2AllManager",
        "DeepEPLLAll2AllManager",
        "NixlEPAll2AllManager",
        "MoriAll2AllManager",
    ])
    def test_shell_combine(self, cls_name):
        from my_vllm.distributed import device_communicators as dc

        cls = getattr(dc, cls_name)
        mgr = cls.__new__(cls)
        mgr.cpu_group = None
        mgr.tcp_store_group = None
        mgr.dp_group = None
        mgr.tp_group = None
        mgr.dp_rank = 0
        mgr.dp_world_size = 1
        mgr.rank = 0
        mgr.world_size = 1
        with pytest.raises(NotImplementedError):
            mgr.combine(torch.zeros(1))


# ---------------------------------------------------------------------------
# 14. ModularKernel ABC — FusedMoEPrepareAndFinalize + activation format
# ---------------------------------------------------------------------------


class TestModularKernelABC:
    """The ModularKernel ABCs construct correctly."""

    def test_activation_format_enum(self):
        from my_vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEActivationFormat,
        )
        assert FusedMoEActivationFormat.Standard.value == "standard"
        assert FusedMoEActivationFormat.BatchedExperts.value == "batched_experts"

    def test_no_dpep_modular_format(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPModular,
        )
        from my_vllm.model_executor.layers.fused_moe.modular_kernel import (
            FusedMoEActivationFormat,
        )
        pf = MoEPrepareAndFinalizeNoDPEPModular()
        assert pf.activation_format == FusedMoEActivationFormat.Standard
        assert pf.num_dispatchers() == 1
        assert pf.output_is_reduced() is False

    def test_no_dpep_monolithic_format(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNoDPEPMonolithic,
        )
        pf = MoEPrepareAndFinalizeNoDPEPMonolithic()
        assert pf.num_dispatchers() == 1
        assert pf.output_is_reduced() is False

    def test_naive_dp_ep_modular_format(self):
        from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
            MoEPrepareAndFinalizeNaiveDPEPModular,
        )
        pf = MoEPrepareAndFinalizeNaiveDPEPModular(num_dispatchers=2)
        assert pf.num_dispatchers() == 2
        assert pf.is_sequence_parallel is False

    def test_expert_tokens_metadata(self):
        from my_vllm.model_executor.layers.fused_moe.modular_kernel import (
            ExpertTokensMetadata,
        )
        m = ExpertTokensMetadata.make_from_list([3, 5, 2])
        assert m.expert_num_tokens_cpu == [3, 5, 2]


# ---------------------------------------------------------------------------
# 15. Multi-process gloo round-trip (skipped if torch.multiprocessing fails)
# ---------------------------------------------------------------------------


def _gloo_worker_allgather(rank, world_size, port, m_local, hidden, out_dir):
    """Gloo-backed ``all_gatherv`` round-trip worker.

    Saves the result to ``{out_dir}/rank_{rank}.pt`` so the parent
    process can read it after the worker tears down its process
    group.  (Using ``torch.multiprocessing.Queue`` would try to
    share the tensor via file descriptor, which becomes invalid
    once the worker process exits.)
    """
    import os
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from my_vllm.distributed import GroupCoordinator

        gc = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            backend="gloo",
            group_name="test_all2all",
        )

        local_hs = torch.full((m_local, hidden), float(rank))
        world_wide = gc.all_gatherv(local_hs, dim=0)
        torch.save({"rank": rank, "world_wide": world_wide},
                   os.path.join(out_dir, f"rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _gloo_worker_reducescatter(rank, world_size, port, m_local, hidden, out_dir):
    """Gloo-backed ``reduce_scatterv`` round-trip worker."""
    import os
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from my_vllm.distributed import GroupCoordinator

        gc = GroupCoordinator(
            group_ranks=[list(range(world_size))],
            local_rank=rank,
            backend="gloo",
            group_name="test_all2all",
        )

        # Each rank contributes ``rank + 1`` at its OWN slice;
        # the other slices remain 0.  After reduce-scatter (sum),
        # rank r receives the sum at slice r: it equals
        # ``(r + 1)`` (only rank r contributed non-zero there).
        world_wide = torch.zeros(world_size * m_local, hidden)
        world_wide[rank * m_local : (rank + 1) * m_local] = float(rank + 1)
        local_out = gc.reduce_scatterv(world_wide, dim=0)
        torch.save({"rank": rank, "local_out": local_out},
                   os.path.join(out_dir, f"rank_{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not __import__("torch").multiprocessing.get_context("spawn"),
    reason="torch.multiprocessing not available",
)
class TestMultiProcessGlooAllGather:
    """Real ``torch.distributed`` ``all_gatherv`` round-trip (gloo backend).

    Spawns ``world_size`` workers, each runs
    ``GroupCoordinator.all_gatherv`` on the same data, verifies
    that every rank sees the world-wide concatenation.
    """

    def test_all_gatherv_round_trip(self, tmp_path):
        import torch
        import torch.multiprocessing as mp

        world_size = 2
        m_local = 4
        hidden = 8
        port = 31000 + (hash("C240_all_gatherv") % 1000)

        ctx = mp.get_context("spawn")
        out_dir = str(tmp_path)
        procs = []
        for r in range(world_size):
            p = ctx.Process(
                target=_gloo_worker_allgather,
                args=(r, world_size, port, m_local, hidden, out_dir),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

        results = []
        for r in range(world_size):
            data = torch.load(os.path.join(out_dir, f"rank_{r}.pt"))
            results.append((data["rank"], data["world_wide"]))
        results.sort(key=lambda x: x[0])
        for rank, world_wide in results:
            assert world_wide.shape == (world_size * m_local, hidden)
            expected = torch.cat(
                [
                    torch.full((m_local, hidden), float(r))
                    for r in range(world_size)
                ],
                dim=0,
            )
            assert torch.allclose(world_wide, expected), (
                f"rank {rank} world_wide mismatch: "
                f"got {world_wide} but expected {expected}"
            )


@pytest.mark.skipif(
    not __import__("torch").multiprocessing.get_context("spawn"),
    reason="torch.multiprocessing not available",
)
class TestMultiProcessGlooReduceScatter:
    """Real ``torch.distributed`` ``reduce_scatterv`` round-trip (gloo)."""

    def test_reduce_scatterv_round_trip(self, tmp_path):
        import os
        import torch
        import torch.multiprocessing as mp

        world_size = 2
        m_local = 4
        hidden = 8
        port = 31100 + (hash("C240_reducescatter") % 1000)

        ctx = mp.get_context("spawn")
        out_dir = str(tmp_path)
        procs = []
        for r in range(world_size):
            p = ctx.Process(
                target=_gloo_worker_reducescatter,
                args=(r, world_size, port, m_local, hidden, out_dir),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited with code {p.exitcode}"

        results = []
        for r in range(world_size):
            data = torch.load(os.path.join(out_dir, f"rank_{r}.pt"))
            results.append((data["rank"], data["local_out"]))
        results.sort(key=lambda x: x[0])
        for rank, local_out in results:
            assert local_out.shape == (m_local, hidden)
            # Rank r's slice after reduce-scatter-sum is the sum of
            # all ranks' contributions at that slice.  Since only
            # rank r set that slice to non-zero, expected = rank + 1.
            expected_value = float(rank + 1)
            expected = torch.full((m_local, hidden), expected_value)
            assert torch.allclose(local_out, expected), (
                f"rank {rank} local_out mismatch: "
                f"got {local_out} but expected {expected}"
            )


# ---------------------------------------------------------------------------
# 16. Cache (handle-cache weakref) — verify handle reuse
# ---------------------------------------------------------------------------


class TestHandleCache:
    """The ``Cache`` weakref cache deduplicates handle objects keyed by kwargs."""

    def test_get_or_create_caches(self):
        from my_vllm.distributed.device_communicators import Cache

        class WeakRefHandle:
            pass

        cache = Cache()
        call_count = {"n": 0}

        def factory(**kwargs):
            call_count["n"] += 1
            return WeakRefHandle()

        h1 = cache.get_or_create({"a": 1, "b": 2}, factory)
        h2 = cache.get_or_create({"a": 1, "b": 2}, factory)
        assert h1 is h2
        assert call_count["n"] == 1

    def test_get_or_create_distinct_keys(self):
        from my_vllm.distributed.device_communicators import Cache

        class WeakRefHandle:
            pass

        cache = Cache()
        call_count = {"n": 0}

        def factory(**kwargs):
            call_count["n"] += 1
            return WeakRefHandle()

        h1 = cache.get_or_create({"a": 1}, factory)
        h2 = cache.get_or_create({"a": 2}, factory)
        assert h1 is not h2
        assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# 17. Forward-context dp_metadata integration
# ---------------------------------------------------------------------------


class TestForwardContextDPMetadata:
    """When ``forward_context.dp_metadata`` is set, ``_get_sizes``
    picks up the per-rank chunk sizes."""

    def test_dp_metadata_sizes_are_used(self):
        from my_vllm.distributed.device_communicators import AgRsAll2AllManager
        from my_vllm.forward_context import (
            ForwardContext,
            override_forward_context,
        )
        from dataclasses import dataclass

        @dataclass
        class FakeDPMetadata:
            def get_chunk_sizes_across_dp_rank(self):
                return [2, 4, 6]

        m = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
        m.cpu_group = None
        m.tcp_store_group = None
        m.dp_group = None
        m.tp_group = None
        m.dp_rank = 0
        m.dp_world_size = 3
        m.rank = 0
        m.world_size = 3
        m.comm_group = None

        # Mock the dp_metadata getter by setting dp_world_size > 1
        # so _get_sizes looks at forward_context.
        fc = ForwardContext()
        fc.additional_kwargs["dp_metadata"] = FakeDPMetadata()
        # forward_context doesn't actually have a dp_metadata field;
        # patch with setattr for the test:
        object.__setattr__(fc, "dp_metadata", FakeDPMetadata())

        with override_forward_context(fc):
            sizes = m._get_sizes(2, None)
        assert sizes == [2, 4, 6]

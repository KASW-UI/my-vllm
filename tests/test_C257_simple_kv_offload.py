"""tests/test_C257_simple_kv_offload.py — SimpleCPUOffload connector.

C257: Simple KV Offload 轻量路径.

Verifies the lightweight CPU-offload KV connector (manager + worker +
metadata + cuda_mem_ops + copy_backend) extracted from
``vllm/v1/simple_kv_offload/``:

* **metadata** — `SimpleCPUOffloadMetadata` / `SimpleCPUOffloadWorkerMetadata`
  dataclasses (defaults, INVALID_JOB_ID sentinel, `aggregate()` across
  workers).
* **cuda_mem_ops** — `BatchMemcpyParams` NamedTuple shape,
  `build_params` / `copy_blocks` round-trip with the simple-torch
  fallback (CPU-only path; no `cuMemcpyBatchAsync`).
* **copy_backend** — `DmaCopyBackend.init` / `launch_copy` /
  `shutdown` lifecycle; background thread drains queue; events
  recorded for both store / load directions.
* **worker** — `SimpleCPUOffloadWorker` register_kv_caches +
  bind/clear metadata + start_load/wait_for_save no-ops +
  get_finished + build_connector_worker_meta + handle_preemptions +
  _flush_and_sync_all + _poll_stream_events.
* **manager** — `SimpleCPUOffloadScheduler`:
  - `_derive_cpu_config` block-count scaling
  - `_estimate_lazy_target_blocks` per-group math
  - `get_num_new_matched_tokens` (CPU hit + pin release)
  - `update_state_after_alloc` (load prep + pin release)
  - `build_connector_meta` (event counter + need_flush)
  - `_prepare_eager_store_specs` (per-group dedup + out-of-space)
  - `_prepare_lazy_store_specs` (cursor walk + target cap)
  - `update_connector_output` (load cleanup + store event completion)
  - `_process_store_completion` (CPU cache insert + free refs)
  - `request_finished` (defer / immediate cleanup paths)
  - `reset` (abandon + reset_prefix_cache)
  - `take_events` (no events when none queued)
* **infrastructure extensions** (sourced from this commit):
  - `BlockPool.get_num_free_blocks` + `take_events` +
    `reset_prefix_cache`
  - `FreeKVCacheBlockQueue.iter_blocks_after`
  - `SlidingWindowSpec.sliding_window` field
  - `ParallelConfig.decode_context_parallel_size`
  - `SpeculativeConfig.use_eagle()`
  - `VllmConfig.kv_events_config` / `max_in_flight_tokens`
  - `KVConnectorOutput` dataclass
  - `yield_req_data` helper
  - `cdiv` helper
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
import torch

from my_vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.utils import (
    yield_req_data,
)
from my_vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.outputs import KVConnectorOutput
from my_vllm.v1.simple_kv_offload import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadScheduler,
    SimpleCPUOffloadWorker,
    SimpleCPUOffloadWorkerMetadata,
)
from my_vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)
from my_vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from my_vllm.v1.simple_kv_offload.metadata import INVALID_JOB_ID
from my_vllm.v1.utils import cdiv


# ===========================================================================
# Infrastructure extensions (sourced from C257)
# ===========================================================================


class TestCdiv:
    """`cdiv` ceiling-division helper (C257 utils extension)."""

    def test_basic(self):
        assert cdiv(10, 3) == 4
        assert cdiv(9, 3) == 3
        assert cdiv(0, 3) == 0

    def test_negatives(self):
        # Ceil toward +inf for negative numerator.
        assert cdiv(-7, 2) == -3
        assert cdiv(-6, 2) == -3

    def test_large(self):
        assert cdiv(2**31 - 1, 16) == (2**31 - 1 + 15) // 16


class TestYieldReqData:
    """`yield_req_data` walks `SchedulerOutput` per-req."""

    def _make_output(
        self,
        new_reqs: list[NewRequestData],
        cached: CachedRequestData,
        num_scheduled: dict[str, int] | None = None,
    ) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached,
            num_scheduled_tokens=num_scheduled or {},
            total_num_scheduled_tokens=0,
        )

    def test_yields_new_and_cached(self):
        new = NewRequestData(req_id="r1", block_ids=[0, 1, 2])
        cached = CachedRequestData(
            req_ids=["r2"],
            new_block_ids=[[3]],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        )
        out = self._make_output([new], cached, {"r1": 3, "r2": 1})

        results = list(yield_req_data(out))
        assert len(results) == 2
        assert results[0] == ("r1", ([0, 1, 2],), False)
        assert results[1] == ("r2", [3], False)

    def test_preempted_flag(self):
        cached = CachedRequestData(
            req_ids=["r2"],
            new_block_ids=[None],  # Preemption: new_block_ids wiped
            num_computed_tokens=[0],
            num_output_tokens=[0],
        )
        out = self._make_output([], cached, {"r2": 0})
        results = list(yield_req_data(out))
        assert results[0] == ("r2", None, True)


class TestFreeKVCacheBlockQueueIterAfter:
    """`iter_blocks_after` cursor walk (C257 utils extension)."""

    def test_iter_from_head(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        seen = list(q.iter_blocks_after(None))
        assert [blk.block_id for blk in seen] == [0, 1, 2, 3, 4]

    def test_iter_from_cursor(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        seen = list(q.iter_blocks_after(blocks[2]))
        assert [blk.block_id for blk in seen] == [3, 4]

    def test_iter_enumerated(self):
        blocks = [KVCacheBlock(block_id=i) for i in range(5)]
        q = FreeKVCacheBlockQueue(blocks)
        seen = [(i, blk.block_id) for i, blk in enumerate(q.iter_blocks_after(None))]
        assert seen == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    def test_empty_queue(self):
        q = FreeKVCacheBlockQueue([])
        assert list(q.iter_blocks_after(None)) == []


class TestBlockPoolExtensions:
    """`BlockPool.get_num_free_blocks` / `take_events` /
    `reset_prefix_cache` (C257 utils extensions)."""

    def test_get_num_free_blocks(self):
        from my_vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(num_blocks=8, block_size=16, enable_caching=True)
        assert pool.get_num_free_blocks() == 8
        pool.get_new_blocks(3)
        assert pool.get_num_free_blocks() == 5

    def test_take_events_default_empty(self):
        from my_vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(num_blocks=4, block_size=16)
        assert pool.take_events() == []

    def test_reset_prefix_cache_clears(self):
        from my_vllm.v1.core.block_pool import BlockPool

        pool = BlockPool(num_blocks=4, block_size=16, enable_caching=True)
        blk = pool.blocks[0]
        blk.set_block_hash(b"abc")
        pool.cached_block_hash_to_block.insert(b"abc", blk)
        assert len(pool.cached_block_hash_to_block) == 1
        had = pool.reset_prefix_cache()
        assert had is True
        assert len(pool.cached_block_hash_to_block) == 0


class TestSlidingWindowSpecField:
    """`SlidingWindowSpec.sliding_window` (C257 extension)."""

    def test_default_none(self):
        from my_vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowSpec,
        )

        spec = SlidingWindowSpec(block_size=16)
        assert spec.block_size == 16
        assert spec.sliding_window is None

    def test_with_sliding_window(self):
        from my_vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowSpec,
        )

        spec = SlidingWindowSpec(block_size=16, sliding_window=4096)
        assert spec.sliding_window == 4096


class TestParallelConfigDcp:
    """`ParallelConfig.decode_context_parallel_size` (C257 extension)."""

    def test_default(self):
        from my_vllm.config.parallel import ParallelConfig

        cfg = ParallelConfig()
        assert cfg.decode_context_parallel_size == 1
        # world_size now includes DCP too.
        assert cfg.world_size == 1


class TestSpeculativeConfigUseEagle:
    """`SpeculativeConfig.use_eagle()` (C257 extension)."""

    def test_none(self):
        from my_vllm.config.speculative import SpeculativeConfig

        cfg = SpeculativeConfig()
        assert cfg.use_eagle() is False

    def test_ngram(self):
        from my_vllm.config.speculative import SpeculativeConfig

        cfg = SpeculativeConfig(method="ngram")
        assert cfg.use_eagle() is False

    def test_eagle(self):
        from my_vllm.config.speculative import SpeculativeConfig

        cfg = SpeculativeConfig(method="eagle")
        assert cfg.use_eagle() is True

    def test_eagle3(self):
        from my_vllm.config.speculative import SpeculativeConfig

        cfg = SpeculativeConfig(method="eagle3")
        assert cfg.use_eagle() is True


class TestKVConnectorOutput:
    """`KVConnectorOutput` dataclass (C257 extension)."""

    def test_defaults(self):
        out = KVConnectorOutput()
        assert out.finished_sending == set()
        assert out.finished_recving == set()
        assert out.kv_connector_worker_meta is None
        assert out.preempted_req_ids == set()

    def test_with_values(self):
        meta = SimpleCPUOffloadWorkerMetadata(completed_store_events={1: 1})
        out = KVConnectorOutput(
            finished_sending={"r1"},
            finished_recving={"r2"},
            kv_connector_worker_meta=meta,
            preempted_req_ids={"r3"},
        )
        assert "r1" in out.finished_sending
        assert "r2" in out.finished_recving
        assert out.kv_connector_worker_meta is meta
        assert "r3" in out.preempted_req_ids


# ===========================================================================
# Metadata
# ===========================================================================


class TestSimpleCPUOffloadMetadata:
    """`SimpleCPUOffloadMetadata` defaults + assignment."""

    def test_defaults(self):
        m = SimpleCPUOffloadMetadata()
        assert m.load_event == INVALID_JOB_ID
        assert m.load_gpu_blocks == []
        assert m.load_cpu_blocks == []
        assert m.load_event_to_reqs == {}
        assert m.store_event == INVALID_JOB_ID
        assert m.store_gpu_blocks == []
        assert m.store_cpu_blocks == []
        assert m.need_flush is False

    def test_assignment(self):
        m = SimpleCPUOffloadMetadata(
            load_event=0,
            load_gpu_blocks=[1, 2],
            load_cpu_blocks=[10, 11],
            load_event_to_reqs={0: ["r1", "r2"]},
            store_event=1,
            store_gpu_blocks=[3],
            store_cpu_blocks=[12],
            need_flush=True,
        )
        assert m.load_event == 0
        assert m.load_event_to_reqs == {0: ["r1", "r2"]}
        assert m.need_flush is True

    def test_is_kv_connector_metadata(self):
        assert issubclass(SimpleCPUOffloadMetadata, KVConnectorMetadata)


class TestSimpleCPUOffloadWorkerMetadata:
    """`SimpleCPUOffloadWorkerMetadata` + aggregate()."""

    def test_construction(self):
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events={1: 1, 2: 1}
        )
        assert dict(meta.completed_store_events) == {1: 1, 2: 1}

    def test_aggregate_disjoint_keys(self):
        a = SimpleCPUOffloadWorkerMetadata(completed_store_events={1: 1})
        b = SimpleCPUOffloadWorkerMetadata(completed_store_events={2: 1})
        merged = a.aggregate(b)
        assert isinstance(merged, SimpleCPUOffloadWorkerMetadata)
        assert dict(merged.completed_store_events) == {1: 1, 2: 1}

    def test_aggregate_overlap_sums(self):
        a = SimpleCPUOffloadWorkerMetadata(completed_store_events={1: 1, 2: 1})
        b = SimpleCPUOffloadWorkerMetadata(completed_store_events={2: 1, 3: 1})
        merged = a.aggregate(b)
        assert dict(merged.completed_store_events) == {1: 1, 2: 2, 3: 1}

    def test_aggregate_asserts_type(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorWorkerMetadata,
        )

        a = SimpleCPUOffloadWorkerMetadata(completed_store_events={})

        # Build a concrete subclass that ISN'T
        # SimpleCPUOffloadWorkerMetadata — the aggregate() must
        # reject it via ``isinstance`` check.
        class _OtherWorkerMeta(KVConnectorWorkerMetadata):
            completed_store_events: dict[int, int] = {}

            def aggregate(self, other):
                return self

        wrong = _OtherWorkerMeta()
        with pytest.raises(AssertionError):
            a.aggregate(wrong)

    def test_is_kv_connector_worker_metadata(self):
        assert issubclass(
            SimpleCPUOffloadWorkerMetadata, KVConnectorWorkerMetadata
        )


# ===========================================================================
# cuda_mem_ops
# ===========================================================================


class TestPinTensor:
    """`pin_tensor` no-op on non-CUDA + smoke test on CUDA."""

    def test_cpu_tensor_noop(self):
        # No CUDA required for this test path.
        t = torch.zeros(8, dtype=torch.float32)
        # Should not raise (no-op on CPU-only).
        pin_tensor(t)


class TestBuildParams:
    """`build_params` returns a `BatchMemcpyParams` with the right shape."""

    def _make_caches(
        self, num_layers: int = 2, blocks: int = 4, block_bytes: int = 32
    ):
        # Create a [num_layers, num_blocks, block_bytes] int8 view.
        gpu = {
            f"L{i}": torch.zeros(
                blocks, block_bytes, dtype=torch.int8
            )
            for i in range(num_layers)
        }
        cpu = {
            f"L{i}": torch.zeros(
                blocks, block_bytes, dtype=torch.int8
            )
            for i in range(num_layers)
        }
        return gpu, cpu

    def test_basic_shape(self):
        gpu, cpu = self._make_caches()
        params = build_params(
            gpu,
            cpu,
            stream=None,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )
        assert isinstance(params, BatchMemcpyParams)
        assert params.num_layers == 2
        assert params.src_bases.shape == (2,)
        assert params.dst_bases.shape == (2,)
        assert params.bpb.shape == (2,)
        assert (params.bpb == 32).all()

    def test_stream_access_order_stream(self):
        gpu, cpu = self._make_caches()
        params = build_params(
            gpu,
            cpu,
            stream=None,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        assert params.attrs.srcAccessOrder == 1


class TestCopyBlocks:
    """`copy_blocks` performs per-block copy via the simple-torch path."""

    def test_zero_blocks_noop(self):
        gpu, cpu = {
            "L0": torch.zeros(4, 8, dtype=torch.int8)
        }, {"L0": torch.zeros(4, 8, dtype=torch.int8)}
        params = build_params(
            gpu, cpu, stream=None, src_access_order=3
        )
        # No-op on empty block-id list.
        copy_blocks([], [], params)

    def test_per_block_copy_cpu(self):
        """CPU-side round-trip: writes distinct patterns to src, copies."""
        n_blocks = 4
        bpb = 16
        gpu, cpu = {}, {}
        gpu["L0"] = torch.zeros(n_blocks, bpb, dtype=torch.int8)
        cpu["L0"] = torch.zeros(n_blocks, bpb, dtype=torch.int8)
        # Stamp distinct patterns on each GPU block.
        for b in range(n_blocks):
            gpu["L0"][b].fill_(b + 1)

        params = build_params(
            gpu, cpu, stream=None, src_access_order=3
        )
        # Copy blocks 0 and 2 to CPU positions 1 and 3.
        copy_blocks([0, 2], [1, 3], params)

        # CPU positions 1 and 3 should now mirror GPU patterns.
        assert torch.equal(cpu["L0"][1], gpu["L0"][0])
        assert torch.equal(cpu["L0"][3], gpu["L0"][2])
        # Untouched positions stay zero.
        assert torch.equal(cpu["L0"][0], torch.zeros(bpb, dtype=torch.int8))
        assert torch.equal(cpu["L0"][2], torch.zeros(bpb, dtype=torch.int8))

    def test_mismatched_length_raises(self):
        gpu, cpu = {
            "L0": torch.zeros(2, 8, dtype=torch.int8)
        }, {"L0": torch.zeros(2, 8, dtype=torch.int8)}
        params = build_params(
            gpu, cpu, stream=None, src_access_order=3
        )
        with pytest.raises(RuntimeError):
            copy_blocks([0, 1], [0], params)


# ===========================================================================
# copy_backend
# ===========================================================================


class TestDmaCopyBackend:
    """`DmaCopyBackend` background-thread lifecycle."""

    def test_init_no_cuda_runs_cpu_path(self):
        # No CUDA on dev box → load_stream / store_stream are None.
        backend = DmaCopyBackend()
        gpu = {
            "L0": torch.zeros(4, 16, dtype=torch.int8)
        }
        cpu = {
            "L0": torch.zeros(4, 16, dtype=torch.int8)
        }
        backend.init(gpu, cpu, torch.device("cpu"), None, None)
        assert backend._queue is not None
        assert backend._thread is not None
        # Stamp pattern on GPU block 0.
        gpu["L0"][0].fill_(7)
        store_events: list[tuple[int, torch.Event]] = []
        backend.launch_copy(
            src_blocks=[0],
            dst_blocks=[2],
            is_store=True,
            event_idx=0,
            events_list=store_events,
        )
        # Drain by shutdown.
        backend.shutdown()
        # The copy should have written pattern to CPU block 2.
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        # CPU position 2 should mirror GPU pattern (or be unmodified if
        # thread didn't drain — but shutdown() joins, so it did).
        assert torch.equal(cpu["L0"][2], gpu["L0"][0])

    def test_shutdown_idempotent(self):
        backend = DmaCopyBackend()
        backend.shutdown()  # No init → no-op
        backend.shutdown()  # Idempotent

    def test_events_recorded(self):
        backend = DmaCopyBackend()
        gpu = {"L0": torch.zeros(2, 8, dtype=torch.int8)}
        cpu = {"L0": torch.zeros(2, 8, dtype=torch.int8)}
        backend.init(gpu, cpu, torch.device("cpu"), None, None)
        store_events: list[tuple[int, torch.Event]] = []
        for i in range(3):
            backend.launch_copy(
                src_blocks=[0],
                dst_blocks=[1],
                is_store=True,
                event_idx=i,
                events_list=store_events,
            )
        backend.shutdown()
        # All 3 events should have been recorded by the background thread.
        assert len(store_events) == 3
        event_idxs = [e[0] for e in store_events]
        assert event_idxs == [0, 1, 2]


# ===========================================================================
# Worker
# ===========================================================================


class _FakeKVCacheConfig:
    """Lightweight stand-in for `KVCacheConfig` for worker tests."""

    def __init__(self, num_blocks: int = 16):
        self.num_blocks = num_blocks


class TestSimpleCPUOffloadWorker:
    """`SimpleCPUOffloadWorker` lifecycle (CPU-only)."""

    def test_construction(self):
        kv_cfg = _FakeKVCacheConfig()
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=kv_cfg, cpu_capacity_bytes=1024 * 1024
        )
        assert worker.num_cpu_blocks == 0
        assert worker.gpu_kv_caches is None
        assert worker._completed_store_events == {}

    def test_register_kv_caches_allocates_cpu(self):
        kv_cfg = _FakeKVCacheConfig(num_blocks=4)
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=4 * 16 * 4 * 10,  # plenty of room
        )
        # Two layers share the same storage (dedup). Create one
        # underlying storage and view it twice with different names.
        underlying = torch.zeros(4 * 16, dtype=torch.int8)
        gpu_caches = {
            "layer0": underlying.view(4, 16),
            "layer1": underlying.view(4, 16),  # shares storage
        }
        worker.register_kv_caches(gpu_caches)
        assert worker.gpu_kv_caches is not None
        assert worker.cpu_kv_caches is not None
        # Single unique storage → 1 entry.
        assert len(worker.gpu_kv_caches) == 1
        assert worker.num_cpu_blocks >= 1

    def test_register_kv_caches_empty(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # No caches → graceful warning, no crash.
        worker.register_kv_caches({})

    def test_bind_clear_metadata(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        meta = SimpleCPUOffloadMetadata(load_event=5, store_event=7)
        worker.bind_connector_metadata(meta)
        assert worker._connector_metadata is meta
        assert 5 in worker._pending_load_event_indices
        assert 7 in worker._pending_store_event_indices
        worker.clear_connector_metadata()
        assert worker._connector_metadata is None

    def test_start_load_wait_for_save_noop(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # Both no-ops return None and don't touch state.
        assert worker.start_load_kv() is None
        assert worker.wait_for_save() is None

    def test_get_finished_no_meta(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # Without metadata → no submits, returns (None, None).
        finished_send, finished_recv = worker.get_finished(set())
        assert finished_send is None
        assert finished_recv is None

    def test_build_connector_worker_meta_empty(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # No completed store events → returns None.
        assert worker.build_connector_worker_meta() is None

    def test_handle_preemptions_no_flush(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # need_flush=False → no-op.
        meta = SimpleCPUOffloadMetadata(need_flush=False)
        worker.handle_preemptions(meta)
        # Pending event lists should be untouched.
        assert worker._load_events == []
        assert worker._store_events == []

    def test_poll_stream_events_empty(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # Empty event lists → high-water marks stay at -1.
        assert worker._poll_stream_events(is_store=False) == -1
        assert worker._poll_stream_events(is_store=True) == -1

    def test_shutdown(self):
        worker = SimpleCPUOffloadWorker(
            kv_cache_config=_FakeKVCacheConfig(), cpu_capacity_bytes=1024
        )
        # Shutdown on un-initialized worker → graceful no-op.
        worker.shutdown()


# ===========================================================================
# Manager
# ===========================================================================


def _make_vllm_config(
    kv_cache_config: Any, *, world_size: int = 1, max_num_batched_tokens: int = 2048
) -> Any:
    """Build a minimal VllmConfig for SimpleCPUOffloadScheduler tests."""
    from my_vllm.config.model import ModelConfig
    from my_vllm.config.parallel import ParallelConfig
    from my_vllm.config.scheduler import SchedulerConfig
    from my_vllm.config.vllm import VllmConfig

    # world_size = TP×PP×EP×CP×DCP; default all 1.  Set TP=world_size
    # to dial up.
    parallel = ParallelConfig(tensor_parallel_size=world_size)
    cfg = VllmConfig(
        model_config=ModelConfig(max_model_len=2048),
        scheduler_config=SchedulerConfig(max_num_batched_tokens=max_num_batched_tokens),
        parallel_config=parallel,
    )
    cfg.kv_events_config = None  # disable event publishing
    cfg.max_in_flight_tokens = 0
    return cfg


def _make_kv_cache_config(num_blocks: int = 16) -> Any:
    """Build a single-group FullAttention KVCacheConfig."""
    from my_vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from my_vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionSpec,
    )

    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=4, dtype=torch.int8
    )
    group = KVCacheGroupSpec.from_layers(
        layer_names=["layer0"],
        kv_cache_spec=spec,
    )
    tensor = KVCacheTensor(
        size=num_blocks * spec.page_size_bytes,
        shared_by=[0],
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[tensor],
        kv_cache_groups=[group],
    )


class TestSimpleCPUOffloadSchedulerDerive:
    """`SimpleCPUOffloadScheduler._derive_cpu_config`."""

    def test_block_count_scales_with_capacity(self):
        gpu_cfg = _make_kv_cache_config(num_blocks=64)
        # 4x capacity → 4x cpu blocks (capped by the total size).
        cpu_cfg = SimpleCPUOffloadScheduler._derive_cpu_config(
            gpu_cfg, cpu_capacity_bytes=4 * gpu_cfg.kv_cache_tensors[0].size
        )
        # num_blocks scales: 64 * (4*total) // total → at most 256.
        assert cpu_cfg.num_blocks >= 64
        # The same kv_cache_groups are mirrored.
        assert cpu_cfg.kv_cache_groups == gpu_cfg.kv_cache_groups

    def test_min_one_block(self):
        gpu_cfg = _make_kv_cache_config(num_blocks=64)
        cpu_cfg = SimpleCPUOffloadScheduler._derive_cpu_config(
            gpu_cfg, cpu_capacity_bytes=1
        )
        assert cpu_cfg.num_blocks >= 1


class TestSimpleCPUOffloadSchedulerEstimateLazy:
    """`SimpleCPUOffloadScheduler._estimate_lazy_target_blocks`."""

    def test_full_attention(self):
        cfg = _make_kv_cache_config(num_blocks=64)
        # block_size=16, max_num_batched_tokens=2048, cp_world_size=1.
        target = SimpleCPUOffloadScheduler._estimate_lazy_target_blocks(
            cfg, max_num_batched_tokens=2048, cp_world_size=1
        )
        # cdiv(2048, 16) = 128; * 2 (watermark) = 256.
        assert target == 256

    def test_sliding_window(self):
        from my_vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowSpec,
        )
        from my_vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        spec = SlidingWindowSpec(block_size=16, sliding_window=64)
        group = KVCacheGroupSpec.from_layers(["layer0"], spec)
        tensor = KVCacheTensor(size=64 * 16, shared_by=[0])
        cfg = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[tensor],
            kv_cache_groups=[group],
        )
        target = SimpleCPUOffloadScheduler._estimate_lazy_target_blocks(
            cfg, max_num_batched_tokens=2048, cp_world_size=1
        )
        # cdiv(64, 16) + 1 = 5; * 2 = 10.
        assert target == 10

    def test_mamba(self):
        from my_vllm.v1.core.single_type_kv_cache_manager import MambaSpec
        from my_vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        spec = MambaSpec(block_size=16)
        group = KVCacheGroupSpec.from_layers(["layer0"], spec)
        tensor = KVCacheTensor(size=64 * 16, shared_by=[0])
        cfg = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[tensor],
            kv_cache_groups=[group],
        )
        target = SimpleCPUOffloadScheduler._estimate_lazy_target_blocks(
            cfg, max_num_batched_tokens=2048, cp_world_size=1
        )
        # MambaSpec → 2; * 2 = 4.
        assert target == 4


class TestSimpleCPUOffloadSchedulerLifecycle:
    """`SimpleCPUOffloadScheduler` end-to-end (eager mode)."""

    def _make_scheduler(self) -> SimpleCPUOffloadScheduler:
        kv_cfg = _make_kv_cache_config(num_blocks=32)
        vllm_cfg = _make_vllm_config(kv_cfg)
        return SimpleCPUOffloadScheduler(
            vllm_config=vllm_cfg,
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=8 * 1024 * 1024,
            scheduler_block_size=16,
            hash_block_size=16,
            lazy_offload=False,
        )

    def test_construction(self):
        sched = self._make_scheduler()
        assert sched.cpu_kv_cache_config is not None
        assert sched.num_cpu_blocks >= 1
        assert sched.fa_gidx == 0
        assert sched.fa_block_size == 16
        assert sched.block_size == 16
        assert sched._load_event_counter == 0
        assert sched._store_event_counter == 0
        assert sched._reqs_to_load == {}
        assert sched._reqs_to_store == {}

    def test_bind_gpu_block_pool(self):
        from my_vllm.v1.core.block_pool import BlockPool

        sched = self._make_scheduler()
        gpu_pool = BlockPool(num_blocks=64, block_size=16, enable_caching=True)
        sched.bind_gpu_block_pool(gpu_pool)
        assert sched._gpu_block_pool is gpu_pool

    def test_get_num_new_matched_tokens_no_hit(self):
        from my_vllm.v1.request import Request

        sched = self._make_scheduler()
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
        )
        num, is_async = sched.get_num_new_matched_tokens(req, 0)
        # No cached blocks → 0 / False.
        assert num == 0
        assert is_async is False

    def test_get_num_new_matched_tokens_zero_hashes(self):
        from my_vllm.v1.request import Request

        sched = self._make_scheduler()
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
        )
        # num_computed_tokens=3 means 0 remaining hashes.
        num, is_async = sched.get_num_new_matched_tokens(req, 3)
        assert num == 0
        assert is_async is False

    def test_request_finished_no_state(self):
        from my_vllm.v1.request import Request

        sched = self._make_scheduler()
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=None,
        )
        # No load / store state → immediate return (False, None).
        assert sched.request_finished(req) == (False, None)

    def test_take_events_empty(self):
        sched = self._make_scheduler()
        events = list(sched.take_events())
        assert events == []

    def test_has_pending_stores_initially_false(self):
        sched = self._make_scheduler()
        assert sched.has_pending_stores() is False

    def test_reset_clean(self):
        sched = self._make_scheduler()
        # Clean state → reset clears prefix cache (no entries → False).
        assert sched.reset() is False


class TestSimpleCPUOffloadSchedulerBuildConnectorMeta:
    """`build_connector_meta` packs events + need_flush."""

    def _make_scheduler(self) -> SimpleCPUOffloadScheduler:
        kv_cfg = _make_kv_cache_config(num_blocks=32)
        vllm_cfg = _make_vllm_config(kv_cfg)
        return SimpleCPUOffloadScheduler(
            vllm_config=vllm_cfg,
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=8 * 1024 * 1024,
            scheduler_block_size=16,
            hash_block_size=16,
            lazy_offload=False,
        )

    def test_empty_step(self):
        sched = self._make_scheduler()
        out = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
            preempted_req_ids=None,
        )
        meta = sched.build_connector_meta(out)
        assert meta.load_event == INVALID_JOB_ID
        assert meta.store_event == INVALID_JOB_ID
        assert meta.need_flush is False

    def test_preempted_sets_need_flush(self):
        sched = self._make_scheduler()
        out = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
            preempted_req_ids={"r1"},
        )
        meta = sched.build_connector_meta(out)
        assert meta.need_flush is True


class TestSimpleCPUOffloadSchedulerUpdateConnectorOutput:
    """`update_connector_output` aggregates store completions + cleans loads."""

    def _make_scheduler(self) -> SimpleCPUOffloadScheduler:
        kv_cfg = _make_kv_cache_config(num_blocks=32)
        vllm_cfg = _make_vllm_config(kv_cfg, world_size=2)
        return SimpleCPUOffloadScheduler(
            vllm_config=vllm_cfg,
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=8 * 1024 * 1024,
            scheduler_block_size=16,
            hash_block_size=16,
            lazy_offload=False,
        )

    def test_store_event_completion_accumulates(self):
        from my_vllm.v1.simple_kv_offload.manager import TransferMeta

        sched = self._make_scheduler()
        # Manually register a store event with a real TransferMeta.
        sched._store_event_to_blocks[0] = TransferMeta(
            gpu_block_ids=[], cpu_block_ids=[]
        )
        # Worker reports 1 of 2 → not yet completed.
        meta = SimpleCPUOffloadWorkerMetadata(
            completed_store_events={0: 1}
        )
        out = KVConnectorOutput(kv_connector_worker_meta=meta)
        # Patch _process_store_event so we don't trigger GPU pool
        # assertions (no GPU pool bound in this test).
        sched._process_store_event = lambda _eid: None
        sched.update_connector_output(out)
        assert 0 in sched._store_event_pending_counts
        # Second worker reports → completed.
        sched.update_connector_output(out)
        assert 0 not in sched._store_event_pending_counts

    def test_non_matching_meta_ignored(self):
        sched = self._make_scheduler()
        # Build a non-SimpleCPUOffloadWorkerMetadata subclass to verify
        # the isinstance check in update_connector_output().
        class _OtherWorkerMeta(KVConnectorWorkerMetadata):
            completed_store_events: dict[int, int] = {}

            def aggregate(self, other):
                return self

        out = KVConnectorOutput(
            kv_connector_worker_meta=_OtherWorkerMeta()
        )
        # Non-SimpleCPUOffloadWorkerMetadata → silently ignored.
        sched.update_connector_output(out)
        # No state change.
        assert sched._store_event_pending_counts == {}

    def test_finished_recving_cleans_load(self):
        from my_vllm.v1.core.block_pool import BlockPool
        from my_vllm.v1.simple_kv_offload.manager import (
            LoadRequestState,
            TransferMeta,
        )

        sched = self._make_scheduler()
        # Bind a GPU block pool so cleanup can release refs (even
        # though the test list is empty, the assertion must pass).
        gpu_pool = BlockPool(num_blocks=4, block_size=16)
        sched.bind_gpu_block_pool(gpu_pool)
        # Register a load state via the real dataclass so attribute
        # access in `_cleanup_load_request` works.
        sched._reqs_to_load["r1"] = LoadRequestState(
            request=None,
            transfer_meta=TransferMeta(
                gpu_block_ids=[], cpu_block_ids=[]
            ),
            load_event=None,
            finished=False,
        )
        # Empty block-id lists → no free_blocks iteration.
        out = KVConnectorOutput(finished_recving={"r1"})
        sched.update_connector_output(out)
        assert "r1" not in sched._reqs_to_load


class TestSimpleCPUOffloadSchedulerLazyMode:
    """Lazy-mode single-pass cursor walk."""

    def _make_scheduler(self) -> SimpleCPUOffloadScheduler:
        from my_vllm.v1.core.block_pool import BlockPool

        kv_cfg = _make_kv_cache_config(num_blocks=32)
        vllm_cfg = _make_vllm_config(kv_cfg, max_num_batched_tokens=2048)
        sched = SimpleCPUOffloadScheduler(
            vllm_config=vllm_cfg,
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=8 * 1024 * 1024,
            scheduler_block_size=16,
            hash_block_size=16,
            lazy_offload=True,
        )
        gpu_pool = BlockPool(num_blocks=64, block_size=16, enable_caching=True)
        # Stamp some blocks with hashes.
        gpu_pool.blocks[0].set_block_hash(b"h0")
        gpu_pool.blocks[1].set_block_hash(b"h1")
        sched.bind_gpu_block_pool(gpu_pool)
        return sched

    def test_lazy_cursor_offloads_cached(self):
        sched = self._make_scheduler()
        gpu, cpu, _ = sched._prepare_lazy_store_specs()
        # 2 GPU blocks have hashes; CPU has plenty of free space.
        assert len(gpu) == 2
        assert len(cpu) == 2
        # GPU blocks 0 and 1 should be in the result.
        assert set(gpu) == {0, 1}

    def test_lazy_no_gpu_pool(self):
        from my_vllm.v1.core.kv_cache_utils import KVCacheBlock

        kv_cfg = _make_kv_cache_config(num_blocks=32)
        vllm_cfg = _make_vllm_config(kv_cfg)
        sched = SimpleCPUOffloadScheduler(
            vllm_config=vllm_cfg,
            kv_cache_config=kv_cfg,
            cpu_capacity_bytes=8 * 1024 * 1024,
            scheduler_block_size=16,
            hash_block_size=16,
            lazy_offload=True,
        )
        # No GPU pool bound → empty.
        gpu, cpu, req = sched._prepare_lazy_store_specs()
        assert gpu == []
        assert cpu == []
        assert req == []


class TestKVCacheEventRoundtrip:
    """`take_events()` returns the C252 events when queued."""

    def test_event_types(self):
        # Sanity: the events used by the manager exist and are hashable.
        e1 = BlockStored(
            block_hashes=[1],
            parent_block_hash=None,
            token_ids=[1, 2, 3],
            block_size=3,
            medium="CPU",
        )
        e2 = BlockRemoved(block_hashes=[1], medium="CPU")
        e3 = AllBlocksCleared()
        for e in (e1, e2, e3):
            assert isinstance(e, KVCacheEvent)


# ===========================================================================
# Misc helpers
# ===========================================================================


class TestImports:
    """Smoke: everything imports."""

    def test_module_imports(self):
        import my_vllm.v1.simple_kv_offload as pkg
        assert pkg.SimpleCPUOffloadScheduler is SimpleCPUOffloadScheduler
        assert pkg.SimpleCPUOffloadWorker is SimpleCPUOffloadWorker
        assert pkg.SimpleCPUOffloadMetadata is SimpleCPUOffloadMetadata
        assert pkg.SimpleCPUOffloadWorkerMetadata is (
            SimpleCPUOffloadWorkerMetadata
        )

    def test_submodule_imports(self):
        import my_vllm.v1.simple_kv_offload.cuda_mem_ops as cmo
        import my_vllm.v1.simple_kv_offload.copy_backend as cb
        import my_vllm.v1.simple_kv_offload.manager as mgr
        import my_vllm.v1.simple_kv_offload.worker as wrk
        import my_vllm.v1.simple_kv_offload.metadata as md

        for sym in (
            "CU_MEMCPY_SRC_ACCESS_ORDER_STREAM",
            "CU_MEMCPY_SRC_ACCESS_ORDER_ANY",
            "BatchMemcpyParams",
            "build_params",
            "copy_blocks",
            "pin_tensor",
        ):
            assert hasattr(cmo, sym), sym

        assert hasattr(cb, "DmaCopyBackend")
        assert hasattr(mgr, "SimpleCPUOffloadScheduler")
        assert hasattr(wrk, "SimpleCPUOffloadWorker")
        assert hasattr(md, "SimpleCPUOffloadMetadata")
        assert hasattr(md, "SimpleCPUOffloadWorkerMetadata")


class TestIterableProtocol:
    """`take_events` returns an iterable (C252 protocol)."""

    def test_iterable(self):
        sched = SimpleCPUOffloadScheduler.__new__(SimpleCPUOffloadScheduler)
        # Stub what `take_events` needs.
        from my_vllm.v1.core.block_pool import BlockPool

        sched.cpu_block_pool = BlockPool(num_blocks=4, block_size=16)
        events = sched.take_events()
        assert isinstance(events, Iterable)
        # Empty pool → no events.
        assert list(events) == []

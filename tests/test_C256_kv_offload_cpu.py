"""tests/test_C256_kv_offload_cpu.py — CPU + Tiering offload backend.

C256: KV Offload tiering (CPU / NVMe).

Verifies the CPU offload concrete backend + the tiering
composing-manager scaffold:

* **CPUOffloadingSpec** — construction reads `cpu_bytes_to_use` +
  computes `num_blocks` from the worker-KV-bytes-per-block and
  world_size.  Block-level alignment preserved (1 for
  single-group Dense; vLLM uses 512 for the mmap region).
* **CPUOffloadingManager** — pluggable CachePolicy (LRU or ARC),
  prepare_load + complete_load ref-counting, prepare_store +
  complete_store with eviction fallback, `store_threshold` ≥ 2
  filtering via `counts` LRU, events emission when
  `enable_events=True`, reset_cache + take_events lifecycle.
* **`LRUCachePolicy`** real impl — OrderedDict-based evictable
  list + full dict lookup; mark_evictable / mark_non_evictable
  for ref-count transitions; LRU eviction skipping protected keys.
* **`ARCCachePolicy`** shell — declared + importable; raises
  NotImplementedError on actual use.
* **`BlockStatus`** — ctypes.Structure; `ref_cnt = -1` initially
  (not ready); `is_ready` reflects ref_cnt ≥ 0.
* **`CPUOffloadingWorker`** — async submit_store / submit_load
  via single-direction handlers (stream-pool + event-pool +
  buffer-pool).  Synchronous fallback for CPU-only path.
* **`TieringOffloadingManager`** — lookup fan-out across primary
  + secondary tiers; OffloadingManager interface delegated to
  primary; `has_pending_work` aggregates across tiers;
  ParentManager `create_store_job` placeholder.
* **`CPUPrimaryTierOffloadingManager`** — wraps a
  `CPUOffloadingManager` as the primary tier + provides
  `get_kv_memoryview` accessor.
* **`TieringOffloadingSpec`** — extends `CPUOffloadingSpec`;
  parses `secondary_tiers` list from `extra_config`; constructs
  the composing manager with primary + secondary tiers.
* **`SecondaryTierFactory`** — lazy-loading registry; `example`
  tier pre-registered as a shell.
* **Backward compat** — C255 base tests still pass.

C256 doesn't ship a concrete secondary tier (Storage / FS /
P2P / ObjectStore land when needed). The `example` shell
suffices for testing the registry + factory wiring.
"""

from __future__ import annotations

from typing import Type

import pytest
import torch

from my_vllm.v1.kv_offload import (
    BlockStatus,
    CachePolicy,
    CPUOffloadingManager,
    CPUOffloadingMetrics,
    CPUOffloadingSpec,
    CPUOffloadingWorker,
    CPULoadStoreSpec,
    JobMetadata,
    LRUCachePolicy,
    LookupResult,
    OffloadKey,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingCacheConfig,
    OffloadingParallelConfig,
    OffloadingSpecFactory,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    make_offload_key,
)
from my_vllm.v1.kv_offload.cpu import (
    ARCCachePolicy,
)
from my_vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    Transfer,
)
from my_vllm.v1.kv_offload.tiering import (
    CPUPrimaryTierOffloadingManager,
    ExampleSecondaryTierManager,
    SecondaryTierFactory,
    TieringOffloadingManager,
    TieringOffloadingSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    extra_config: dict | None = None,
    worker_kv_bytes_per_block: int = 4096,
    world_size: int = 1,
    blocks_per_chunk: int = 1,
    tokens_per_hash: int = 16,
) -> OffloadingConfig:
    """Build a sane `OffloadingConfig` for spec tests."""
    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=16,
                layer_names=("layer.0",),
            ),
        ),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=extra_config or {},
        engine_id="engine-0",
        model=OffloadingModelConfig(name="m", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_hash,
            blocks_per_chunk=blocks_per_chunk,
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=False,
        ),
    )


def _make_keys(num: int, offset: int = 0) -> list[OffloadKey]:
    """Build N distinct OffloadKeys with deterministic hashes.

    `offset` shifts the hash bytes so two calls produce different
    key sets (useful for eviction tests that need 2nd-time keys
    distinct from 1st-time keys).
    """
    return [
        make_offload_key(
            bytes([i + offset, i + offset + 1, i + offset + 2, i + offset + 3]),
            group_idx=0,
        )
        for i in range(num)
    ]


def _make_canonical_kvcaches(
    num_gpu_blocks: int = 4, page_size_bytes: int = 4096
) -> "CanonicalKVCaches":
    """Build a tiny CanonicalKVCaches for worker tests."""
    from my_vllm.v1.kv_offload.base import (
        CanonicalKVCacheRef,
        CanonicalKVCacheTensor,
        CanonicalKVCaches,
    )
    t = torch.zeros(
        (num_gpu_blocks, page_size_bytes), dtype=torch.int8,
    )
    return CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(t, page_size_bytes)],
        group_data_refs=[[CanonicalKVCacheRef(0, page_size_bytes)]],
    )


# ===========================================================================
# CPUOffloadingMetrics + CPULoadStoreSpec
# ===========================================================================


class TestCPUCommon:
    """Common CPU offload types."""

    def test_metrics_constants_defined(self):
        assert CPUOffloadingMetrics.STORES_SKIPPED == "vllm:kv_offload_stores_skipped"
        assert CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC == "vllm:kv_offload_cpu_cache_usage_perc"
        assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE == "vllm:kv_offload_cpu_allocation_size"
        assert CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC == "vllm:kv_offload_cpu_cache_write_usage_perc"
        assert CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC == "vllm:kv_offload_cpu_cache_read_usage_perc"

    def test_cpu_load_store_spec_stores_int64(self):
        s = CPULoadStoreSpec([0, 1, 2, 3])
        assert s.block_ids.dtype == "int64"
        assert list(s.block_ids) == [0, 1, 2, 3]

    def test_cpu_load_store_spec_rejects_negative(self):
        with pytest.raises(AssertionError, match="non-negative"):
            CPULoadStoreSpec([-1])


# ===========================================================================
# BlockStatus + CachePolicy ABC + LRUCachePolicy
# ===========================================================================


class TestBlockStatus:
    """BlockStatus (ctypes.Structure)."""

    def test_block_id_set_on_construct(self):
        b = BlockStatus(block_id=42)
        assert b.block_id == 42

    def test_ref_cnt_initially_minus_one(self):
        b = BlockStatus(block_id=0)
        assert b.ref_cnt == -1
        assert not b.is_ready

    def test_is_ready_when_ref_cnt_zero(self):
        b = BlockStatus(block_id=0)
        b.ref_cnt = 0
        assert b.is_ready

    def test_is_ready_when_ref_cnt_positive(self):
        b = BlockStatus(block_id=0)
        b.ref_cnt = 1
        assert b.is_ready

    def test_is_not_ready_when_ref_cnt_negative(self):
        b = BlockStatus(block_id=0)
        b.ref_cnt = -2
        assert not b.is_ready


class TestLRUCachePolicy:
    """The LRU eviction policy."""

    def test_get_returns_none_for_missing(self):
        p = LRUCachePolicy(cache_capacity=10)
        assert p.get(make_offload_key(b"\x00", 0)) is None

    def test_insert_then_get(self):
        p = LRUCachePolicy(cache_capacity=10)
        k = make_offload_key(b"\x01", 0)
        b = BlockStatus(block_id=0)
        b.ref_cnt = 0  # ready
        p.insert(k, b)
        assert p.get(k) is b

    def test_remove_clears_from_both_lists(self):
        p = LRUCachePolicy(cache_capacity=10)
        k = make_offload_key(b"\x01", 0)
        b = BlockStatus(block_id=0)
        b.ref_cnt = 0
        p.insert(k, b)
        p.remove(k)
        assert p.get(k) is None
        assert k not in p.evictable_blocks

    def test_touch_moves_to_end_of_evictable(self):
        p = LRUCachePolicy(cache_capacity=10)
        k1 = make_offload_key(b"\x01", 0)
        k2 = make_offload_key(b"\x02", 0)
        b1 = BlockStatus(block_id=0)
        b1.ref_cnt = 0
        b2 = BlockStatus(block_id=1)
        b2.ref_cnt = 0
        p.insert(k1, b1)
        p.insert(k2, b2)
        # Order: k1, k2. Touch k1 → k1 moves to end.
        p.touch([k1], req_context=ReqContext(req_id="r1"))
        assert list(p.evictable_blocks.keys()) == [k2, k1]

    def test_evict_skips_protected(self):
        p = LRUCachePolicy(cache_capacity=10)
        k1 = make_offload_key(b"\x01", 0)
        k2 = make_offload_key(b"\x02", 0)
        b1 = BlockStatus(block_id=0)
        b1.ref_cnt = 0
        b2 = BlockStatus(block_id=1)
        b2.ref_cnt = 0
        p.insert(k1, b1)
        p.insert(k2, b2)
        # Evict 1, but k1 is protected → evict k2.
        evicted = p.evict(1, protected={k1})
        assert evicted is not None
        assert len(evicted) == 1
        assert evicted[0][0] == k2
        # k1 stays.
        assert p.get(k1) is b1

    def test_evict_returns_none_when_insufficient(self):
        p = LRUCachePolicy(cache_capacity=10)
        # No blocks → can't evict 1.
        assert p.evict(1, protected=set()) is None

    def test_mark_evictable_adds_to_list(self):
        p = LRUCachePolicy(cache_capacity=10)
        k = make_offload_key(b"\x01", 0)
        b = BlockStatus(block_id=0)
        b.ref_cnt = 0
        p.insert(k, b)
        # After insert, it's already in the evictable list.
        assert k in p.evictable_blocks
        # mark_non_evictable removes.
        p.mark_non_evictable(k)
        assert k not in p.evictable_blocks
        # mark_evictable re-adds.
        p.mark_evictable(k)
        assert k in p.evictable_blocks

    def test_clear_resets_state(self):
        p = LRUCachePolicy(cache_capacity=10)
        k = make_offload_key(b"\x01", 0)
        b = BlockStatus(block_id=0)
        b.ref_cnt = 0
        p.insert(k, b)
        p.clear()
        assert p.get(k) is None
        assert len(p.evictable_blocks) == 0


class TestARCCachePolicy:
    """ARC shell — declared but raises NotImplementedError on use."""

    def test_constructs(self):
        p = ARCCachePolicy(cache_capacity=10)
        # Shell has state slots that the real impl would maintain.
        assert hasattr(p, "t1")
        assert hasattr(p, "t2")
        assert hasattr(p, "b1")
        assert hasattr(p, "b2")
        assert hasattr(p, "p")

    def test_lookup_raises(self):
        p = ARCCachePolicy(cache_capacity=10)
        with pytest.raises(NotImplementedError, match="C256 shell"):
            p.get(make_offload_key(b"\x01", 0))

    def test_evict_raises(self):
        p = ARCCachePolicy(cache_capacity=10)
        with pytest.raises(NotImplementedError, match="C256 shell"):
            p.evict(1, protected=set())


# ===========================================================================
# CPUOffloadingManager
# ===========================================================================


class TestCPUOffloadingManager:
    """The scheduler-side manager for CPU offload."""

    def test_constructs_with_lru(self):
        m = CPUOffloadingManager(num_blocks=8, cache_policy="lru")
        assert m._policy.__class__ is LRUCachePolicy

    def test_constructs_with_arc_shell(self):
        m = CPUOffloadingManager(num_blocks=8, cache_policy="arc")
        assert m._policy.__class__ is ARCCachePolicy

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="Unknown cache policy"):
            CPUOffloadingManager(num_blocks=8, cache_policy="fifo")

    def test_lookup_returns_miss_when_empty(self):
        m = CPUOffloadingManager(num_blocks=8)
        assert m.lookup(
            make_offload_key(b"\x00", 0),
            ReqContext(req_id="r1"),
        ) == LookupResult.MISS

    def test_lookup_returns_hit_pending_during_write(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(1)
        out = m.prepare_store(keys, ctx)
        assert out is not None
        # Block is allocated but not ready → HIT_PENDING.
        assert m.lookup(keys[0], ctx) == LookupResult.HIT_PENDING
        # Complete the store → ready.
        m.complete_store(keys, ctx)
        assert m.lookup(keys[0], ctx) == LookupResult.HIT

    def test_prepare_load_requires_existing_block(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        # Prepare a store first.
        keys = _make_keys(1)
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx)
        # Now prepare_load works.
        spec = m.prepare_load(keys, ctx)
        assert spec is not None
        assert isinstance(spec, CPULoadStoreSpec)

    def test_prepare_load_unknown_key_raises(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        with pytest.raises(AssertionError, match="not found"):
            m.prepare_load(_make_keys(1), ctx)

    def test_prepare_load_block_must_be_ready(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(1)
        m.prepare_store(keys, ctx)
        # Block is write-pending → can't read.
        with pytest.raises(AssertionError, match="not ready"):
            m.prepare_load(keys, ctx)

    def test_complete_load_decrements_ref_cnt(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(1)
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx)
        block = m._policy.get(keys[0])
        assert block.ref_cnt == 0
        m.prepare_load(keys, ctx)
        assert block.ref_cnt == 1
        m.complete_load(keys, ctx)
        assert block.ref_cnt == 0

    def test_prepare_store_allocates_block_ids(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(3)
        out = m.prepare_store(keys, ctx)
        assert out is not None
        assert len(out.keys_to_store) == 3
        # 3 fresh blocks allocated.
        assert m._num_allocated_blocks == 3
        assert isinstance(out.store_spec, CPULoadStoreSpec)

    def test_prepare_store_deduplicates(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(2)
        # Store once.
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx)
        # Re-store same keys → 0 new blocks.
        out = m.prepare_store(keys, ctx)
        assert out is not None
        assert out.keys_to_store == []

    def test_prepare_store_evicts_to_make_room(self):
        m = CPUOffloadingManager(num_blocks=4)
        ctx = ReqContext(req_id="r1")
        # Fill cache with 4 blocks.
        first_keys = _make_keys(4, offset=0)
        first_out = m.prepare_store(first_keys, ctx)
        assert first_out is not None
        m.complete_store(first_keys, ctx)
        # Now try to store 4 NEW (distinct) keys → must evict some.
        second_keys = _make_keys(4, offset=10)
        second_out = m.prepare_store(second_keys, ctx)
        assert second_out is not None
        assert len(second_out.evicted_keys) > 0

    def test_prepare_store_returns_none_when_no_room(self):
        # 2 blocks, no evictions possible → 3-block store fails.
        m = CPUOffloadingManager(num_blocks=2)
        ctx = ReqContext(req_id="r1")
        # Hold 1 block with ref_cnt > 0 (un-evictable).
        keys = _make_keys(2)
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx)
        m.prepare_load(keys[:1], ctx)
        # Now try to store 3 more — only 1 free, but only 1 evictable.
        new_keys = _make_keys(3)
        out = m.prepare_store(new_keys, ctx)
        assert out is None  # Cannot satisfy.

    def test_complete_store_failure_rolls_back(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(2)
        m.prepare_store(keys, ctx)
        assert m._num_write_pending_blocks == 2
        # Fail the store → blocks freed.
        m.complete_store(keys, ctx, success=False)
        assert m._num_write_pending_blocks == 0
        # Lookup returns MISS.
        for k in keys:
            assert m.lookup(k, ctx) == LookupResult.MISS

    def test_store_threshold_tracks_lookups(self):
        m = CPUOffloadingManager(
            num_blocks=8, store_threshold=2, max_tracker_size=100,
        )
        ctx = ReqContext(req_id="r1")
        # First lookup → count 1 (below threshold).
        k = _make_keys(1)[0]
        m.lookup(k, ctx)
        # Threshold ≥ 2 → block not yet eligible.
        out = m.prepare_store([k], ctx)
        assert out is not None
        assert out.keys_to_store == []  # Skipped (count < threshold).
        # Second lookup → count 2 (at threshold) → eligible.
        m.lookup(k, ctx)
        out = m.prepare_store([k], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1

    def test_store_threshold_default_1_eligible_immediately(self):
        # Default threshold=1 → no filter (every block eligible).
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        k = _make_keys(1)[0]
        out = m.prepare_store([k], ctx)
        assert out is not None
        assert len(out.keys_to_store) == 1

    def test_events_disabled_by_default(self):
        m = CPUOffloadingManager(num_blocks=8)
        assert m.events is None
        # take_events yields nothing.
        assert list(m.take_events()) == []

    def test_events_emitted_on_store(self):
        m = CPUOffloadingManager(num_blocks=8, enable_events=True)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(2)
        out = m.prepare_store(keys, ctx)
        assert out is not None
        m.complete_store(keys, ctx)
        events = list(m.take_events())
        # One event per batch: 1 on store completion (2 keys).
        # (No eviction event — no evictions needed.)
        assert len(events) == 1
        assert events[0].removed is False  # stored, not removed

    def test_reset_cache(self):
        m = CPUOffloadingManager(num_blocks=8, enable_events=True)
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(2)
        m.prepare_store(keys, ctx)
        m.complete_store(keys, ctx)
        assert m._num_allocated_blocks == 2
        m.reset_cache()
        assert m._num_allocated_blocks == 0
        assert len(list(m.take_events())) == 0

    def test_on_new_request_default_policy(self):
        m = CPUOffloadingManager(num_blocks=8)
        ctx = ReqContext(req_id="r1")
        out = m.on_new_request(ctx)
        assert isinstance(out, RequestOffloadingContext)
        assert out.policy.value == "block_level"

    def test_get_stats_returns_none(self):
        # Phase 5 wires Prometheus; C256 returns None.
        m = CPUOffloadingManager(num_blocks=8)
        assert m.get_stats() is None


# ===========================================================================
# CPUOffloadingWorker + SingleDirectionOffloadingHandler
# ===========================================================================


class TestCPUOffloadingWorker:
    """The worker-side async transfer handler."""

    def test_constructs(self):
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        # Two handlers (store + load).
        assert w._store_handler is not None
        assert w._load_handler is not None
        w.shutdown()

    def test_submit_store_returns_true(self):
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        from my_vllm.v1.kv_offload.base import GPULoadStoreSpec
        gpu_spec = GPULoadStoreSpec(
            block_ids=[0, 1], group_sizes=[2], block_indices=[0],
        )
        cpu_spec = CPULoadStoreSpec([0, 1])
        assert w.submit_store(job_id=1, src_spec=gpu_spec, dst_spec=cpu_spec)
        w.shutdown()

    def test_submit_load_returns_true(self):
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        from my_vllm.v1.kv_offload.base import GPULoadStoreSpec
        gpu_spec = GPULoadStoreSpec(
            block_ids=[0, 1], group_sizes=[2], block_indices=[0],
        )
        cpu_spec = CPULoadStoreSpec([0, 1])
        assert w.submit_load(job_id=2, src_spec=cpu_spec, dst_spec=gpu_spec)
        w.shutdown()

    def test_get_finished_drains_cpu_path(self):
        """CPU-only path (no CUDA): transfers complete synchronously."""
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        from my_vllm.v1.kv_offload.base import GPULoadStoreSpec
        gpu_spec = GPULoadStoreSpec(
            block_ids=[0, 1], group_sizes=[2], block_indices=[0],
        )
        cpu_spec = CPULoadStoreSpec([0, 1])
        w.submit_store(job_id=1, src_spec=gpu_spec, dst_spec=cpu_spec)
        # On CPU path, get_finished returns the result immediately.
        results = w.get_finished()
        assert len(results) == 1
        assert results[0].job_id == 1
        assert results[0].success
        w.shutdown()

    def test_wait_does_not_crash(self):
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        w.wait(set())
        w.shutdown()

    def test_shutdown_clears_handlers(self):
        kvc = _make_canonical_kvcaches(num_gpu_blocks=4)
        w = CPUOffloadingWorker(
            kv_caches=kvc, blocks_per_chunk=1, num_cpu_blocks=8,
        )
        w.shutdown()
        # After shutdown, internal pools cleared.
        assert len(w._store_handler._transfers) == 0
        assert len(w._load_handler._transfers) == 0


class TestSingleDirectionOffloadingHandler:
    """The per-direction handler."""

    def test_cpu_path_completes_synchronously(self):
        gpu_t = [torch.zeros((2, 16), dtype=torch.int8)]
        cpu_t = [torch.zeros((4, 16), dtype=torch.int8)]
        h = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_t, cpu_tensors=cpu_t, gpu_to_cpu=True,
        )
        h.transfer_async(
            job_id=1,
            src_spec=CPULoadStoreSpec([0]),
            dst_spec=CPULoadStoreSpec([0]),
        )
        results = h.get_finished()
        assert len(results) == 1
        assert results[0].success

    def test_asserts_src_block_ids(self):
        gpu_t = [torch.zeros((2, 16), dtype=torch.int8)]
        cpu_t = [torch.zeros((4, 16), dtype=torch.int8)]
        h = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_t, cpu_tensors=cpu_t, gpu_to_cpu=True,
        )
        # Mismatched len → assertion.
        with pytest.raises(AssertionError, match="1:1 block mapping"):
            h.transfer_async(
                job_id=1,
                src_spec=CPULoadStoreSpec([0, 1]),
                dst_spec=CPULoadStoreSpec([0]),
            )


# ===========================================================================
# CPUOffloadingSpec
# ===========================================================================


class TestCPUOffloadingSpec:
    """The CPU offload connector spec."""

    def test_requires_cpu_bytes_to_use(self):
        cfg = _make_config(extra_config={})
        with pytest.raises(Exception, match="cpu_bytes_to_use"):
            CPUOffloadingSpec(cfg)

    def test_construction_with_cpu_bytes(self):
        cfg = _make_config(
            extra_config={"cpu_bytes_to_use": 1024 * 1024 * 100},  # 100 MB
        )
        spec = CPUOffloadingSpec(cfg)
        # 100 MB / 4096 bytes-per-block = 25600 blocks (rounded).
        assert spec.num_blocks > 0
        assert spec.eviction_policy == "lru"

    def test_eviction_policy_override(self):
        cfg = _make_config(
            extra_config={
                "cpu_bytes_to_use": 1024 * 1024,
                "eviction_policy": "arc",
            },
        )
        spec = CPUOffloadingSpec(cfg)
        assert spec.eviction_policy == "arc"

    def test_get_manager_lazily_constructs(self):
        cfg = _make_config(
            extra_config={"cpu_bytes_to_use": 1024 * 1024},
        )
        spec = CPUOffloadingSpec(cfg)
        assert spec._manager is None
        m = spec.get_manager()
        assert isinstance(m, CPUOffloadingManager)
        # Second call returns the same instance.
        assert spec.get_manager() is m

    def test_get_worker_lazily_constructs(self):
        cfg = _make_config(
            extra_config={"cpu_bytes_to_use": 1024 * 1024},
        )
        spec = CPUOffloadingSpec(cfg)
        kvc = _make_canonical_kvcaches()
        w = spec.get_worker(kvc)
        assert isinstance(w, CPUOffloadingWorker)
        w.shutdown()

    def test_register_via_factory(self):
        # C256 ships CPUOffloadingSpec pre-registered.
        spec_cls = OffloadingSpecFactory.get_spec_cls({})
        # Default name → CPUOffloadingSpec.
        assert spec_cls.__name__ == "CPUOffloadingSpec"


# ===========================================================================
# Tiering
# ===========================================================================


class TestTiering:
    """The tiering composing manager + spec + factory."""

    def test_example_tier_lookup_returns_miss(self):
        tier = ExampleSecondaryTierManager(
            offloading_spec=None,
            primary_kv_view=memoryview(b""),
            tier_type="example",
        )
        result = tier.lookup(
            make_offload_key(b"\x00", 0),
            ReqContext(req_id="r1"),
        )
        assert result == LookupResult.MISS

    def test_example_tier_submit_store_raises(self):
        tier = ExampleSecondaryTierManager(
            offloading_spec=None,
            primary_kv_view=memoryview(b""),
            tier_type="example",
        )
        with pytest.raises(NotImplementedError, match="C256 shell"):
            tier.submit_store(
                JobMetadata(
                    job_id=0,
                    keys=[],
                    block_ids=None,
                    is_promotion=False,
                    req_context=ReqContext(req_id="r1"),
                )
            )

    def test_secondary_factory_example_registered(self):
        assert "example" in SecondaryTierFactory.known_tiers()

    def test_secondary_factory_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown secondary tier type"):
            SecondaryTierFactory.get_tier_class({"type": "mystery"})

    def test_secondary_factory_create(self):
        # The factory pops `type` and forwards the rest as kwargs;
        # ExampleSecondaryTierManager accepts **no** extras.
        tier = SecondaryTierFactory.create_secondary_tier(
            {"type": "example"},
            primary_kv_view=memoryview(b""),
            offloading_spec=None,
        )
        assert isinstance(tier, ExampleSecondaryTierManager)
        assert tier.tier_type == "example"

    def test_cpu_primary_tier_wraps_manager(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        # OffloadingManager delegation.
        ctx = ReqContext(req_id="r1")
        assert primary.lookup(
            make_offload_key(b"\x00", 0), ctx,
        ) == LookupResult.MISS
        # Memoryview accessor returns a memoryview.
        mv = primary.get_kv_memoryview()
        assert isinstance(mv, memoryview)

    def test_tiering_manager_lookup_fanout(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        # Secondary tier that always MISSes.
        secondary = ExampleSecondaryTierManager(
            offloading_spec=None,
            primary_kv_view=memoryview(b""),
            tier_type="example",
        )
        tm = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[secondary],
        )
        ctx = ReqContext(req_id="r1")
        # No primary, no secondary → MISS.
        assert tm.lookup(
            make_offload_key(b"\x00", 0), ctx,
        ) == LookupResult.MISS

    def test_tiering_manager_lookup_primary_wins(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        # Store + complete a block in primary.
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(1)
        primary.prepare_store(keys, ctx)
        primary.complete_store(keys, ctx)
        tm = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[],
        )
        # Primary has it → HIT.
        assert tm.lookup(keys[0], ctx) == LookupResult.HIT

    def test_tiering_manager_offloading_manager_delegation(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        tm = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[],
        )
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(1)
        # on_new_request delegates to primary.
        roc = tm.on_new_request(ctx)
        assert isinstance(roc, RequestOffloadingContext)
        # prepare_store delegates to primary.
        out = tm.prepare_store(keys, ctx)
        assert out is not None
        # complete_store delegates.
        tm.complete_store(keys, ctx)
        # Lookup now HIT.
        assert tm.lookup(keys[0], ctx) == LookupResult.HIT
        # prepare_load / complete_load round-trip.
        spec = tm.prepare_load(keys, ctx)
        assert isinstance(spec, CPULoadStoreSpec)
        tm.complete_load(keys, ctx)

    def test_tiering_manager_has_pending_work_aggregates(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        secondary = ExampleSecondaryTierManager(
            offloading_spec=None,
            primary_kv_view=memoryview(b""),
            tier_type="example",
        )
        tm = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[secondary],
        )
        # Both primary and ExampleSecondary report False.
        assert not tm.has_pending_work()

    def test_tiering_manager_parent_manager_create_store_job(self):
        primary = CPUPrimaryTierOffloadingManager(
            num_blocks=8, cache_policy="lru",
        )
        tm = TieringOffloadingManager(
            primary_tier=primary, secondary_tiers=[],
        )
        ctx = ReqContext(req_id="r1")
        keys = _make_keys(2)
        job = tm.create_store_job(keys, ctx)
        assert isinstance(job, JobMetadata)
        assert list(job.keys) == keys
        assert job.req_context is ctx

    def test_tiering_spec_construction(self):
        cfg = _make_config(
            extra_config={
                "cpu_bytes_to_use": 1024 * 1024,
                "secondary_tiers": [{"type": "example"}],
            },
        )
        spec = TieringOffloadingSpec(cfg)
        assert spec.secondary_tier_configs == [{"type": "example"}]

    def test_tiering_spec_secondary_tiers_must_be_list(self):
        cfg = _make_config(
            extra_config={
                "cpu_bytes_to_use": 1024 * 1024,
                "secondary_tiers": "not a list",
            },
        )
        with pytest.raises(ValueError, match="must be a list"):
            TieringOffloadingSpec(cfg)

    def test_tiering_spec_get_manager(self):
        cfg = _make_config(
            extra_config={
                "cpu_bytes_to_use": 1024 * 1024,
                "secondary_tiers": [{"type": "example"}],
            },
        )
        spec = TieringOffloadingSpec(cfg)
        m = spec.get_manager()
        assert isinstance(m, TieringOffloadingManager)

    def test_tiering_spec_registered_in_factory(self):
        spec_cls = OffloadingSpecFactory.get_spec_cls(
            {"spec_name": "TieringOffloadingSpec"}
        )
        assert spec_cls.__name__ == "TieringOffloadingSpec"


# ===========================================================================
# CPUOffloadingSpec BLOCK_SIZE_ALIGNMENT
# ===========================================================================


class TestSpecConstants:
    """Spec-level constants + edge cases."""

    def test_block_size_alignment(self):
        # Single-group Dense path needs no alignment (1).
        assert CPUOffloadingSpec.BLOCK_SIZE_ALIGNMENT == 1
        assert TieringOffloadingSpec.BLOCK_SIZE_ALIGNMENT == 1

    def test_build_metric_definitions_default(self):
        defs = CPUOffloadingSpec.build_metric_definitions({})
        # All 5 default metrics.
        assert CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC in defs
        assert CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC in defs
        assert CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC in defs
        assert CPUOffloadingMetrics.CPU_ALLOCATION_SIZE in defs
        # STORES_SKIPPED only added with store_threshold >= 2.
        assert CPUOffloadingMetrics.STORES_SKIPPED not in defs

    def test_build_metric_definitions_with_store_threshold(self):
        defs = CPUOffloadingSpec.build_metric_definitions(
            {"store_threshold": 3},
        )
        assert CPUOffloadingMetrics.STORES_SKIPPED in defs

    def test_build_metric_definitions_tiering_inherits(self):
        # TieringOffloadingSpec inherits from CPUOffloadingSpec (no
        # override of build_metric_definitions in my-vllm).
        defs = TieringOffloadingSpec.build_metric_definitions({})
        assert CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC in defs
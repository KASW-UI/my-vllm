"""tests/test_C251_kv_connector_base.py — Tests for the extended KV connector base.

C251: KV ownership / metadata lifecycle / failure retransmission /
partial transfer / KV index.

Verifies:

  1. Module imports — every name in __all__ is reachable.
  2. KVIndex — add / remove / lookup / state updates / cleanup.
  3. KVOwnership + KVOwnershipState — lifecycle states.
  4. KVFailureTracker — record / extend / clear / prune.
  5. KVConnectorBase_V1 metadata lifecycle — bind / clear / has.
  6. KVConnectorBase_V1 KV cache registration — register_kv_caches /
     register_cross_layers_kv_cache / set_host_xfer_buffer_ops.
  7. KVConnectorBase_V1 block pool binding — bind_gpu_block_pool.
  8. KVConnectorBase_V1 default request lifecycle — update_state_after_alloc
     registers blocks; request_finished removes them.
  9. KVConnectorBase_V1 failure API — get_block_ids_with_load_errors /
     get_finished / get_finished_count.
 10. KVConnectorBase_V1 handshake metadata — set / get round-trip.
 11. KVConnectorBase_V1 lifecycle — shutdown / reset_cache.
 12. NoopConnector — construction + identity behavior.
 13. KVTransferMethod enum.
 14. Backward compat — C212 PD example + C250 NIXL imports.

Run:
    pytest tests/test_C251_kv_connector_base.py -v
"""

from __future__ import annotations

import time

import pytest
import torch

from my_vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    KVFailureTracker,
    KVIndex,
    KVIndexEntry,
    KVOwnership,
    KVOwnershipState,
    KVTransferFailure,
    KVTransferMethod,
    NoopConnector,
    RecvResult,
    SupportsHMA,
    TransferBundle,
    TransferStatus,
    supports_hma,
)


# ---------------------------------------------------------------------------
# Helpers — minimal concrete subclass for testing the ABC
# ---------------------------------------------------------------------------


class _ConcreteConnector(KVConnectorBase_V1):
    """Minimal concrete subclass used to exercise the default
    implementations on the ABC."""

    def __init__(self, role=KVConnectorRole.SCHEDULER) -> None:
        super().__init__(vllm_config=None, role=role, kv_cache_config=None)
        self.saved_layers: list[str] = []
        self.loaded_layers: list[str] = []

    def start_load_kv(self, forward_context=None, **kwargs):
        pass

    def save_kv_layer(
        self, layer_name, kv_layer, attn_metadata=None, **kwargs,
    ):
        self.saved_layers.append(layer_name)

    def wait_for_layer_load(self, layer_name):
        self.loaded_layers.append(layer_name)

    def wait_for_save(self):
        pass


# ---------------------------------------------------------------------------
# Test 1: module imports + names
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_kv_index_import(self):
        assert KVIndex is not None
        assert KVIndexEntry is not None

    def test_kv_ownership_import(self):
        assert KVOwnership is not None
        assert KVOwnershipState is not None

    def test_kv_failure_tracker_import(self):
        assert KVFailureTracker is not None
        assert KVTransferFailure is not None

    def test_kv_transfer_method_enum(self):
        assert KVTransferMethod.SHARED_MEMORY.value == "shared_memory"
        assert KVTransferMethod.NCCL.value == "nccl"
        assert KVTransferMethod.RDMA.value == "rdma"


# ---------------------------------------------------------------------------
# Test 2: KVIndex
# ---------------------------------------------------------------------------


class TestKVIndex:
    def test_empty_index(self):
        idx = KVIndex()
        assert len(idx) == 0
        assert idx.get_blocks_for_request("req1") == []
        assert idx.get_request_for_block(0) is None

    def test_add_blocks(self):
        idx = KVIndex()
        idx.add("req1", [10, 11, 12])
        assert len(idx) == 3
        assert idx.get_blocks_for_request("req1") == [10, 11, 12]
        assert idx.get_request_for_block(10) == "req1"

    def test_add_multiple_requests(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        idx.add("req2", [4, 5, 6])
        assert len(idx) == 6
        assert idx.get_blocks_for_request("req1") == [1, 2, 3]
        assert idx.get_blocks_for_request("req2") == [4, 5, 6]

    def test_remove_request(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        idx.add("req2", [4, 5])
        removed = idx.remove("req1")
        assert sorted(removed) == [1, 2, 3]
        assert idx.get_blocks_for_request("req1") == []
        assert idx.get_request_for_block(1) is None
        # req2 still intact.
        assert idx.get_blocks_for_request("req2") == [4, 5]

    def test_remove_block(self):
        idx = KVIndex()
        idx.add("req1", [10, 11, 12])
        owner = idx.remove_block(11)
        assert owner == "req1"
        assert 11 not in idx
        assert idx.get_blocks_for_request("req1") == [10, 12]

    def test_update_state_all(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        n = idx.update_state("req1", KVOwnershipState.OWNED)
        assert n == 3
        assert idx.get_state("req1", 1) == KVOwnershipState.OWNED

    def test_update_state_subset(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        n = idx.update_state(
            "req1", KVOwnershipState.TRANSFERRING, block_ids=[1, 2],
        )
        assert n == 2
        assert idx.get_state("req1", 1) == KVOwnershipState.TRANSFERRING
        assert idx.get_state("req1", 2) == KVOwnershipState.TRANSFERRING
        assert idx.get_state("req1", 3) == KVOwnershipState.ALLOCATED

    def test_get_blocks_in_state(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        idx.add("req2", [4, 5])
        idx.update_state("req1", KVOwnershipState.OWNED, block_ids=[1])
        idx.update_state("req2", KVOwnershipState.OWNED)
        pairs = idx.get_blocks_in_state(KVOwnershipState.OWNED)
        # Sort by (request_id, block_id) for deterministic ordering.
        pairs_sorted = sorted(pairs)
        assert ("req1", 1) in pairs_sorted
        assert ("req2", 4) in pairs_sorted
        assert ("req2", 5) in pairs_sorted

    def test_contains(self):
        idx = KVIndex()
        idx.add("req1", [10, 11])
        assert 10 in idx
        assert 12 not in idx

    def test_clear(self):
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        idx.clear()
        assert len(idx) == 0

    def test_update_state_unknown_request(self):
        idx = KVIndex()
        # Unknown request returns 0 updated.
        n = idx.update_state("nope", KVOwnershipState.OWNED)
        assert n == 0


# ---------------------------------------------------------------------------
# Test 3: KVOwnership + KVOwnershipState
# ---------------------------------------------------------------------------


class TestKVOwnership:
    def test_construction(self):
        own = KVOwnership(
            request_id="req1",
            local_block_ids=[1, 2, 3],
            owner_engine_id="engine-0",
            state="owned",
            created_at=time.monotonic(),
        )
        assert own.request_id == "req1"
        assert own.local_block_ids == [1, 2, 3]
        assert own.owner_engine_id == "engine-0"

    def test_state_enum_values(self):
        assert KVOwnershipState.ALLOCATED.value == "allocated"
        assert KVOwnershipState.OWNED.value == "owned"
        assert KVOwnershipState.TRANSFERRING.value == "transferring"
        assert KVOwnershipState.LEASED.value == "leased"
        assert KVOwnershipState.EXPIRED.value == "expired"
        assert KVOwnershipState.FREED.value == "freed"

    def test_state_transitions(self):
        # Verify the lifecycle is sensible.
        states = [
            KVOwnershipState.ALLOCATED,
            KVOwnershipState.OWNED,
            KVOwnershipState.TRANSFERRING,
            KVOwnershipState.LEASED,
            KVOwnershipState.EXPIRED,
            KVOwnershipState.FREED,
        ]
        assert len(states) == 6


# ---------------------------------------------------------------------------
# Test 4: KVFailureTracker
# ---------------------------------------------------------------------------


class TestKVFailureTracker:
    def test_empty_tracker(self):
        t = KVFailureTracker()
        assert len(t) == 0
        assert t.get_failure("req1") is None
        assert not t.has_failure("req1")

    def test_record_failure(self):
        t = KVFailureTracker()
        t.record_failure(
            "req1", failed_block_ids=[1, 2, 3],
            num_completed_tokens=128, error_message="timeout",
        )
        assert t.has_failure("req1")
        assert len(t) == 1
        f = t.get_failure("req1")
        assert f is not None
        assert f.failed_block_ids == [1, 2, 3]
        assert f.num_completed_tokens == 128
        assert f.error_message == "timeout"

    def test_record_failure_extends(self):
        t = KVFailureTracker()
        t.record_failure("req1", failed_block_ids=[1, 2])
        t.record_failure("req1", failed_block_ids=[3, 4])
        f = t.get_failure("req1")
        assert f is not None
        # Extended, not overwritten.
        assert f.failed_block_ids == [1, 2, 3, 4]

    def test_record_failure_updates_completed_tokens(self):
        t = KVFailureTracker()
        t.record_failure("req1", failed_block_ids=[1], num_completed_tokens=64)
        t.record_failure("req1", failed_block_ids=[2], num_completed_tokens=128)
        f = t.get_failure("req1")
        # Take the max.
        assert f is not None
        assert f.num_completed_tokens == 128

    def test_clear(self):
        t = KVFailureTracker()
        t.record_failure("req1", failed_block_ids=[1])
        t.clear("req1")
        assert not t.has_failure("req1")

    def test_prune_stale(self):
        t = KVFailureTracker(max_age_seconds=0.0)
        t.record_failure("req1", failed_block_ids=[1])
        # All entries older than 0s → pruned.
        n = t.prune_stale()
        assert n == 1
        assert not t.has_failure("req1")


# ---------------------------------------------------------------------------
# Test 5: KVConnectorBase_V1 metadata lifecycle
# ---------------------------------------------------------------------------


class TestMetadataLifecycle:
    def test_construction_defaults(self):
        c = _ConcreteConnector()
        assert c._connector_metadata is None
        assert not c.has_connector_metadata()

    def test_bind_clear_metadata(self):
        c = _ConcreteConnector()

        class FakeMeta(KVConnectorMetadata):
            pass

        meta = FakeMeta()
        c.bind_connector_metadata(meta)
        assert c.has_connector_metadata()
        assert c._get_connector_metadata() is meta

        c.clear_connector_metadata()
        assert not c.has_connector_metadata()
        assert c._get_connector_metadata() is None


# ---------------------------------------------------------------------------
# Test 6: KVConnectorBase_V1 KV cache registration
# ---------------------------------------------------------------------------


class TestKVCacheRegistration:
    def test_register_kv_caches(self):
        c = _ConcreteConnector()
        cache = torch.empty(4, 16, 8, dtype=torch.float32)
        c.register_kv_caches({"layer0": cache})
        assert "layer0" in c._kv_caches
        assert c._kv_caches["layer0"] is cache

    def test_register_cross_layers_kv_cache(self):
        c = _ConcreteConnector()
        cache = torch.empty(12, 4, 16, 8, dtype=torch.float32)
        c.register_cross_layers_kv_cache(cache)
        assert "cross_layers" in c._kv_caches

    def test_set_host_xfer_buffer_ops(self):
        c = _ConcreteConnector()

        def fake_op(s, d, si, di, direction):
            pass

        c.set_host_xfer_buffer_ops(fake_op)
        assert c._copy_op is fake_op


# ---------------------------------------------------------------------------
# Test 7: KVConnectorBase_V1 block pool binding
# ---------------------------------------------------------------------------


class TestBlockPoolBinding:
    def test_bind_gpu_block_pool(self):
        c = _ConcreteConnector()
        fake_pool = object()
        c.bind_gpu_block_pool(fake_pool)
        assert c._gpu_block_pool is fake_pool

    def test_handle_preemptions_noop(self):
        c = _ConcreteConnector()
        # Should not raise.
        c.handle_preemptions(None)


# ---------------------------------------------------------------------------
# Test 8: KVConnectorBase_V1 default request lifecycle
# ---------------------------------------------------------------------------


class TestRequestLifecycle:
    def test_update_state_after_alloc_registers_blocks(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [10, 11, 12], num_external_tokens=0)
        assert c._kv_index.get_blocks_for_request("req1") == [10, 11, 12]

    def test_update_state_after_alloc_no_blocks(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [], num_external_tokens=0)
        # No blocks → no entry.
        assert c._kv_index.get_blocks_for_request("req1") == []

    def test_request_finished_cleans_up(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [10, 11, 12], num_external_tokens=0)
        # Add a failure too.
        c._failure_tracker.record_failure("req1", failed_block_ids=[11])
        c._finished_req_ids.add("req1")
        c._failed_req_ids.add("req1")

        c.request_finished(fake_request, None)
        assert c._kv_index.get_blocks_for_request("req1") == []
        assert not c._failure_tracker.has_failure("req1")
        assert "req1" not in c._finished_req_ids
        assert "req1" not in c._failed_req_ids

    def test_request_finished_with_explicit_block_ids(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [10, 11, 12], num_external_tokens=0)
        c.request_finished(fake_request, [10, 11])
        # block 12 still in index (caller didn't list it).
        # But the request_id key is dropped because we removed
        # the listed blocks and the index is empty for this req.
        # Actually: my-vllm implementation only drops the request key
        # if no more entries. block 12 remains → request key remains.
        assert c._kv_index.get_blocks_for_request("req1") == [12]


# ---------------------------------------------------------------------------
# Test 9: KVConnectorBase_V1 failure API
# ---------------------------------------------------------------------------


class TestFailureAPI:
    def test_get_block_ids_with_load_errors_empty(self):
        c = _ConcreteConnector()
        assert c.get_block_ids_with_load_errors() == set()

    def test_get_block_ids_with_load_errors_populated(self):
        c = _ConcreteConnector()
        c._failed_req_ids.add("req1")
        c._failure_tracker.record_failure("req1", failed_block_ids=[1, 2, 3])
        c._failed_req_ids.add("req2")
        c._failure_tracker.record_failure("req2", failed_block_ids=[10])
        errors = c.get_block_ids_with_load_errors()
        assert errors == {1, 2, 3, 10}

    def test_get_finished(self):
        c = _ConcreteConnector()
        c._finished_req_ids = {"req1", "req2"}
        c._failed_req_ids = {"req3"}
        done, failed = c.get_finished()
        assert done == {"req1", "req2"}
        assert failed == {"req3"}
        # Cleared after read.
        assert c._finished_req_ids == set()
        assert c._failed_req_ids == set()

    def test_get_finished_count(self):
        c = _ConcreteConnector()
        c._finished_req_ids = {"req1", "req2"}
        c._failed_req_ids = {"req3"}
        assert c.get_finished_count() == 3


# ---------------------------------------------------------------------------
# Test 10: KVConnectorBase_V1 handshake metadata
# ---------------------------------------------------------------------------


class TestHandshakeMetadata:
    def test_default_is_none(self):
        c = _ConcreteConnector()
        assert c.get_handshake_metadata() is None

    def test_set_get_roundtrip(self):
        c = _ConcreteConnector()

        class FakeHS(KVConnectorHandshakeMetadata):
            pass

        hs = FakeHS()
        c.set_xfer_handshake_metadata(hs)
        assert c.get_handshake_metadata() is hs


# ---------------------------------------------------------------------------
# Test 11: KVConnectorBase_V1 lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_shutdown(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [1, 2, 3], num_external_tokens=0)
        c._failure_tracker.record_failure("req1", failed_block_ids=[1])
        c._finished_req_ids.add("req1")
        c._connector_metadata = object()
        c._xfer_handshake_metadata = object()
        c.register_kv_caches({"k": torch.empty(0)})

        c.shutdown()

        assert len(c._kv_index) == 0
        assert len(c._failure_tracker) == 0
        assert c._finished_req_ids == set()
        assert c._connector_metadata is None
        assert c._xfer_handshake_metadata is None
        assert c._kv_caches == {}

    def test_reset_cache_returns_true(self):
        c = _ConcreteConnector()
        assert c.reset_cache() is True

    def test_reset_cache_clears(self):
        c = _ConcreteConnector()
        fake_request = type("Req", (), {"request_id": "req1"})()
        c.update_state_after_alloc(fake_request, [1, 2, 3], num_external_tokens=0)
        c.reset_cache()
        assert len(c._kv_index) == 0

    def test_get_kv_connector_stats_default_none(self):
        c = _ConcreteConnector()
        assert c.get_kv_connector_stats() is None

    def test_get_required_kv_cache_layout_default_none(self):
        assert _ConcreteConnector.get_required_kv_cache_layout(None) is None

    def test_prefer_cross_layer_blocks_default_false(self):
        c = _ConcreteConnector()
        assert c.prefer_cross_layer_blocks is False


# ---------------------------------------------------------------------------
# Test 12: NoopConnector
# ---------------------------------------------------------------------------


class TestNoopConnector:
    def test_construction(self):
        c = NoopConnector()
        assert c.role == KVConnectorRole.SCHEDULER

    def test_construction_explicit_role(self):
        c = NoopConnector(role=KVConnectorRole.WORKER)
        assert c.role == KVConnectorRole.WORKER

    def test_all_ops_noop(self):
        c = NoopConnector()
        c.start_load_kv()
        c.save_kv_layer("layer0", torch.empty(0))
        c.wait_for_layer_load("layer0")
        c.wait_for_save()
        # Should not raise.
        result = c.check_transfer_done()
        assert result.status == TransferStatus.SUCCESS
        # No acks by default.
        assert c.drain_acks() == {}

    def test_supports_hma_returns_false(self):
        assert supports_hma(NoopConnector()) is False
        assert supports_hma(NoopConnector) is False


# ---------------------------------------------------------------------------
# Test 13: KVTransferMethod enum
# ---------------------------------------------------------------------------


class TestKVTransferMethod:
    def test_members(self):
        members = list(KVTransferMethod)
        assert len(members) == 3

    def test_values(self):
        assert KVTransferMethod("shared_memory") is KVTransferMethod.SHARED_MEMORY
        assert KVTransferMethod("nccl") is KVTransferMethod.NCCL
        assert KVTransferMethod("rdma") is KVTransferMethod.RDMA


# ---------------------------------------------------------------------------
# Test 14: backward compat — C212 + C250 still work
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_c250_nixl_connector_still_imports(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
            NixlConnector,
            NixlPullConnector,
            NixlPushConnector,
            compute_tp_mapping,
        )
        assert NixlConnector is NixlPullConnector
        assert NixlPullConnector is not None
        assert NixlPushConnector is not None

    def test_c212_example_connector_inherits(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
            ExampleConnector,
        )
        assert issubclass(ExampleConnector, KVConnectorBase_V1)

    def test_nixl_connector_uses_kv_index(self):
        """NIXL connector (C250) inherits the new KVIndex."""
        from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
            NixlConnector,
        )
        c = NixlConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        assert c.kv_index is not None
        assert isinstance(c.kv_index, KVIndex)
        assert c.failure_tracker is not None
        assert isinstance(c.failure_tracker, KVFailureTracker)

    def test_nixl_connector_register_kv_caches(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
            NixlConnector,
        )
        c = NixlConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        cache = torch.empty(4, 16, 8, dtype=torch.float32)
        c.register_kv_caches({"layer0": cache})
        assert "layer0" in c._kv_caches


# ---------------------------------------------------------------------------
# Test 15: KVConnectorWorkerMetadata aggregation (existing ABC contract)
# ---------------------------------------------------------------------------


class TestWorkerMetadataAggregation:
    def test_aggregate_subclass(self):
        class MyWorkerMeta(KVConnectorWorkerMetadata):
            def __init__(self, value):
                self.value = value

            def aggregate(self, other):
                return MyWorkerMeta(self.value + other.value)

        a = MyWorkerMeta(1)
        b = MyWorkerMeta(2)
        c = a.aggregate(b)
        assert c.value == 3


# ---------------------------------------------------------------------------
# Test 16: TransferBundle / RecvResult (existing dataclasses)
# ---------------------------------------------------------------------------


class TestTransferBundleRecvResult:
    def test_transfer_bundle(self):
        b = TransferBundle(
            request_id="req1",
            block_ids=[1, 2, 3],
            num_tokens=64,
        )
        assert b.request_id == "req1"
        assert b.block_ids == [1, 2, 3]

    def test_recv_result_defaults(self):
        r = RecvResult()
        assert r.status == TransferStatus.SUCCESS
        assert r.bundles == []
        assert r.invalid_req_ids == set()
        assert r.valid_prefix_tokens == {}

    def test_recv_result_partial_transfer(self):
        # Partial transfer: req1 failed at token 64, req2 succeeded.
        r = RecvResult(
            status=TransferStatus.TIMEOUT,
            bundles=[],
            invalid_req_ids={"req1"},
            valid_prefix_tokens={"req1": 64},
        )
        assert r.valid_prefix_tokens["req1"] == 64

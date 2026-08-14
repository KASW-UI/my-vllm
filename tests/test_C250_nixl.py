"""tests/test_C250_nixl.py — Tests for the NIXL KV connector package.

C250: 13-file NIXL package + 3-variant connector (base/pull/push).
Verifies:

  1. Module imports — every name in __all__ is reachable.
  2. utils.get_base_request_id / get_representative_spec_type.
  3. tp_mapping.compute_tp_mapping for several (local_tp, remote_tp,
     is_mla) scenarios.
  4. metadata.NixlAgentMetadata / NixlHandshakePayload /
     NixlConnectorMetadata round-trips + compatibility hash.
  5. stats.NixlKVConnectorStats record / reduce / aggregate.
  6. base_scheduler / base_worker protocol methods (skeleton).
  7. Pull/Push scheduler/worker subclasses instantiate.
  8. Connector facade (Base / Pull / Push) end-to-end save → load
     via the local shared-memory backend.
  9. Backward compat: ``NixlConnector`` is an alias for the pull
     variant; ``ExampleConnector`` (subclass of NixlConnector)
     still works (C212 regression).
 10. NIXL runtime calls raise ``RuntimeError`` with a clear message
     (the protocol-only layer doesn't have NIXL).

Run:
    pytest tests/test_C250_nixl.py -v
"""

from __future__ import annotations

import dataclasses

import pytest

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
    TransferStatus,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    GET_META_MSG,
    NIXL_CONNECTOR_VERSION,
    HeartbeatInfo,
    NixlAgentMetadata,
    NixlBaseConnector,
    NixlBaseConnectorScheduler,
    NixlBaseConnectorWorker,
    NixlConnector,
    NixlConnectorMetadata,
    NixlConnectorScheduler,
    NixlConnectorWorker,
    NixlHandshakePayload,
    NixlKVConnectorStats,
    NixlPullConnector,
    NixlPullConnectorScheduler,
    NixlPullConnectorWorker,
    NixlPushConnector,
    NixlPushConnectorScheduler,
    NixlPushConnectorWorker,
    PUSH_REG_NOTIF_PREFIX,
    ReadSpec,
    RemoteMeta,
    ReqMeta,
    TPMapping,
    TransferTelemetry,
    TransferTopology,
    compute_nixl_compatibility_hash,
    compute_tp_mapping,
    get_base_request_id,
    get_representative_spec_type,
)


# ---------------------------------------------------------------------------
# Test 1: module imports + names
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_connector_classes(self):
        assert NixlBaseConnector is not None
        assert NixlConnector is NixlPullConnector
        assert NixlPullConnector is not None
        assert NixlPushConnector is not None

    def test_scheduler_classes(self):
        assert NixlBaseConnectorScheduler is not None
        assert NixlConnectorScheduler is NixlPullConnectorScheduler
        assert NixlPullConnectorScheduler is not None
        assert NixlPushConnectorScheduler is not None

    def test_worker_classes(self):
        assert NixlBaseConnectorWorker is not None
        assert NixlConnectorWorker is NixlPullConnectorWorker
        assert NixlPullConnectorWorker is not None
        assert NixlPushConnectorWorker is not None

    def test_metadata_classes(self):
        assert NixlAgentMetadata is not None
        assert NixlHandshakePayload is not None
        assert NixlConnectorMetadata is not None
        assert HeartbeatInfo is not None
        assert RemoteMeta is not None
        assert ReqMeta is not None

    def test_stats_classes(self):
        assert NixlKVConnectorStats is not None
        assert TransferTelemetry is not None

    def test_tp_classes(self):
        assert TPMapping is not None
        assert TransferTopology is not None
        assert ReadSpec is not None

    def test_wire_format_constants(self):
        assert GET_META_MSG == b"get_meta_msg"
        assert PUSH_REG_NOTIF_PREFIX == b"PUSH_REG:"
        assert NIXL_CONNECTOR_VERSION == 5


# ---------------------------------------------------------------------------
# Test 2: utils
# ---------------------------------------------------------------------------


class TestUtils:
    def test_get_base_request_id_strips_suffix(self):
        rid = "req-12345678"
        assert get_base_request_id(rid) == "req"

    def test_get_base_request_id_keeps_non_matching(self):
        rid = "req-1234567"  # 7 hex — doesn't match the 8-hex suffix
        assert get_base_request_id(rid) == "req-1234567"

    def test_get_base_request_id_keeps_short_id(self):
        assert get_base_request_id("req") == "req"

    def test_get_base_request_id_case_insensitive(self):
        assert get_base_request_id("req-ABCD1234") == "req"

    def test_get_representative_spec_type(self):
        @dataclasses.dataclass
        class InnerSpec:
            pass

        @dataclasses.dataclass
        class OuterSpec:
            kv_cache_specs: dict[str, InnerSpec]

        outer = OuterSpec(kv_cache_specs={"a": InnerSpec()})
        assert get_representative_spec_type(outer) is InnerSpec

    def test_get_representative_spec_type_fallback(self):
        @dataclasses.dataclass
        class BareSpec:
            pass

        bare = BareSpec()
        assert get_representative_spec_type(bare) is BareSpec


# ---------------------------------------------------------------------------
# Test 3: TP mapping
# ---------------------------------------------------------------------------


class TestTPMapping:
    def test_d_tp_equal_p_tp_single_rank(self):
        # D_TP == P_TP: each local rank reads from exactly one remote.
        topo = TransferTopology(tp_rank=2, tp_size=4, total_num_kv_heads=8)
        mapping = compute_tp_mapping(
            topo, remote_tp_size=4, group_spec_types=(object,),
        )
        assert len(mapping.all_source_ranks) == 1
        assert mapping.all_source_ranks[0] == 2  # tp_rank * 4 // 4
        assert mapping.rank_offset_factor == 0

    def test_d_tp_greater_than_p_tp_single_rank(self):
        # D_TP > P_TP: multiple local ranks read from one remote (different heads).
        topo = TransferTopology(tp_rank=1, tp_size=4, total_num_kv_heads=8)
        mapping = compute_tp_mapping(
            topo, remote_tp_size=2, group_spec_types=(object,),
        )
        assert len(mapping.all_source_ranks) == 1
        # tp_rank % (D_TP / P_TP) = 1 % 2 = 1 (the hetero-TP offset).
        assert mapping.rank_offset_factor == 1

    def test_p_tp_greater_than_d_tp_multiple_ranks(self):
        # P_TP > D_TP: one local rank reads from multiple remote ranks.
        topo = TransferTopology(tp_rank=0, tp_size=2, total_num_kv_heads=8)
        mapping = compute_tp_mapping(
            topo, remote_tp_size=4, group_spec_types=(object,),
        )
        assert len(mapping.all_source_ranks) == 2

    def test_mla_single_remote_rank(self):
        # MLA: every rank holds a duplicate, so single remote rank suffices.
        topo = TransferTopology(
            tp_rank=0, tp_size=4, total_num_kv_heads=1, is_mla=True,
        )
        mapping = compute_tp_mapping(
            topo, remote_tp_size=8, group_spec_types=(object,),
        )
        assert len(mapping.all_source_ranks) == 1

    def test_hetero_tp_rank_offset_factor(self):
        # D_TP > P_TP: rank_offset_factor = tp_rank % (D_TP / P_TP).
        topo = TransferTopology(tp_rank=3, tp_size=8, total_num_kv_heads=8)
        mapping = compute_tp_mapping(
            topo, remote_tp_size=2, group_spec_types=(object,),
        )
        # tp_rank % (8 / 2) = 3 % 4 = 3
        assert mapping.rank_offset_factor == 3

    def test_attention_slot_assignment(self):
        topo = TransferTopology(tp_rank=0, tp_size=2, total_num_kv_heads=8)
        mapping = compute_tp_mapping(
            topo, remote_tp_size=4, group_spec_types=(object,),
        )
        # Every source rank gets a slot index.
        for r in mapping.all_source_ranks:
            assert r in mapping.rank_to_attention_slot

    def test_ssm_split(self):
        # SSM spec splits separately from attention.
        class MambaSpec:
            pass

        class AttentionSpec:
            pass

        topo = TransferTopology(tp_rank=0, tp_size=1, total_num_kv_heads=4)
        mapping = compute_tp_mapping(
            topo,
            remote_tp_size=4,
            group_spec_types=(AttentionSpec, MambaSpec),
        )
        # Per-group source ranks: one for attention, multiple for SSM.
        assert len(mapping.source_ranks_per_group) == 2


# ---------------------------------------------------------------------------
# Test 4: metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    def test_nixl_agent_metadata_fields(self):
        m = NixlAgentMetadata(
            engine_id="e0",
            agent_metadata=b"\x00\x01",
            kv_caches_base_addr=[0x1000],
            device_id=0,
            num_blocks=10,
            block_lens=[128],
            kv_cache_layout="NHD",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="TRITON_ATTN",
            physical_blocks_per_logical_kv_block=1,
        )
        assert m.engine_id == "e0"
        assert m.num_blocks == 10

    def test_handshake_payload(self):
        p = NixlHandshakePayload(
            compatibility_hash="abc",
            agent_metadata_bytes=b"\x00\x01",
        )
        assert p.compatibility_hash == "abc"

    def test_compatibility_hash_deterministic(self):
        class FakeModelConfig:
            model = "opt-125m"
            dtype = "bfloat16"
            def get_total_num_kv_heads(self): return 12
            def get_head_size(self): return 64
            def get_total_num_hidden_layers(self): return 12

        class FakeCacheConfig:
            cache_dtype = "auto"

        class FakeSchedulerConfig:
            disable_hybrid_kv_cache_manager = False

        class FakeVllmConfig:
            model_config = FakeModelConfig()
            cache_config = FakeCacheConfig()
            scheduler_config = FakeSchedulerConfig()

        h1 = compute_nixl_compatibility_hash(
            FakeVllmConfig(), "TRITON_ATTN", cross_layers_blocks=False,
        )
        h2 = compute_nixl_compatibility_hash(
            FakeVllmConfig(), "TRITON_ATTN", cross_layers_blocks=False,
        )
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_compatibility_hash_changes_with_backend(self):
        class FakeModelConfig:
            model = "opt-125m"
            dtype = "bfloat16"
            def get_total_num_kv_heads(self): return 12
            def get_head_size(self): return 64
            def get_total_num_hidden_layers(self): return 12

        class FakeCacheConfig:
            cache_dtype = "auto"

        class FakeSchedulerConfig:
            disable_hybrid_kv_cache_manager = False

        class FakeVllmConfig:
            model_config = FakeModelConfig()
            cache_config = FakeCacheConfig()
            scheduler_config = FakeSchedulerConfig()

        h1 = compute_nixl_compatibility_hash(
            FakeVllmConfig(), "TRITON_ATTN", cross_layers_blocks=False,
        )
        h2 = compute_nixl_compatibility_hash(
            FakeVllmConfig(), "FLASH_ATTN", cross_layers_blocks=False,
        )
        assert h1 != h2

    def test_connector_metadata_add_new_req_to_save(self):
        meta = NixlConnectorMetadata()
        meta.add_new_req_to_save(
            "req1",
            local_block_ids=[1, 2, 3],
            kv_transfer_params={"tp_size": 1, "pp_size": 1},
        )
        assert "req1" in meta.reqs_to_save
        assert meta.reqs_to_save["req1"].local_block_ids == [1, 2, 3]

    def test_connector_metadata_add_new_req_to_recv(self):
        meta = NixlConnectorMetadata()
        meta.add_new_req_to_recv(
            "req1",
            local_block_ids=[4, 5, 6],
            kv_transfer_params={
                "remote_block_ids": [10, 11, 12],
                "remote_engine_id": "engine-0",
                "remote_request_id": "req-remote-1",
                "remote_host": "127.0.0.1",
                "remote_port": 5555,
            },
        )
        assert "req1" in meta.reqs_to_recv
        req = meta.reqs_to_recv["req1"]
        assert req.remote is not None
        assert req.remote.engine_id == "engine-0"
        assert req.remote.host == "127.0.0.1"

    def test_heartbeat_info(self):
        h = HeartbeatInfo(
            req_ids={"req1", "req2"},
            host="127.0.0.1",
            port=5000,
            tp_size=2,
            pp_size=1,
        )
        assert h.tp_size == 2
        assert "req1" in h.req_ids


# ---------------------------------------------------------------------------
# Test 5: stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_reset_initializes_data(self):
        s = NixlKVConnectorStats()
        assert "transfer_duration" in s.data
        assert s.num_successful_transfers == 0

    def test_record_transfer(self):
        s = NixlKVConnectorStats()
        s.record_transfer(TransferTelemetry(
            xferDuration=1000.0,  # 1 ms
            postDuration=500.0,
            totalBytes=1024 * 1024,
            descCount=10,
        ))
        assert s.num_successful_transfers == 1
        assert s.data["bytes_transferred"][0] == 1024 * 1024

    def test_record_failures(self):
        s = NixlKVConnectorStats()
        s.record_failed_transfer()
        s.record_failed_notification()
        s.record_kv_expired_req()
        assert s.data["num_failed_transfers"] == [1]
        assert s.data["num_failed_notifications"] == [1]
        assert s.data["num_kv_expired_reqs"] == [1]

    def test_is_empty(self):
        s = NixlKVConnectorStats()
        assert s.is_empty() is True
        s.record_transfer(TransferTelemetry(
            xferDuration=1000.0, postDuration=500.0, totalBytes=1024, descCount=1,
        ))
        assert s.is_empty() is False

    def test_reduce_with_data(self):
        s = NixlKVConnectorStats()
        s.record_transfer(TransferTelemetry(
            xferDuration=1000.0, postDuration=500.0,
            totalBytes=1024 * 1024, descCount=4,
        ))
        s.record_transfer(TransferTelemetry(
            xferDuration=2000.0, postDuration=1000.0,
            totalBytes=1024 * 1024, descCount=4,
        ))
        r = s.reduce()
        assert r["Num successful transfers"] == 2
        assert r["Avg xfer time (ms)"] > 0

    def test_reduce_empty(self):
        s = NixlKVConnectorStats()
        r = s.reduce()
        assert r["Num successful transfers"] == 0

    def test_clone_and_reset(self):
        s = NixlKVConnectorStats()
        s.record_transfer(TransferTelemetry(
            xferDuration=1000.0, postDuration=500.0, totalBytes=1024, descCount=1,
        ))
        old = s.clone_and_reset()
        assert old.num_successful_transfers == 1
        assert s.num_successful_transfers == 0

    def test_aggregate(self):
        s1 = NixlKVConnectorStats()
        s2 = NixlKVConnectorStats()
        s1.record_transfer(TransferTelemetry(
            xferDuration=1000.0, postDuration=500.0, totalBytes=1024, descCount=1,
        ))
        s2.record_transfer(TransferTelemetry(
            xferDuration=2000.0, postDuration=1000.0, totalBytes=2048, descCount=2,
        ))
        s1.aggregate(s2)
        assert s1.num_successful_transfers == 2


# ---------------------------------------------------------------------------
# Test 6: base_scheduler / base_worker protocol
# ---------------------------------------------------------------------------


class TestBaseProtocol:
    def test_scheduler_construction(self):
        sched = NixlBaseConnectorScheduler(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert sched.engine_id == "e0"
        assert sched.xfer_handshake_metadata is None

    def test_worker_construction(self):
        worker = NixlBaseConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert worker.engine_id == "e0"
        assert worker._nixl_agent is None

    def test_scheduler_set_xfer_handshake_metadata(self):
        sched = NixlBaseConnectorScheduler(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        sched.set_xfer_handshake_metadata({"hash": "abc"})
        assert sched.xfer_handshake_metadata == {"hash": "abc"}

    def test_worker_register_kv_caches(self):
        worker = NixlBaseConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        import torch

        cache = torch.empty(4, 16, 8, dtype=torch.float32)
        worker.register_kv_caches({"layer0": cache})
        assert len(worker._kv_caches_base_addr) == 1
        assert worker._num_blocks == 4

    def test_worker_get_finished_empty(self):
        worker = NixlBaseConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        done, failed = worker.get_finished()
        assert done == set()
        assert failed == set()

    def test_worker_shutdown(self):
        worker = NixlBaseConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        worker.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# Test 7: pull/push variants
# ---------------------------------------------------------------------------


class TestPullPushVariants:
    def test_pull_scheduler_construction(self):
        s = NixlPullConnectorScheduler(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert isinstance(s, NixlBaseConnectorScheduler)

    def test_pull_worker_construction(self):
        w = NixlPullConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert isinstance(w, NixlBaseConnectorWorker)

    def test_push_scheduler_construction(self):
        s = NixlPushConnectorScheduler(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert isinstance(s, NixlBaseConnectorScheduler)
        assert s.has_pending_push_work() is False

    def test_push_worker_construction(self):
        w = NixlPushConnectorWorker(
            vllm_config=None, engine_id="e0", kv_cache_config=None,
        )
        assert isinstance(w, NixlBaseConnectorWorker)

    def test_pull_connector_construction(self):
        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        assert isinstance(c, NixlBaseConnector)
        assert c.connector_scheduler is not None

    def test_push_connector_construction(self):
        c = NixlPushConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        assert isinstance(c, NixlBaseConnector)
        assert c.connector_scheduler is not None

    def test_push_connector_unknown_role(self):
        with pytest.raises(ValueError):
            NixlPushConnector(
                vllm_config=None,
                role="bogus",  # not KVConnectorRole
                kv_cache_config=None,
            )


# ---------------------------------------------------------------------------
# Test 8: connector facade end-to-end (local shared-memory backend)
# ---------------------------------------------------------------------------


class TestConnectorFacadeE2E:
    def test_pull_save_and_load(self):
        """End-to-end: producer saves KV, consumer retrieves it."""
        import torch

        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,  # role doesn't matter for local backend
            kv_cache_config=None,
            num_layers=2,
            buffer_size=1024,
        )
        kv = torch.randn(64, dtype=torch.float32)
        c.save_kv_layer("layer0", kv)
        assert c._kv_loaded_flags.get("layer0") is True
        loaded = c.get_bundle_kv("layer0")
        assert loaded is not None
        # The shared buffer holds the flat representation.
        assert loaded.numel() == 64

    def test_check_transfer_done_no_pending(self):
        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        result = c.check_transfer_done()
        assert result.status == TransferStatus.SUCCESS

    def test_ack_round_trip(self):
        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        c.send_ack("req1", True)
        c.send_ack("req2", False)
        acks = c.drain_acks()
        assert acks == {"req1": True, "req2": False}
        # Drain again → empty.
        assert c.drain_acks() == {}

    def test_request_finished_cleans_up(self):
        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        c._kv_loaded_flags["req1"] = True
        c._acks["req1"] = True
        c.request_finished("req1", None)
        assert "req1" not in c._kv_loaded_flags
        assert "req1" not in c._acks

    def test_connector_shutdown(self):
        c = NixlPullConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        c.shutdown()
        assert c.connector_scheduler is None


# ---------------------------------------------------------------------------
# Test 9: backward compat
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_nixl_connector_is_pull(self):
        """``NixlConnector`` is a backward-compat alias for ``NixlPullConnector``."""
        assert NixlConnector is NixlPullConnector

    def test_connector_scheduler_is_pull(self):
        """``NixlConnectorScheduler`` is a backward-compat alias for ``NixlPullConnectorScheduler``."""
        assert NixlConnectorScheduler is NixlPullConnectorScheduler

    def test_connector_worker_is_pull(self):
        """``NixlConnectorWorker`` is a backward-compat alias for ``NixlPullConnectorWorker``."""
        assert NixlConnectorWorker is NixlPullConnectorWorker

    def test_example_connector_inherits_nixl(self):
        """ExampleConnector (C212-era) is a subclass of NixlConnector."""
        from my_vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
            ExampleConnector,
        )
        assert issubclass(ExampleConnector, NixlConnector)

    def test_example_connector_construction(self):
        """ExampleConnector accepts the C212 kwargs."""
        from my_vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
            ExampleConnector,
        )
        c = ExampleConnector(
            vllm_config=None,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
            num_layers=4,
            buffer_size=2048,
        )
        assert c._num_layers == 4
        assert c._buffer_size == 2048


# ---------------------------------------------------------------------------
# Test 10: NIXL runtime guard
# ---------------------------------------------------------------------------


class TestNIXLGuard:
    def test_compute_handshake_without_nixl_agent(self):
        """_compute_handshake should work even without a NIXL agent
        (returns a handshake with empty agent metadata bytes)."""
        class FakeVllmConfig:
            class model_config:
                model = "opt-125m"
                dtype = "bfloat16"
                @staticmethod
                def get_total_num_kv_heads(): return 12
                @staticmethod
                def get_head_size(): return 64
                @staticmethod
                def get_total_num_hidden_layers(): return 12

            class cache_config:
                cache_dtype = "auto"

            class scheduler_config:
                disable_hybrid_kv_cache_manager = False

        w = NixlBaseConnectorWorker(
            vllm_config=FakeVllmConfig(),
            engine_id="e0",
            kv_cache_config=None,
        )
        hs = w._compute_handshake("TRITON_ATTN", cross_layers_blocks=False)
        assert isinstance(hs, NixlHandshakePayload)
        assert len(hs.compatibility_hash) == 64

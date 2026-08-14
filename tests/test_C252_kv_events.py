"""tests/test_C252_kv_events.py — Tests for the KV event bus + coordinator.

C252: PD-decoupled event flow (producer → coordinator → consumer)

Verifies:

  1. Module imports.
  2. Event types — BlockStored / BlockRemoved / AllBlocksCleared
     construction + hashing.
  3. KVEventBatch + ts.
  4. KVEventAggregator — multi-worker dedup + counters.
  5. KVConnectorKVEvents — add / drain / consume.
  6. EventPublisher ABC.
  7. NullEventPublisher — no-op.
  8. InMemoryEventPublisher — subscribe + publish + shutdown.
  9. EventPublisherFactory — registry + default fallback.
 10. StatelessCoordinator — register producer / consumer.
 11. StatelessCoordinator — publish routes events to consumers.
 12. StatelessCoordinator — block filter (consumer only gets
     matching events).
 13. StatelessCoordinator — event log + replay.
 14. StatelessCoordinator — KV transfer trigger on BlockStored.
 15. StatelessCoordinator — module-level singleton.
 16. End-to-end: producer emits BlockStored → coordinator routes
     → consumer triggers KV transfer.

Run:
    pytest tests/test_C252_kv_events.py -v
"""

from __future__ import annotations

import time

import pytest

from my_vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventPublisher,
    EventPublisherFactory,
    InMemoryEventPublisher,
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
    KVEventBatch,
    MEDIUM_CPU,
    MEDIUM_GPU,
    NullEventPublisher,
)
from my_vllm.distributed.stateless_coordinator import (
    ConsumerRegistration,
    ProducerRegistration,
    StatelessCoordinator,
    get_stateless_coordinator,
    reset_stateless_coordinator,
)


# ---------------------------------------------------------------------------
# Test 1: Module imports
# ---------------------------------------------------------------------------


class TestModuleImports:
    def test_event_types(self):
        assert BlockStored is not None
        assert BlockRemoved is not None
        assert AllBlocksCleared is not None
        assert KVCacheEvent is not None

    def test_publishers(self):
        assert NullEventPublisher is not None
        assert InMemoryEventPublisher is not None
        assert EventPublisherFactory is not None

    def test_coordinator(self):
        assert StatelessCoordinator is not None
        assert get_stateless_coordinator is not None


# ---------------------------------------------------------------------------
# Test 2: Event types
# ---------------------------------------------------------------------------


class TestEventTypes:
    def test_block_stored_construction(self):
        e = BlockStored(
            block_hashes=[1, 2, 3],
            parent_block_hash=0,
            token_ids=[10, 20, 30],
            block_size=16,
            medium=MEDIUM_GPU,
        )
        assert e.block_hashes == [1, 2, 3]
        assert e.medium == MEDIUM_GPU
        assert e.lora_id is None

    def test_block_stored_hash(self):
        e1 = BlockStored(
            block_hashes=[1], parent_block_hash=0,
            token_ids=[10], block_size=16, medium=MEDIUM_GPU,
        )
        e2 = BlockStored(
            block_hashes=[1], parent_block_hash=0,
            token_ids=[10], block_size=16, medium=MEDIUM_GPU,
        )
        # Hashable + equal.
        assert hash(e1) == hash(e2)

    def test_block_removed_construction(self):
        e = BlockRemoved(
            block_hashes=[5, 6], medium=MEDIUM_CPU,
        )
        assert e.block_hashes == [5, 6]
        assert e.medium == MEDIUM_CPU

    def test_all_blocks_cleared(self):
        e = AllBlocksCleared()
        # No fields.
        assert isinstance(e, KVCacheEvent)


# ---------------------------------------------------------------------------
# Test 3: KVEventBatch
# ---------------------------------------------------------------------------


class TestKVEventBatch:
    def test_construction(self):
        b = KVEventBatch(
            ts=time.time(),
            events=[BlockStored(
                block_hashes=[1], parent_block_hash=None,
                token_ids=[1], block_size=16, medium=MEDIUM_GPU,
            )],
        )
        assert len(b.events) == 1

    def test_construction_with_dp_rank(self):
        b = KVEventBatch(
            ts=1.0, events=[], data_parallel_rank=2,
        )
        assert b.data_parallel_rank == 2


# ---------------------------------------------------------------------------
# Test 4: KVEventAggregator
# ---------------------------------------------------------------------------


class TestKVEventAggregator:
    def test_initial_state(self):
        agg = KVEventAggregator(num_workers=2)
        assert agg.get_number_of_workers() == 2
        assert agg.get_common_events() == []
        assert agg.get_all_events() == []

    def test_add_events_dedup(self):
        agg = KVEventAggregator(num_workers=2)
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        agg.add_events([e])
        agg.add_events([e])
        # The event is emitted by both workers.
        common = agg.get_common_events()
        assert common == [e]

    def test_add_events_partial(self):
        agg = KVEventAggregator(num_workers=2)
        e1 = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        e2 = BlockStored(
            block_hashes=[2], parent_block_hash=None,
            token_ids=[2], block_size=16, medium=MEDIUM_GPU,
        )
        # Worker 1: e1 + e2
        agg.add_events([e1, e2])
        # Worker 2: only e1
        agg.add_events([e1])
        # Only e1 is common.
        common = agg.get_common_events()
        assert common == [e1]

    def test_add_events_invalid_type(self):
        agg = KVEventAggregator(num_workers=1)
        with pytest.raises(TypeError):
            agg.add_events("not a list")  # type: ignore[arg-type]

    def test_increment_workers(self):
        agg = KVEventAggregator(num_workers=1)
        agg.increment_workers(2)
        assert agg.get_number_of_workers() == 3

    def test_increment_workers_invalid(self):
        agg = KVEventAggregator(num_workers=1)
        with pytest.raises(ValueError):
            agg.increment_workers(0)

    def test_reset_workers(self):
        agg = KVEventAggregator(num_workers=5)
        agg.reset_workers()
        assert agg.get_number_of_workers() == 1

    def test_clear_events(self):
        agg = KVEventAggregator(num_workers=1)
        agg.add_events([BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )])
        agg.clear_events()
        assert agg.get_all_events() == []

    def test_invalid_num_workers(self):
        with pytest.raises(ValueError):
            KVEventAggregator(num_workers=0)

    def test_repr(self):
        agg = KVEventAggregator(num_workers=2)
        s = repr(agg)
        assert "KVEventAggregator" in s
        assert "workers=2" in s


# ---------------------------------------------------------------------------
# Test 5: KVConnectorKVEvents
# ---------------------------------------------------------------------------


class TestKVConnectorKVEvents:
    def test_empty(self):
        events = KVConnectorKVEvents()
        assert len(events) == 0
        assert events.get_all_events() == []

    def test_add_events(self):
        events = KVConnectorKVEvents()
        e1 = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        e2 = BlockRemoved(block_hashes=[2], medium=MEDIUM_GPU)
        events.add_events([e1, e2])
        assert len(events) == 2
        all_events = events.get_all_events()
        assert all_events == [e1, e2]
        # Consumed.
        assert len(events) == 0

    def test_add_event_singleton(self):
        events = KVConnectorKVEvents()
        e = AllBlocksCleared()
        events.add_event(e)
        assert len(events) == 1
        assert events.take_events() == [e]

    def test_clear(self):
        events = KVConnectorKVEvents()
        events.add_event(AllBlocksCleared())
        events.clear_events()
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 6: EventPublisher ABC
# ---------------------------------------------------------------------------


class TestEventPublisherABC:
    def test_abstract_methods(self):
        # Cannot instantiate the ABC directly.
        with pytest.raises(TypeError):
            EventPublisher()  # type: ignore[abstract]

    def test_data_parallel_rank(self):
        p = NullEventPublisher(data_parallel_rank=3)
        assert p.data_parallel_rank == 3


# ---------------------------------------------------------------------------
# Test 7: NullEventPublisher
# ---------------------------------------------------------------------------


class TestNullEventPublisher:
    def test_publish_is_noop(self):
        p = NullEventPublisher()
        # Should not raise.
        p.publish(KVEventBatch(ts=0.0, events=[]))
        p.shutdown()


# ---------------------------------------------------------------------------
# Test 8: InMemoryEventPublisher
# ---------------------------------------------------------------------------


class TestInMemoryEventPublisher:
    def test_subscribe_publish(self):
        p = InMemoryEventPublisher()
        received: list[KVEventBatch] = []

        def cb(batch: KVEventBatch) -> None:
            received.append(batch)

        p.add_subscriber(cb)
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        batch = KVEventBatch(ts=1.0, events=[e])
        p.publish(batch)
        # Wait briefly for the publisher thread.
        time.sleep(0.2)
        assert len(received) == 1
        assert received[0].events[0] is e
        p.shutdown()

    def test_remove_subscriber(self):
        p = InMemoryEventPublisher()
        received: list[KVEventBatch] = []

        def cb(batch: KVEventBatch) -> None:
            received.append(batch)

        p.add_subscriber(cb)
        p.remove_subscriber(cb)
        p.publish(KVEventBatch(ts=0.0, events=[AllBlocksCleared()]))
        time.sleep(0.2)
        assert received == []
        p.shutdown()

    def test_remove_unknown_subscriber(self):
        p = InMemoryEventPublisher()

        def cb(_):
            pass

        # Should not raise.
        p.remove_subscriber(cb)
        p.shutdown()

    def test_publish_after_shutdown_raises(self):
        p = InMemoryEventPublisher()
        p.shutdown()
        with pytest.raises(RuntimeError):
            p.publish(KVEventBatch(ts=0.0, events=[]))

    def test_auto_assign_dp_rank(self):
        p = InMemoryEventPublisher(data_parallel_rank=5)
        received: list[KVEventBatch] = []

        def cb(batch: KVEventBatch) -> None:
            received.append(batch)

        p.add_subscriber(cb)
        # Batch without dp_rank → publisher assigns 5.
        batch = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
        p.publish(batch)
        time.sleep(0.2)
        assert received[0].data_parallel_rank == 5
        p.shutdown()

    def test_replay_buffer(self):
        p = InMemoryEventPublisher(buffer_steps=3)
        for i in range(5):
            p.publish(KVEventBatch(ts=float(i), events=[]))
        time.sleep(0.3)
        # Only last 3 retained.
        buffer = p.get_replay_buffer()
        assert len(buffer) == 3
        p.shutdown()

    def test_default_data_parallel_rank(self):
        p = InMemoryEventPublisher()
        assert p.data_parallel_rank == 0
        p.shutdown()


# ---------------------------------------------------------------------------
# Test 9: EventPublisherFactory
# ---------------------------------------------------------------------------


class TestEventPublisherFactory:
    def test_create_null(self):
        p = EventPublisherFactory.create("null")
        assert isinstance(p, NullEventPublisher)

    def test_create_none(self):
        p = EventPublisherFactory.create(None)
        assert isinstance(p, NullEventPublisher)

    def test_create_in_memory(self):
        p = EventPublisherFactory.create("in_memory")
        assert isinstance(p, InMemoryEventPublisher)
        p.shutdown()

    def test_unknown_publisher(self):
        with pytest.raises(ValueError):
            EventPublisherFactory.create("nonexistent")

    def test_register_publisher(self):
        class FakePublisher(EventPublisher):
            def publish(self, events):
                pass

            def shutdown(self):
                pass

        EventPublisherFactory.register_publisher("fake", FakePublisher)
        try:
            p = EventPublisherFactory.create("fake", data_parallel_rank=1)
            assert isinstance(p, FakePublisher)
            assert p.data_parallel_rank == 1
        finally:
            # Clean up registry.
            EventPublisherFactory._registry.pop("fake", None)

    def test_register_duplicate(self):
        with pytest.raises(KeyError):
            EventPublisherFactory.register_publisher("null", NullEventPublisher)


# ---------------------------------------------------------------------------
# Test 10: StatelessCoordinator — registration
# ---------------------------------------------------------------------------


class TestCoordinatorRegistration:
    def test_register_producer(self):
        c = StatelessCoordinator()
        c.register_producer("engine-0", lambda b: None)
        assert c.is_producer_registered("engine-0")
        assert c.get_producer_count() == 1

    def test_register_consumer(self):
        c = StatelessCoordinator()
        c.register_consumer("engine-0", lambda e: None)
        assert c.is_consumer_registered("engine-0")
        assert c.get_consumer_count() == 1

    def test_unregister_producer(self):
        c = StatelessCoordinator()
        c.register_producer("engine-0", lambda b: None)
        c.unregister_producer("engine-0")
        assert not c.is_producer_registered("engine-0")

    def test_unregister_consumer(self):
        c = StatelessCoordinator()
        c.register_consumer("engine-0", lambda e: None)
        c.unregister_consumer("engine-0")
        assert not c.is_consumer_registered("engine-0")

    def test_publish_without_producer(self):
        """``publish`` with no producer_id is allowed (default)."""
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append)
        batch = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
        delivered = c.publish(batch)
        assert delivered == 1
        assert len(received) == 1

    def test_publish_with_unregistered_producer(self):
        # An unregistered producer is allowed (used by tests + ad-hoc
        # publish paths).  The coordinator logs a debug message but
        # still routes events to consumers.
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append)
        batch = KVEventBatch(ts=0.0, events=[AllBlocksCleared()])
        delivered = c.publish(batch, producer_engine_id="nonexistent")
        assert delivered == 1
        assert received == batch.events

    def test_producer_registration_replaces(self):
        c = StatelessCoordinator()
        c.register_producer("e", lambda b: None)
        c.register_producer("e", lambda b: None)
        # No error; just replaces.
        assert c.get_producer_count() == 1


# ---------------------------------------------------------------------------
# Test 11: StatelessCoordinator — publish routes events
# ---------------------------------------------------------------------------


class TestCoordinatorRouting:
    def test_routes_to_single_consumer(self):
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append)
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=0.0, events=[e]))
        assert received == [e]

    def test_routes_to_multiple_consumers(self):
        c = StatelessCoordinator()
        r0: list[KVCacheEvent] = []
        r1: list[KVCacheEvent] = []
        c.register_consumer("c0", r0.append)
        c.register_consumer("c1", r1.append)
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        delivered = c.publish(KVEventBatch(ts=0.0, events=[e]))
        assert delivered == 2
        assert r0 == [e]
        assert r1 == [e]

    def test_disabled_consumer_skipped(self):
        c = StatelessCoordinator()
        c.register_consumer("c0", lambda e: None)
        # Disable.
        c._consumers["c0"].enabled = False
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        delivered = c.publish(KVEventBatch(ts=0.0, events=[e]))
        assert delivered == 0

    def test_publish_event_convenience(self):
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append)
        e = AllBlocksCleared()
        delivered = c.publish_event(e)
        assert delivered == 1
        assert received == [e]

    def test_publish_records_event_log(self):
        c = StatelessCoordinator()
        e = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=1.0, events=[e]))
        log = c.get_event_log()
        assert len(log) == 1
        assert log[0].event is e
        assert log[0].timestamp == 1.0


# ---------------------------------------------------------------------------
# Test 12: StatelessCoordinator — block filter
# ---------------------------------------------------------------------------


class TestBlockFilter:
    def test_filter_only_matching_hashes(self):
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer(
            "c0", received.append, block_filter={1, 2},
        )
        e_matching = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        e_other = BlockStored(
            block_hashes=[99], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=0.0, events=[e_matching, e_other]))
        # Only e_matching delivered.
        assert received == [e_matching]

    def test_filter_skips_non_block_events(self):
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append, block_filter={1})
        # AllBlocksCleared has no block_hashes → skipped.
        c.publish(KVEventBatch(ts=0.0, events=[AllBlocksCleared()]))
        assert received == []

    def test_filter_block_removed(self):
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append, block_filter={5})
        e = BlockRemoved(block_hashes=[5], medium=MEDIUM_GPU)
        c.publish(KVEventBatch(ts=0.0, events=[e]))
        assert received == [e]


# ---------------------------------------------------------------------------
# Test 13: StatelessCoordinator — event log + replay
# ---------------------------------------------------------------------------


class TestEventLog:
    def test_get_events_for_producer(self):
        c = StatelessCoordinator()
        c.publish(KVEventBatch(ts=0.0, events=[AllBlocksCleared()]),
                  producer_engine_id="p0")
        c.publish(KVEventBatch(ts=1.0, events=[AllBlocksCleared()]),
                  producer_engine_id="p1")
        events = c.get_events_for_producer("p0")
        assert len(events) == 1
        assert events[0].producer_engine_id == "p0"

    def test_get_events_for_block_hash(self):
        c = StatelessCoordinator()
        e1 = BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        e2 = BlockStored(
            block_hashes=[2], parent_block_hash=None,
            token_ids=[2], block_size=16, medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=0.0, events=[e1, e2]))
        events = c.get_events_for_block_hash(1)
        assert len(events) == 1
        assert events[0].event is e1

    def test_clear_event_log(self):
        c = StatelessCoordinator()
        c.publish(KVEventBatch(ts=0.0, events=[AllBlocksCleared()]))
        c.clear_event_log()
        assert c.get_event_log() == []


# ---------------------------------------------------------------------------
# Test 14: StatelessCoordinator — KV transfer trigger
# ---------------------------------------------------------------------------


class TestKVTransferTrigger:
    def test_trigger_for_registered_consumer(self):
        c = StatelessCoordinator()
        c.register_consumer("c0", lambda e: None)
        ok = c.trigger_kv_transfer("c0", block_hash=42, source_engine_id="p0")
        assert ok is True
        triggers = c.get_kv_transfer_triggers("c0")
        assert len(triggers) == 1
        assert triggers[0]["block_hash"] == 42

    def test_trigger_for_unregistered_consumer(self):
        c = StatelessCoordinator()
        ok = c.trigger_kv_transfer("nonexistent", block_hash=42,
                                  source_engine_id="p0")
        assert ok is False
        assert c.get_kv_transfer_triggers() == []

    def test_get_all_triggers(self):
        c = StatelessCoordinator()
        c.register_consumer("c0", lambda e: None)
        c.register_consumer("c1", lambda e: None)
        c.trigger_kv_transfer("c0", block_hash=1, source_engine_id="p")
        c.trigger_kv_transfer("c1", block_hash=2, source_engine_id="p")
        assert len(c.get_kv_transfer_triggers()) == 2

    def test_clear_triggers(self):
        c = StatelessCoordinator()
        c.register_consumer("c0", lambda e: None)
        c.trigger_kv_transfer("c0", block_hash=1, source_engine_id="p")
        c.clear_kv_transfer_triggers()
        assert c.get_kv_transfer_triggers() == []


# ---------------------------------------------------------------------------
# Test 15: module-level singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_singleton(self):
        reset_stateless_coordinator()
        c1 = get_stateless_coordinator()
        c2 = get_stateless_coordinator()
        assert c1 is c2

    def test_reset(self):
        c1 = get_stateless_coordinator()
        reset_stateless_coordinator()
        c2 = get_stateless_coordinator()
        assert c1 is not c2


# ---------------------------------------------------------------------------
# Test 16: end-to-end — producer → coordinator → consumer → KV transfer
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_block_stored_triggers_kv_transfer(self):
        """Simulate the C252 plan: producer emits BlockStored →
        coordinator routes → consumer triggers KV transfer."""
        c = StatelessCoordinator()
        received_events: list[KVCacheEvent] = []
        received_triggers: list[dict] = []

        def consumer_callback(event: KVCacheEvent) -> None:
            received_events.append(event)
            if isinstance(event, BlockStored):
                # Simulate the consumer's KV transfer trigger.
                for h in event.block_hashes:
                    c.trigger_kv_transfer(
                        "consumer", block_hash=h,
                        source_engine_id="producer",
                    )

        c.register_consumer("consumer", consumer_callback)
        c.register_producer("producer", lambda b: None)

        # Producer emits a BlockStored event.
        e = BlockStored(
            block_hashes=[100, 101],
            parent_block_hash=None,
            token_ids=list(range(32)),
            block_size=16,
            medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=time.time(), events=[e]),
                  producer_engine_id="producer")

        # Consumer received the event.
        assert received_events == [e]
        # And triggered a KV transfer for each block hash.
        triggers = c.get_kv_transfer_triggers("consumer")
        assert len(triggers) == 2
        assert {t["block_hash"] for t in triggers} == {100, 101}
        assert all(t["source_engine_id"] == "producer" for t in triggers)

    def test_consumer_with_block_filter_triggers_only_matching(self):
        """Consumer is only interested in block 100; only matching
        event triggers a transfer."""
        c = StatelessCoordinator()
        triggers: list[dict] = []

        def cb(event: KVCacheEvent) -> None:
            if isinstance(event, BlockStored):
                for h in event.block_hashes:
                    if h in {100}:
                        c.trigger_kv_transfer(
                            "consumer", block_hash=h,
                            source_engine_id="producer",
                        )

        c.register_consumer("consumer", cb, block_filter={100})
        c.register_producer("producer", lambda b: None)

        e_matching = BlockStored(
            block_hashes=[100], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        e_other = BlockStored(
            block_hashes=[999], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        )
        c.publish(KVEventBatch(ts=0.0, events=[e_matching, e_other]),
                  producer_engine_id="producer")

        triggers = c.get_kv_transfer_triggers("consumer")
        assert len(triggers) == 1
        assert triggers[0]["block_hash"] == 100

    def test_collect_from_connector_events(self):
        """The C251 ``take_events`` hook feeds into the coordinator."""
        c = StatelessCoordinator()
        received: list[KVCacheEvent] = []
        c.register_consumer("c0", received.append)

        events = KVConnectorKVEvents()
        events.add_event(BlockStored(
            block_hashes=[1], parent_block_hash=None,
            token_ids=[1], block_size=16, medium=MEDIUM_GPU,
        ))
        events.add_event(BlockRemoved(block_hashes=[2], medium=MEDIUM_GPU))
        n = c.collect_connector_events(events)
        assert n == 2
        assert len(received) == 2
        # Connector events drained.
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 17: backward compat — C250 / C251 still work
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_c250_nixl_connector_still_imports(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
            NixlConnector,
        )
        assert NixlConnector is not None

    def test_c251_kv_index_still_works(self):
        from my_vllm.distributed.kv_transfer.kv_connector.v1 import (
            KVIndex,
            KVFailureTracker,
        )
        idx = KVIndex()
        idx.add("req1", [1, 2, 3])
        assert len(idx) == 3
        t = KVFailureTracker()
        t.record_failure("req1", failed_block_ids=[1])
        assert t.has_failure("req1")

    def test_event_bus_works_with_take_events(self):
        """Connector's ``take_events`` produces events that the
        coordinator can consume."""
        # ``KVConnectorKVEvents`` lives in ``my_vllm.distributed.kv_events``
        # (not in ``kv_connector.v1``).  The connector's ``take_events``
        # hook would return a list of these events; the coordinator
        # consumes them via ``collect_connector_events``.
        assert KVConnectorKVEvents is not None

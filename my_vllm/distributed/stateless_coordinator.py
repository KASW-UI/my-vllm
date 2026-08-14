"""Stateless coordinator for PD-decoupled KV event routing.

C252: extracted from ``vllm/distributed/stateless_coordinator.py``
(381 lines).  Scope reductions:

  * **No TCPStore / stateless process group** — vLLM's
    ``StatelessGroupCoordinator`` is a 3-port multi-process group
    built on ``torch.distributed.TCPStore`` + ``StatelessProcessGroup``.
    my-vllm doesn't vendor either of these (the C243 weight transfer
    path uses vanilla ``torch.distributed.broadcast`` directly, and
    the C242 SHM broadcast primitive handles same-host messages).
  * **No ``StatelessProcessGroup``** — replaced with an in-process
    ``threading.Lock`` + Python queue for the producer → consumer
    event routing.
  * **No ``GroupCoordinator``** — replaced with a much simpler
    ``StatelessCoordinator`` that just routes events between
    registered producers and consumers in one process.

The coordinator's role in the C252 plan:

  1. **Producer registration** — a prefill engine registers a
     callback that emits ``BlockStored`` events when it completes
     prefill.
  2. **Consumer subscription** — a decode engine registers a
     callback that receives events and triggers KV transfer.
  3. **Routing** — the coordinator forwards each event from every
     producer to every interested consumer.
  4. **Trigger** — when a consumer receives a ``BlockStored`` event
     for a block whose hash matches a request the consumer is
     waiting for, it initiates a KV transfer via the C250 NIXL
     connector.

For C252, the coordinator is in-process only.  Cross-process
routing (e.g., prefill on host A, decode on host B) would require
either the C242 SHM broadcast primitive (same-host) or a
TCP-based event bus (deferred to a later C25x commit).

References:
  * vLLM stateless_coordinator:
    ``vllm/distributed/stateless_coordinator.py``.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from my_vllm.distributed.kv_events import (
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
    KVConnectorKVEvents,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Producer / consumer registration
# ---------------------------------------------------------------------------


@dataclass
class ProducerRegistration:
    """Registration record for a producer engine."""

    engine_id: str
    callback: Callable[[KVEventBatch], None]
    """Called by the coordinator when the producer publishes events."""

    medium: str = "GPU"
    """Storage medium (matches vLLM MEDIUM_* constants)."""

    locality: str = "LOCAL"
    """Where the producer is relative to the coordinator (LOCAL/REMOTE)."""

    enabled: bool = True


@dataclass
class ConsumerRegistration:
    """Registration record for a consumer engine."""

    engine_id: str
    callback: Callable[[KVCacheEvent], None]
    """Called by the coordinator for each event destined for this consumer."""

    block_filter: set[int] | None = None
    """Optional set of block hashes the consumer cares about.

    ``None`` means "deliver every event".  When set, the coordinator
    only forwards events whose block hashes intersect this set.
    """

    enabled: bool = True


@dataclass
class StoredEvent:
    """Internal record for a queued event (coordinator-side)."""

    event: KVCacheEvent
    producer_engine_id: str
    timestamp: float


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------


class StatelessCoordinator:
    """In-process coordinator for PD-decoupled KV event routing.

    Mirrors ``vllm.distributed.stateless_coordinator
    .StatelessGroupCoordinator`` but scoped to the C252 use case:
    routing ``BlockStored`` / ``BlockRemoved`` events from producers
    to consumers, with optional per-consumer block filtering.

    Threading model:

      * Single coordinator instance per process.
      * Producers call ``publish(events)`` from any thread.
      * Consumers register callbacks that receive events on the
        publisher's thread (synchronous delivery).
      * The lock-free event queue uses ``queue.Queue`` (cross-thread
        safe + bounded) so the publisher can post events without
        blocking on the consumer.

    For cross-process routing, set up a C242 SHM broadcast handler
    that forwards events to remote coordinators.  Not implemented
    in C252 — single-process only.
    """

    def __init__(
        self,
        max_buffer_steps: int = 10_000,
    ) -> None:
        self._producers: dict[str, ProducerRegistration] = {}
        self._consumers: dict[str, ConsumerRegistration] = {}
        self._event_log: deque[StoredEvent] = deque(maxlen=max_buffer_steps)
        self._lock = threading.RLock()
        # Detected consumers' KV transfer triggers — test helper.
        self._kv_transfer_triggers: list[dict[str, Any]] = []

    # ── Producer / consumer registration ──────────────────────────

    def register_producer(
        self,
        engine_id: str,
        callback: Callable[[KVEventBatch], None],
        medium: str = "GPU",
        locality: str = "LOCAL",
    ) -> None:
        """Register a producer engine.

        Args:
            engine_id: Stable identifier for the producer.
            callback: Called by ``publish`` to inject events from the
                producer side.
            medium: Where the producer stores KV blocks
                (MEDIUM_GPU / MEDIUM_CPU / MEDIUM_FS / MEDIUM_OBJ).
            locality: ``LOCAL`` or ``REMOTE``.
        """
        with self._lock:
            self._producers[engine_id] = ProducerRegistration(
                engine_id=engine_id,
                callback=callback,
                medium=medium,
                locality=locality,
            )

    def register_consumer(
        self,
        engine_id: str,
        callback: Callable[[KVCacheEvent], None],
        block_filter: set[int] | None = None,
    ) -> None:
        """Register a consumer engine.

        Args:
            engine_id: Stable identifier for the consumer.
            callback: Called for each event destined for this consumer.
            block_filter: When set, only events whose block hashes
                intersect this set are delivered.
        """
        with self._lock:
            self._consumers[engine_id] = ConsumerRegistration(
                engine_id=engine_id,
                callback=callback,
                block_filter=block_filter,
            )

    def unregister_producer(self, engine_id: str) -> None:
        with self._lock:
            self._producers.pop(engine_id, None)

    def unregister_consumer(self, engine_id: str) -> None:
        with self._lock:
            self._consumers.pop(engine_id, None)

    def is_producer_registered(self, engine_id: str) -> bool:
        with self._lock:
            return engine_id in self._producers

    def is_consumer_registered(self, engine_id: str) -> bool:
        with self._lock:
            return engine_id in self._consumers

    def get_producer_count(self) -> int:
        with self._lock:
            return len(self._producers)

    def get_consumer_count(self) -> int:
        with self._lock:
            return len(self._consumers)

    # ── Event publishing ─────────────────────────────────────────

    def publish(
        self,
        events: KVEventBatch,
        producer_engine_id: str | None = None,
    ) -> int:
        """Publish a batch of events.

        Routes the events to every registered consumer (applying the
        per-consumer block filter).  Returns the count of events
        delivered.

        Args:
            events: The batch to publish.
            producer_engine_id: Which producer is publishing.  When
                ``None``, the events are attributed to a
                "default" producer (used for tests).
        """
        if producer_engine_id is not None:
            with self._lock:
                producer = self._producers.get(producer_engine_id)
            if producer is None:
                # Producer not registered — log and continue.  This
                # keeps the bus functional for tests + ad-hoc publish
                # cases where the producer hasn't been formalized.
                logger.debug(
                    "publish from unregistered producer '%s'",
                    producer_engine_id,
                )

        delivered = 0
        ts = events.ts
        with self._lock:
            consumers = list(self._consumers.values())
        for event in events.events:
            stored = StoredEvent(
                event=event,
                producer_engine_id=producer_engine_id or "default",
                timestamp=ts,
            )
            self._event_log.append(stored)
            for consumer in consumers:
                if not consumer.enabled:
                    continue
                if consumer.block_filter is not None:
                    # Only deliver events whose block hashes intersect.
                    event_hashes = getattr(event, "block_hashes", None)
                    if not event_hashes:
                        continue
                    if not any(h in consumer.block_filter for h in event_hashes):
                        continue
                try:
                    consumer.callback(event)
                    delivered += 1
                except Exception:
                    logger.exception(
                        "error delivering event to consumer '%s'",
                        consumer.engine_id,
                    )
        return delivered

    def publish_event(
        self,
        event: KVCacheEvent,
        producer_engine_id: str | None = None,
        ts: float | None = None,
    ) -> int:
        """Convenience: publish a single event in a fresh batch."""
        import time

        batch = KVEventBatch(
            ts=ts if ts is not None else time.time(),
            events=[event],
        )
        return self.publish(batch, producer_engine_id=producer_engine_id)

    # ── Event log + replay ───────────────────────────────────────

    def get_event_log(self) -> list[StoredEvent]:
        """Snapshot the in-memory event log (FIFO bounded buffer)."""
        with self._lock:
            return list(self._event_log)

    def get_events_for_producer(
        self,
        producer_engine_id: str,
    ) -> list[StoredEvent]:
        with self._lock:
            return [
                e for e in self._event_log
                if e.producer_engine_id == producer_engine_id
            ]

    def get_events_for_block_hash(self, block_hash: int) -> list[StoredEvent]:
        with self._lock:
            result: list[StoredEvent] = []
            for stored in self._event_log:
                event_hashes = getattr(stored.event, "block_hashes", None)
                if event_hashes and block_hash in event_hashes:
                    result.append(stored)
            return result

    def clear_event_log(self) -> None:
        with self._lock:
            self._event_log.clear()

    # ── KV transfer trigger ──────────────────────────────────────

    def trigger_kv_transfer(
        self,
        consumer_engine_id: str,
        block_hash: int,
        source_engine_id: str,
    ) -> bool:
        """Record a KV transfer trigger for ``block_hash``.

        The coordinator records the trigger so consumers can poll
        for "should I issue a KV transfer?" decisions.  Real
        triggering logic (which issues the NIXL READ) lives in
        the consumer's ``get_num_new_matched_tokens`` callback.

        Returns True if the trigger was recorded (i.e. the consumer
        is registered); False otherwise.
        """
        with self._lock:
            if consumer_engine_id not in self._consumers:
                return False
            self._kv_transfer_triggers.append({
                "consumer_engine_id": consumer_engine_id,
                "block_hash": block_hash,
                "source_engine_id": source_engine_id,
            })
        return True

    def get_kv_transfer_triggers(
        self,
        consumer_engine_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recorded triggers (optionally filtered by consumer)."""
        with self._lock:
            if consumer_engine_id is None:
                return list(self._kv_transfer_triggers)
            return [
                t for t in self._kv_transfer_triggers
                if t["consumer_engine_id"] == consumer_engine_id
            ]

    def clear_kv_transfer_triggers(self) -> None:
        with self._lock:
            self._kv_transfer_triggers.clear()

    # ── Connector-side event collector ───────────────────────────

    def collect_connector_events(
        self,
        events: KVConnectorKVEvents,
    ) -> int:
        """Take events from a ``KVConnectorKVEvents`` and publish them.

        Convenience helper for the C251 ``take_events`` hook — the
        scheduler drains connector events each step and pipes them
        through the coordinator.
        """
        events_list = events.take_events()
        if not events_list:
            return 0
        import time
        batch = KVEventBatch(ts=time.time(), events=events_list)
        return self.publish(batch)


# ---------------------------------------------------------------------------
# Module-level singleton (singleton pattern, mirroring vLLM)
# ---------------------------------------------------------------------------


_coordinator_singleton: StatelessCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_stateless_coordinator() -> StatelessCoordinator:
    """Return the process-wide ``StatelessCoordinator`` singleton.

    Mirrors ``vllm.distributed.stateless_coordinator``'s module-level
    helper pattern.  The singleton is created lazily on first call.
    """
    global _coordinator_singleton
    with _coordinator_lock:
        if _coordinator_singleton is None:
            _coordinator_singleton = StatelessCoordinator()
        return _coordinator_singleton


def reset_stateless_coordinator() -> None:
    """Reset the singleton (test helper)."""
    global _coordinator_singleton
    with _coordinator_lock:
        _coordinator_singleton = None


__all__ = [
    "ConsumerRegistration",
    "ProducerRegistration",
    "StatelessCoordinator",
    "StoredEvent",
    "get_stateless_coordinator",
    "reset_stateless_coordinator",
]

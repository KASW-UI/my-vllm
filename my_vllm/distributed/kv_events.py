"""KV cache event bus — events that flow from producers to consumers.

C252: extracted from ``vllm/distributed/kv_events.py`` (543 lines).
Scope reductions:

  * ``msgspec.Struct`` base — replaced with ``@dataclass`` (msgspec
    not vendored).
  * ``ExternalBlockHash`` — replaced with a plain ``int`` (the hash
    values are computed by the scheduler; my-vllm uses whatever
    algorithm the scheduler produces).
  * ``ZmqEventPublisher`` — **omitted** (ZMQ not vendored; the
    C242 SHM broadcast primitive is the substitute for cross-process
    pub/sub).
  * ``KVEventsConfig`` — minimal Pydantic-free dataclass (config
    is duck-typed).
  * Wire-format: kept compatible with vLLM's JSON layout (events are
    ``asdict()``-serializable so consumers across processes can
    reconstruct them via msgpack / JSON).

The goal is to ship the **event types + in-process publisher +
aggregator** so the C252 coordinator can route events between
producer / consumer engines on a single dev box.  Cross-process
ZMQ/TCP pub/sub is deferred until the production wire-up.

References:
  * vLLM kv_events:
    ``vllm/distributed/kv_events.py``.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import count
from queue import Empty, Queue
from typing import Any


# Storage-medium tags (matches vLLM MEDIUM_* constants).
MEDIUM_GPU = "GPU"
MEDIUM_CPU = "CPU"
MEDIUM_FS = "FS"
MEDIUM_OBJ = "OBJ"

# Locality tags (relative to the publisher).
LOCALITY_LOCAL = "LOCAL"
LOCALITY_REMOTE = "REMOTE"


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass
class KVCacheEvent:
    """Base class for all KV cache-related events.

    Mirrors ``vllm.distributed.kv_events.KVCacheEvent``.  Subclasses
    are dataclasses for JSON / msgpack serialization compatibility.
    """


@dataclass
class BlockStored(KVCacheEvent):
    """A block was stored in the cache (KV transfer trigger).

    Mirrors ``vllm.distributed.kv_events.BlockStored``.  Carries the
    block hashes + token IDs + medium + cache-spec metadata so
    consumers can classify and reconstruct the block.
    """

    block_hashes: list[int]
    parent_block_hash: int | None
    token_ids: list[int]
    block_size: int
    medium: str
    lora_id: int | None = None
    lora_name: str | None = None
    extra_keys: list[tuple[Any, ...] | None] | None = None
    group_idx: int | None = None
    kv_cache_spec_kind: str | None = None
    kv_cache_spec_sliding_window: int | None = None
    locality: str | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.parent_block_hash,
                tuple(self.token_ids),
                self.block_size,
                self.lora_id,
                self.medium,
                tuple(self.extra_keys) if self.extra_keys else None,
                self.group_idx,
                self.kv_cache_spec_kind,
                self.kv_cache_spec_sliding_window,
                self.locality,
            ),
        )


@dataclass
class BlockRemoved(KVCacheEvent):
    """A block was evicted from the cache.

    Mirrors ``vllm.distributed.kv_events.BlockRemoved``.  Carries
    only the block hashes + medium so consumers can update their
    cache index.
    """

    block_hashes: list[int]
    medium: str
    group_idx: int | None = None
    locality: str | None = None

    def __hash__(self) -> int:
        return hash(
            (
                tuple(self.block_hashes),
                self.medium,
                self.group_idx,
                self.locality,
            ),
        )


@dataclass
class AllBlocksCleared(KVCacheEvent):
    """All blocks have been cleared from the cache.

    Mirrors ``vllm.distributed.kv_events.AllBlocksCleared``.
    """


# ---------------------------------------------------------------------------
# Batch + aggregator
# ---------------------------------------------------------------------------


@dataclass
class KVEventBatch:
    """A batch of KV cache events emitted at a single point in time.

    Mirrors ``vllm.distributed.kv_events.KVEventBatch`` (which is a
    msgspec.Struct; my-vllm uses a dataclass).  Carries a timestamp
    + data-parallel rank so consumers can attribute events.
    """

    ts: float
    events: list[KVCacheEvent]
    data_parallel_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.events:
            # Defensive: empty batches carry no information.
            pass


class KVEventAggregator:
    """Aggregates KV events across multiple workers.

    Mirrors ``vllm.distributed.kv_events.KVEventAggregator``.  Tracks
    how many times each event appears across worker batches and
    returns only those that were emitted by **all** workers (the
    common-events API used by consumers to deduplicate redundant
    BlockStored events that all TP ranks emit independently).
    """

    __slots__ = ("_event_counter", "_num_workers")

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be greater than zero.")
        self._event_counter: Counter[KVCacheEvent] = Counter()
        self._num_workers: int = num_workers

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """Add events from one worker batch."""
        if not isinstance(events, list):
            raise TypeError(
                f"events must be a list of KVCacheEvent, got {type(events)}",
            )
        self._event_counter.update(events)

    def get_common_events(self) -> list[KVCacheEvent]:
        """Return events that appeared in all workers."""
        return [
            event
            for event, count in self._event_counter.items()
            if count == self._num_workers
        ]

    def get_all_events(self) -> list[KVCacheEvent]:
        """Return all events (with duplicates)."""
        return list(self._event_counter.elements())

    def clear_events(self) -> None:
        self._event_counter.clear()

    def increment_workers(self, count: int = 1) -> None:
        if count <= 0:
            raise ValueError("count must be positive.")
        self._num_workers += count

    def reset_workers(self) -> None:
        self._num_workers = 1

    def get_number_of_workers(self) -> int:
        return self._num_workers

    def __repr__(self) -> str:
        return (
            f"<KVEventAggregator workers={self._num_workers}, "
            f"events={len(self._event_counter)}>"
        )


# ---------------------------------------------------------------------------
# Publisher ABC
# ---------------------------------------------------------------------------


class EventPublisher(ABC):
    """Lightweight publisher for KV event batches.

    Mirrors ``vllm.distributed.kv_events.EventPublisher``.  Subclasses
    implement the actual transport (ZMQ, TCP, in-memory, etc.).
    Implementations should guarantee at-least-once delivery and
    monotonic ordering (e.g., via sequence numbers).
    """

    def __init__(self, data_parallel_rank: int = 0) -> None:
        self._data_parallel_rank = data_parallel_rank

    @property
    def data_parallel_rank(self) -> int:
        return self._data_parallel_rank

    @abstractmethod
    def publish(self, events: KVEventBatch) -> None:
        """Emit a batch of events."""

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the publisher."""


class NullEventPublisher(EventPublisher):
    """No-op implementation (default when events are disabled)."""

    def publish(self, events: KVEventBatch) -> None:
        return

    def shutdown(self) -> None:
        return


class InMemoryEventPublisher(EventPublisher):
    """In-process publisher for development + single-process testing.

    C252: a simple pub/sub backed by a thread-safe queue.  Subscribers
    receive every batch via a callback registered at construction
    time.  This is the my-vllm substitute for the ZMQ-based
    ``ZmqEventPublisher`` (ZMQ not vendored).

    The publisher:
      * Auto-assigns the bound ``data_parallel_rank`` to batches
        that don't already have one.
      * Maintains a bounded replay buffer (last ``buffer_steps``
        batches) so late-attaching subscribers can replay missed
        events.
      * Spawns a background thread to deliver events to subscribers
        (off the publish-thread critical path).
    """

    SHUTDOWN_TIMEOUT: float = 1.0
    END_SEQ = (-1).to_bytes(8, "big", signed=True)

    def __init__(
        self,
        data_parallel_rank: int = 0,
        buffer_steps: int = 10_000,
        max_queue_size: int = 100_000,
    ) -> None:
        super().__init__(data_parallel_rank)
        self._event_queue: Queue[KVEventBatch | None] = Queue(maxsize=max_queue_size)
        self._buffer: deque[tuple[int, bytes]] = deque(maxlen=buffer_steps)
        self._seq_gen = count()
        self._subscribers: list[Callable[[KVEventBatch], None]] = []
        self._replay_subscribers: list[Callable[[int], list[bytes]]] = []
        self._running = True
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._publisher_thread, daemon=True,
            name="kv-events-in-memory-publisher",
        )
        self._thread.start()

    def add_subscriber(self, callback: Callable[[KVEventBatch], None]) -> None:
        """Register a callback that receives every batch."""
        with self._lock:
            self._subscribers.append(callback)

    def remove_subscriber(self, callback: Callable[[KVEventBatch], None]) -> None:
        """Unregister a callback."""
        with self._lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def publish(self, events: KVEventBatch) -> None:
        if not self._running:
            raise RuntimeError("Publisher is closed")
        if events.data_parallel_rank is None:
            events.data_parallel_rank = self._data_parallel_rank
        self._event_queue.put(events)

    def shutdown(self) -> None:
        self._running = False
        self._event_queue.put_nowait(None)
        start = time.time()
        while (
            not self._event_queue.empty()
            and (time.time() - start) < self.SHUTDOWN_TIMEOUT
        ):
            time.sleep(0.05)
        if self._thread.is_alive():
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT)
        with self._lock:
            self._subscribers.clear()
            self._replay_subscribers.clear()

    def _publisher_thread(self) -> None:
        while self._running or self._event_queue.qsize() > 0:
            try:
                batch = self._event_queue.get(timeout=0.1)
            except Empty:
                continue
            if batch is None:
                break
            try:
                seq = next(self._seq_gen)
                payload = self._encode(batch)
                seq_bytes = seq.to_bytes(8, "big")
                self._buffer.append((seq, payload))
                with self._lock:
                    subscribers = list(self._subscribers)
                for cb in subscribers:
                    try:
                        cb(batch)
                    except Exception:
                        # Subscriber callback raised — drop the event
                        # for that subscriber but keep publishing to
                        # the rest.
                        pass
                self._event_queue.task_done()
            except Exception:
                # Publish loop error — back off slightly to avoid a
                # tight error loop.
                time.sleep(0.05)

    def _encode(self, batch: KVEventBatch) -> bytes:
        """Serialize a batch for the replay buffer.

        Uses JSON since msgspec isn't vendored.  Production would
        switch to msgpack for size + speed.
        """
        import json

        def _default(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            if isinstance(obj, Enum):
                return obj.value
            return str(obj)

        return json.dumps(asdict(batch), default=_default).encode()

    def get_replay_buffer(self) -> list[tuple[int, bytes]]:
        """Snapshot the replay buffer (test helper)."""
        with self._lock:
            return list(self._buffer)


# ---------------------------------------------------------------------------
# EventPublisherFactory
# ---------------------------------------------------------------------------


class EventPublisherFactory:
    """Registry + factory for ``EventPublisher`` implementations.

    Mirrors ``vllm.distributed.kv_events.EventPublisherFactory``.
    """

    _registry: dict[str, Callable[..., EventPublisher]] = {
        "null": NullEventPublisher,
        "in_memory": InMemoryEventPublisher,
    }

    @classmethod
    def register_publisher(
        cls, name: str, ctor: Callable[..., EventPublisher],
    ) -> None:
        if name in cls._registry:
            raise KeyError(f"publisher '{name}' already registered")
        cls._registry[name] = ctor

    @classmethod
    def create(
        cls,
        publisher: str | None = None,
        data_parallel_rank: int = 0,
        **kwargs: Any,
    ) -> EventPublisher:
        """Create a publisher by name.

        Defaults to ``NullEventPublisher`` when ``publisher`` is
        ``None`` or invalid.
        """
        if publisher is None or publisher == "null":
            return NullEventPublisher()
        try:
            ctor = cls._registry[publisher]
        except KeyError as exc:
            raise ValueError(
                f"Unknown event publisher '{publisher}'",
            ) from exc
        return ctor(data_parallel_rank=data_parallel_rank, **kwargs)


# ---------------------------------------------------------------------------
# KVConnectorKVEvents — connector-side helper (C251 take_events hook)
# ---------------------------------------------------------------------------


class KVConnectorKVEvents:
    """Container for KV events from a connector.

    Mirrors ``vllm.distributed.kv_events.KVConnectorKVEvents`` (which
    is an ABC; my-vllm uses a concrete dataclass because the
    production use case is "connector stores events → scheduler
    drains them").
    """

    def __init__(self) -> None:
        self._events: list[KVCacheEvent] = []

    def add_events(self, events: list[KVCacheEvent]) -> None:
        """Append events to the buffer."""
        self._events.extend(events)

    def add_event(self, event: KVCacheEvent) -> None:
        """Append a single event."""
        self._events.append(event)

    def get_all_events(self) -> list[KVCacheEvent]:
        """Return all events (consuming them)."""
        events = list(self._events)
        self._events.clear()
        return events

    def take_events(self) -> list[KVCacheEvent]:
        """Alias for ``get_all_events`` (matches vLLM's ABC name)."""
        return self.get_all_events()

    def clear_events(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


__all__ = [
    "AllBlocksCleared",
    "BlockRemoved",
    "BlockStored",
    "EventPublisher",
    "EventPublisherFactory",
    "InMemoryEventPublisher",
    "KVCacheEvent",
    "KVConnectorKVEvents",
    "KVEventAggregator",
    "KVEventBatch",
    "LOCALITY_LOCAL",
    "LOCALITY_REMOTE",
    "MEDIUM_CPU",
    "MEDIUM_FS",
    "MEDIUM_GPU",
    "MEDIUM_OBJ",
    "NullEventPublisher",
]

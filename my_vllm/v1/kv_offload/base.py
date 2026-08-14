"""KV cache offloading — core abstractions.

Mirrors vLLM V1 vllm/v1/kv_offload/base.py (subset).

C255 surface:
- `OffloadKey` NewType + pack/unpack helpers (`make_offload_key`,
  `get_offload_block_hash`, `get_offload_group_idx`).
- Enums: `LookupResult` (MISS / HIT / HIT_PENDING / RETRY),
  `OffloadPolicy` (BLOCK_LEVEL / REQUEST_LEVEL),
  `Locality` (LOCAL / REMOTE).
- Dataclasses: `ReqContext`, `RequestOffloadingContext`,
  `ScheduleEndContext`, `PrepareStoreOutput`, `OffloadingEvent`,
  `CanonicalKVCacheTensor` / `CanonicalKVCacheRef` /
  `CanonicalKVCaches`, `TransferResult`.
- Load/Store spec hierarchy: `LoadStoreSpec` ABC +
  `BlockIDsLoadStoreSpec` ABC + `GPULoadStoreSpec` concrete.
- Metric metadata hierarchy:
  `OffloadingMetricMetadata` / `Counter` / `Gauge` / `Histogram`.
- ABCs:
  - `OffloadingManager` — scheduler-side: lookup / prepare_load /
    complete_load / prepare_store / complete_store /
    on_new_request / on_request_finished / take_events /
    on_schedule_end / has_pending_work / reset_cache /
    get_stats / shutdown.
  - `OffloadingWorker` — worker-side: submit_store / submit_load /
    get_finished / wait / shutdown.
  - `OffloadingSpec` — connector spec:
    get_manager() / get_worker(kv_caches) / build_metric_definitions().

C255 doesn't ship the concrete backends — those land with C256
(CPU/NVMe tiering) and C257 (simple KV offload path). The base +
config + factory + file_mapper together form the **plug-in
interface** that future backends slot into.

The base code is a faithful subset extraction from vLLM V1's
kv_offload/base.py — same field names, same method signatures,
same ABC enforcement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, NamedTuple, NewType

import numpy as np
import torch

if TYPE_CHECKING:
    from my_vllm.v1.kv_offload.config import OffloadingConfig


# ---------------------------------------------------------------------------
# OffloadKey — identifies an offloaded block (block hash + group index)
# ---------------------------------------------------------------------------


OffloadKey = NewType("OffloadKey", bytes)


def make_offload_key(block_hash: bytes, group_idx: int) -> OffloadKey:
    """Pack a block hash + group index into an `OffloadKey`.

    Encoding: `<block_hash><group_idx_be_u32>` — 4-byte big-endian
    unsigned int appended to the raw hash bytes. The encoding is
    deterministic + reversible via `get_offload_block_hash` /
    `get_offload_group_idx`.
    """
    return OffloadKey(block_hash + group_idx.to_bytes(4, "big", signed=False))


def get_offload_block_hash(key: OffloadKey) -> bytes:
    """Extract the block-hash portion of an `OffloadKey`."""
    return key[:-4]


def get_offload_group_idx(key: OffloadKey) -> int:
    """Extract the group-index portion of an `OffloadKey`."""
    return int.from_bytes(key[-4:], "big", signed=False)


# ---------------------------------------------------------------------------
# Per-request + per-step context types
# ---------------------------------------------------------------------------


@dataclass
class ReqContext:
    """Per-request context passed to OffloadingManager hooks.

    Carries the request ID + optional KV-transfer parameters (when the
    offload path is paired with a KV connector — used by C256/C257
    for cross-instance cache lookup).
    """

    req_id: str
    kv_transfer_params: dict[str, Any] | None = None


class LookupResult(Enum):
    """Result of `OffloadingManager.lookup()`."""

    MISS = auto()
    """Block not offloaded."""
    HIT = auto()
    """Block offloaded and ready to be read."""
    HIT_PENDING = auto()
    """Block found but not yet readable (transfer still in flight)."""
    RETRY = auto()
    """Lookup should be retried later (transient backend error)."""


class OffloadPolicy(Enum):
    """Per-request offload policy.

    `BLOCK_LEVEL` (default): offload only newly-computed blocks as
    they arrive. Prefix-hit blocks already offloaded by a prior
    request are skipped.

    `REQUEST_LEVEL`: offload all blocks for the request, including
    prefix hits. Used by tiers that need the complete KV context
    (e.g. some NVMe tiering scenarios).
    """

    BLOCK_LEVEL = "block_level"
    REQUEST_LEVEL = "request_level"


@dataclass
class RequestOffloadingContext:
    """Returned by `OffloadingManager.on_new_request`."""

    policy: OffloadPolicy = OffloadPolicy.BLOCK_LEVEL


class ScheduleEndContext(NamedTuple):
    """Per-step scheduling info passed to `on_schedule_end()`."""

    new_req_ids: Collection[str]
    """Request IDs scheduled for the first time this step."""
    preempted_req_ids: Collection[str]
    """Request IDs preempted this step."""


# ---------------------------------------------------------------------------
# LoadStoreSpec — describes a load/store operation
# ---------------------------------------------------------------------------


class LoadStoreSpec:
    """Base metadata for load/store operations.

    Subclasses are concrete descriptors of the source / destination
    medium. `GPULoadStoreSpec` describes GPU-side block IDs;
    concrete offload medium specs (CPU memory, NVMe file path, etc.)
    subclass this directly.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(...)"


class BlockIDsLoadStoreSpec(LoadStoreSpec, ABC):
    """Spec for load/store ops identified by block IDs.

    Stores the block IDs as a numpy int64 array for efficient
    bulk transfer.
    """

    def __init__(self, block_ids: list[int]):
        self.block_ids = np.array(block_ids, dtype=np.int64)

    def __repr__(self) -> str:
        return repr(self.block_ids)


class GPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing KV blocks to GPU memory.

    If there are multiple KV groups, blocks are ordered by group
    index. `group_sizes[i]` is the number of blocks in the i-th
    KV group, so `sum(group_sizes) == len(block_ids)`.
    `group_sizes=None` means a single KV group.

    If `block_indices` is given, each group (determined by
    `group_sizes`) of block IDs corresponds to logically contiguous
    blocks (e.g. blocks 5-10 of some request). `block_indices[i]`
    is the block index of the first block in group #i. Used to
    support off/loading from offloaded blocks larger than GPU
    blocks (the first GPU block per group may be unaligned to the
    offloaded block size).
    """

    def __init__(
        self,
        block_ids: list[int],
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
    ):
        super().__init__(block_ids)
        assert sum(group_sizes) == len(block_ids), (
            "sum(group_sizes) must equal len(block_ids); "
            f"got {sum(group_sizes)} vs {len(block_ids)}"
        )
        assert len(block_indices) == len(group_sizes), (
            "len(block_indices) must equal len(group_sizes); "
            f"got {len(block_indices)} vs {len(group_sizes)}"
        )
        self.group_sizes: Sequence[int] = group_sizes
        self.block_indices: Sequence[int] = block_indices


@dataclass
class PrepareStoreOutput:
    """Result of `OffloadingManager.prepare_store()`."""

    keys_to_store: list[OffloadKey]
    """Block keys that need to be stored."""
    store_spec: LoadStoreSpec
    """Spec for the store operation (e.g. file path / CPU buffer)."""
    evicted_keys: list[OffloadKey]
    """Keys evicted as a result of this store (LRU pressure)."""


# ---------------------------------------------------------------------------
# Locality + events
# ---------------------------------------------------------------------------


class Locality(Enum):
    """Locality of a tier's storage relative to the publishing instance."""

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


@dataclass
class OffloadingEvent:
    """KV-event-style notification emitted by an offload manager."""

    keys: list[OffloadKey]
    medium: str
    # True if blocks are removed, False if stored.
    removed: bool
    locality: Locality | None = None


# ---------------------------------------------------------------------------
# Metric metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OffloadingMetricMetadata:
    """Base metadata for an offloading metric definition."""

    documentation: str
    labelnames: tuple[str, ...] = ()


@dataclass(frozen=True)
class OffloadingCounterMetadata(OffloadingMetricMetadata):
    """Counter metric metadata (monotonically increasing)."""


@dataclass(frozen=True)
class OffloadingGaugeMetadata(OffloadingMetricMetadata):
    """Gauge metric metadata (point-in-time value)."""


@dataclass(frozen=True)
class OffloadingHistogramMetadata(OffloadingMetricMetadata):
    """Histogram metric metadata (value distribution)."""

    buckets: tuple[float, ...] | None = None


@dataclass(frozen=True)
class OffloadingKVEventsConfig:
    """Offloading-specific KV-event configuration.

    Forwarded to the connector so it knows whether to emit
    BlockStored / BlockRemoved events alongside its offload work.
    """

    enable_kv_cache_events: bool
    self_describing_kv_events: bool


# ---------------------------------------------------------------------------
# Canonical KV cache representation
# ---------------------------------------------------------------------------


@dataclass
class CanonicalKVCacheTensor:
    """A canonicalized KV cache tensor with num_blocks leading.

    For attention backends where the raw tensor has num_blocks at a
    non-leading physical dimension (e.g. FlashAttention's
    (2, num_blocks, ...) layout), the tensor is split so each
    resulting CanonicalKVCacheTensor starts with (num_blocks, ...).
    """

    tensor: torch.Tensor
    page_size_bytes: int


@dataclass
class CanonicalKVCacheRef:
    """Per-layer (or group of layers) reference to a CanonicalKVCacheTensor.

    Records the un-padded page size used by that layer.
    """

    tensor_idx: int
    page_size_bytes: int


@dataclass
class CanonicalKVCaches:
    """Canonicalized block-level representation of the KV caches.

    Composed of:
    - Unique list of KV cache data tensors, each with shape
      (num_blocks, page_size_in_bytes) and int8 dtype.
    - Per-group data references of the tensors — how each KV cache
      group maps to the tensors.
    """

    tensors: list[CanonicalKVCacheTensor]
    group_data_refs: list[list[CanonicalKVCacheRef]]


@dataclass
class TransferResult:
    """Result of an async KV transfer (used by `OffloadingWorker.get_finished`)."""

    job_id: int
    success: bool
    transfer_size: int | None = None
    transfer_time: float | None = None


# ---------------------------------------------------------------------------
# Scheduler-side: OffloadingManager
# ---------------------------------------------------------------------------


class OffloadingManager(ABC):
    """Scheduler-side offload manager.

    Tracks which blocks are offloaded and where. Primitives:
    - `lookup()` — check whether a single block is offloaded + ready.
    - `prepare_load()` — prepare blocks to be read; protects from
      eviction until `complete_load` is called.
    - `touch()` — mark blocks as recently used (LRU bump), without
      needing a load.
    - `complete_load()` — release eviction protection after a load.
    - `prepare_store()` — prepare blocks to be written; returns a
      `PrepareStoreOutput` with the store spec + evicted keys.
    - `complete_store()` — finalize a store (or roll back on failure).
    - `on_new_request()` — called once per request; returns the
      `RequestOffloadingContext` describing the offload policy.
    - `on_request_finished()` — request cleanup hook.
    - `take_events()` — drain offload events for the event bus.
    - `on_schedule_end()` — end-of-step flush hook.
    - `has_pending_work()` — engine keep-stepping signal.
    - `reset_cache()` — evict all tracked blocks.
    - `get_stats()` — collected metrics (None when disabled).
    - `shutdown()` — release resources.
    """

    @abstractmethod
    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult:
        """Check whether a single block is offloaded and ready to be read."""
        pass

    @abstractmethod
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        """Prepare given blocks to be read (protects from eviction)."""
        pass

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """Mark blocks as recently used (LRU bump)."""
        return None

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        """Release eviction protection after a load."""
        return None

    @abstractmethod
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        """Prepare given blocks to be offloaded.

        Returns `None` if the blocks cannot be stored.
        """
        pass

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        """Finalize a previous store (or roll back on failure)."""
        return None

    @abstractmethod
    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext:
        """Return the offload policy for a new request."""
        pass

    def on_request_finished(self, req_context: ReqContext) -> None:
        """Called when a request has finished.

        By the time this is called, the scheduler will issue no more
        submit-side calls for this request (no more `prepare_store`
        or `prepare_load`). Completion callbacks for already-submitted
        transfers may still arrive afterward.

        Managers that cascade to lower tiers should delay those
        tiers' `on_request_finished` calls until no more lower-tier
        submit calls can be issued.
        """
        return None

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Take the offload events since the last call.

        A tier manager emits only events for storage state it owns.
        A composing manager may aggregate child event streams but
        should not synthesize events on behalf of a child tier.
        """
        return ()

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """Called once at the end of each scheduler step.

        Managers may flush deferred work (e.g. batched promotions).
        """
        return None

    def has_pending_work(self) -> bool:
        """Whether this manager needs the engine to keep stepping.

        While True, `on_schedule_end` and `get_finished_jobs`
        continue to be called even when no requests are scheduled.
        """
        return False

    def reset_cache(self) -> None:
        """Evict all tracked blocks and reset internal state."""
        return None

    def get_stats(self) -> Any:
        """Return collected metrics since last call, or None if disabled."""
        return None

    def shutdown(self) -> None:
        """Shutdown the manager and release any resources."""
        return None


# ---------------------------------------------------------------------------
# Worker-side: OffloadingWorker
# ---------------------------------------------------------------------------


class OffloadingWorker(ABC):
    """Worker-side offload worker.

    Runs in the worker process; performs async KV transfers for ONE
    offloaded medium (e.g. CPU memory). Direction is explicit via
    `submit_store` / `submit_load`, so there is no (src, dst)
    routing at this layer.
    """

    @abstractmethod
    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Async GPU → offloaded medium."""
        pass

    @abstractmethod
    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        """Async offloaded medium → GPU."""
        pass

    @abstractmethod
    def get_finished(self) -> list[TransferResult]:
        """Drain finished transfer results."""
        pass

    @abstractmethod
    def wait(self, job_ids: set[int]) -> None:
        """Block until the given job IDs complete."""
        pass

    def shutdown(self) -> None:
        """Release resources held by the worker."""
        return None


# ---------------------------------------------------------------------------
# Connector spec: OffloadingSpec
# ---------------------------------------------------------------------------


class OffloadingSpec(ABC):
    """Connector spec for an offload medium.

    Carries the `OffloadingConfig` + extra_config; produces an
    `OffloadingManager` (scheduler-side) and an `OffloadingWorker`
    (worker-side) that together implement the offload flow for one
    backing medium (CPU memory, NVMe, etc.).
    """

    def __init__(self, config: "OffloadingConfig"):
        self.config = config
        self.extra_config = config.extra_config
        self.kv_events_config = OffloadingKVEventsConfig(
            enable_kv_cache_events=config.enable_kv_cache_events,
            self_describing_kv_events=bool(
                self.extra_config.get("self_describing_kv_events", False)
            ),
        )
        # When True, only prompt (prefill) blocks are offloaded;
        # decode-phase blocks are skipped. Useful for reasoning
        # models that strip thinking between turns.
        self.offload_prompt_only: bool = bool(
            self.extra_config.get("offload_prompt_only", True)
        )
        # Per-group token spans (accounts for context parallelism).
        self.tokens_per_block = tuple(
            group.tokens_per_block for group in config.groups
        )
        self.tokens_per_hash = config.cache.tokens_per_hash
        self.blocks_per_chunk = config.cache.blocks_per_chunk

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, "OffloadingMetricMetadata"]:
        """Return Prometheus metric definitions emitted by this spec."""
        return {}

    @abstractmethod
    def get_manager(self) -> OffloadingManager:
        """Scheduler-side manager for tracking offloaded blocks + evictions."""
        pass

    @abstractmethod
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        """Worker-side async transfer worker for this medium."""
        pass


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "BlockIDsLoadStoreSpec",
    "CanonicalKVCacheRef",
    "CanonicalKVCacheTensor",
    "CanonicalKVCaches",
    "GPULoadStoreSpec",
    "LoadStoreSpec",
    "Locality",
    "LookupResult",
    "OffloadKey",
    "OffloadPolicy",
    "OffloadingCounterMetadata",
    "OffloadingEvent",
    "OffloadingGaugeMetadata",
    "OffloadingHistogramMetadata",
    "OffloadingKVEventsConfig",
    "OffloadingManager",
    "OffloadingMetricMetadata",
    "OffloadingSpec",
    "OffloadingWorker",
    "PrepareStoreOutput",
    "ReqContext",
    "RequestOffloadingContext",
    "ScheduleEndContext",
    "TransferResult",
    "get_offload_block_hash",
    "get_offload_group_idx",
    "make_offload_key",
]
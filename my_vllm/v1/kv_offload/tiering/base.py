"""KV offload tiering — abstract interfaces + data types.

Mirrors vLLM V1 vllm/v1/kv_offload/tiering/base.py.

C256 ships:
- `TieringOffloadingMetrics` (Prometheus metric name constants).
- `JobMetadata` / `JobResult` dataclasses for async transfer
  bookkeeping between secondary tiers and the composing manager.
- `ParentManager` ABC — interface secondary tiers use to call
  back into the tiering manager (lookup / create_store_job /
  on_new_request / on_request_finished).
- `SecondaryTierManager` ABC — abstract secondary tier (e.g.
  Storage / Network / ObjectStore). `lookup` / `submit_store` /
  `submit_load` / `get_finished_jobs` / `on_schedule_end` /
  `reset_cache` / `shutdown`.

**vLLM scope reductions**: my-vllm has no concrete secondary
tier (Storage / FS / P2P / ObjectStore) wired in C256 — those
land when a real workload needs them. The ABC surface +
registry + composing manager are enough to plug them in
later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np

from my_vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)


# Type alias for async transfer job IDs.
JobId = int


class TieringOffloadingMetrics:
    """Prometheus metric name constants for the tiering manager.

    Mirrors vLLM V1 vllm/v1/kv_offload/tiering/base.py:TieringOffloadingMetrics.
    """

    LOOKUP_SYNC_DELAY = "vllm:kv_offload_tiering_lookup_sync_delay_seconds"
    LOOKUP_ASYNC_DELAY = "vllm:kv_offload_tiering_lookup_async_delay_seconds"


@dataclass
class JobMetadata:
    """Metadata for an in-flight async transfer job."""

    job_id: JobId
    keys: Collection[OffloadKey]
    block_ids: np.ndarray
    is_promotion: bool
    req_context: ReqContext


@dataclass
class JobResult:
    """Result of an async transfer (success or failure)."""

    job_id: JobId
    success: bool


class ParentManager(ABC):
    """Interface for secondary tiers to call back into the composing manager.

    Required call sequence per remote request:
    1. `on_new_request(req_context)` — set up per-request state.
    2. `lookup(key, req_context)` — check block availability
       (repeat per block).
    3. `create_store_job(keys, req_context)` — pin blocks + get
       a job handle.
    4. `on_request_finished(req_context)` — clean up per-request
       state (must run even if no blocks were found, to avoid
       leaking async-lookup state).
    """

    @abstractmethod
    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext: ...

    @abstractmethod
    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult: ...

    @abstractmethod
    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> JobMetadata: ...

    @abstractmethod
    def on_request_finished(self, req_context: ReqContext) -> None: ...


class SecondaryTierManager(ABC):
    """Abstract interface for managing a single non-primary offloading tier.

    Secondary tiers can't access GPU directly — all data transfers
    flow through the CPU (primary) tier:
      Store:  GPU → CPU (primary) → secondary (cascade)
      Load:   secondary → CPU (primary) → GPU (promotion)

    All methods run in the scheduler process and must be
    lightweight + non-blocking. `submit_*` submits an async job;
    `get_finished_jobs` polls for completion.
    """

    def __init__(
        self,
        offloading_spec: Any,
        primary_kv_view: "memoryview",
        tier_type: str,
    ) -> None:
        self._offloading_spec = offloading_spec
        self._primary_kv_view: "memoryview" = primary_kv_view
        self.tier_type: str = tier_type

    @abstractmethod
    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult:
        """Check whether a block exists in this secondary tier.

        Returns HIT / MISS / RETRY (RETRY = transfer in flight,
        retry later).
        """
        pass

    @abstractmethod
    def submit_store(self, job_metadata: JobMetadata) -> None:
        """Submit an async primary-tier → this-tier store job."""
        pass

    @abstractmethod
    def submit_load(self, job_metadata: JobMetadata) -> None:
        """Submit an async this-tier → primary-tier load job."""
        pass

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Return all jobs that completed since the last call."""
        pass

    @abstractmethod
    def on_schedule_end(self) -> None:
        """End-of-step hook for batched work flush."""
        pass

    @abstractmethod
    def has_pending_work(self) -> bool:
        """Whether this tier needs the engine to keep stepping."""
        pass

    @abstractmethod
    def reset_cache(self) -> None:
        """Evict all tracked blocks + reset state."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources held by this tier."""
        pass


__all__ = [
    "JobId",
    "JobMetadata",
    "JobResult",
    "ParentManager",
    "SecondaryTierManager",
    "TieringOffloadingMetrics",
]
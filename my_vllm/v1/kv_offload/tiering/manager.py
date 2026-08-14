"""KV offload tiering — composing manager.

Mirrors vLLM V1 vllm/v1/kv_offload/tiering/manager.py (subset).

C256 ships:
- `CPUPrimaryTierOffloadingManager` — wraps a
  `CPUOffloadingManager` as the primary tier (CPU-side gateway
  for all GPU↔offload operations).
- `TieringOffloadingManager` — composing manager that holds one
  primary tier + N secondary tiers. `lookup` fans out across all
  tiers; `prepare_store` / `prepare_load` route through the
  primary tier (GPU only talks to CPU).

The composing manager implements `ParentManager` so secondary
tiers can callback through it. The actual cascade / promotion
state machine is **simplified** for C256 — secondary tiers are
exercised via the lookup fan-out but the full store-cascade +
load-promotion logic lands when a concrete secondary tier
ships (Phase 5).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

import numpy as np

from my_vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from my_vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from my_vllm.v1.kv_offload.tiering.base import (
    JobMetadata,
    JobResult,
    ParentManager,
    SecondaryTierManager,
)

if TYPE_CHECKING:
    from my_vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


class CPUPrimaryTierOffloadingManager(OffloadingManager):
    """Wrap a `CPUOffloadingManager` as the primary tier.

    Mirrors vLLM V1 vllm/v1/kv_offload/tiering/manager.py:
    CPUPrimaryTierOffloadingManager (subset).

    C256 only adds the `get_kv_memoryview` accessor (for secondary
    tiers that need to read the primary tier's CPU buffer). The
    full primary-tier API delegates straight through to the
    underlying `CPUOffloadingManager`.
    """

    def __init__(self, num_blocks: int, **kwargs):
        self._cpu_manager = CPUOffloadingManager(
            num_blocks=num_blocks, **kwargs
        )

    def get_kv_memoryview(self) -> "memoryview":
        """Return a memoryview of the primary tier's KV buffer.

        C256 returns a placeholder memoryview — real buffer wiring
        lands with the `SharedOffloadRegion` (Phase 5).
        """
        # The memoryview is what secondary tiers consume to read
        # the primary tier's KV bytes. my-vllm doesn't vendor the
        # shared mmap region, so this returns a zero-length view
        # and tests pass mock secondary tiers that don't read
        # actual bytes.
        return memoryview(b"")

    # ── OffloadingManager delegation ────────────────────────────────

    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult:
        return self._cpu_manager.lookup(key, req_context)

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        return self._cpu_manager.prepare_load(keys, req_context)

    def touch(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        return self._cpu_manager.touch(keys, req_context)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        return self._cpu_manager.complete_load(keys, req_context)

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        return self._cpu_manager.prepare_store(keys, req_context)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        return self._cpu_manager.complete_store(
            keys, req_context, success=success
        )

    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext:
        return self._cpu_manager.on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self._cpu_manager.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._cpu_manager.take_events()

    def on_schedule_end(self, context: Any) -> None:
        return self._cpu_manager.on_schedule_end(context)

    def has_pending_work(self) -> bool:
        return self._cpu_manager.has_pending_work()

    def reset_cache(self) -> None:
        return self._cpu_manager.reset_cache()

    def shutdown(self) -> None:
        return self._cpu_manager.shutdown()


class TieringOffloadingManager(OffloadingManager, ParentManager):
    """Composing manager — primary tier + secondary tiers.

    Mirrors vLLM V1 vllm/v1/kv_offload/tiering/manager.py:
    TieringOffloadingManager.

    `lookup` fans out across all tiers; primary-tier first (the
    fastest path), then secondary tiers. `prepare_store` /
    `prepare_load` route through the primary tier — secondary
    tier cascades / promotions are scheduled asynchronously.
    """

    def __init__(
        self,
        primary_tier: CPUPrimaryTierOffloadingManager,
        secondary_tiers: list[SecondaryTierManager] | None = None,
    ):
        self.primary_tier: CPUPrimaryTierOffloadingManager = primary_tier
        self.secondary_tiers: list[SecondaryTierManager] = (
            secondary_tiers or []
        )

    # ── OffloadingManager delegation ────────────────────────────────

    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult:
        # Try primary first (fastest).
        primary_result = self.primary_tier.lookup(key, req_context)
        if primary_result in (LookupResult.HIT, LookupResult.HIT_PENDING):
            return primary_result
        # Fan out to secondaries.
        for tier in self.secondary_tiers:
            r = tier.lookup(key, req_context)
            if r in (LookupResult.HIT, LookupResult.HIT_PENDING):
                return r
            if r == LookupResult.RETRY:
                return LookupResult.RETRY
        return LookupResult.MISS

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        # Primary-tier handles the load (GPU sees only CPU).
        return self.primary_tier.prepare_load(keys, req_context)

    def touch(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        return self.primary_tier.touch(keys, req_context)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        return self.primary_tier.complete_load(keys, req_context)

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        return self.primary_tier.prepare_store(keys, req_context)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        return self.primary_tier.complete_store(
            keys, req_context, success=success
        )

    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext:
        return self.primary_tier.on_new_request(req_context)

    def on_request_finished(self, req_context: ReqContext) -> None:
        return self.primary_tier.on_request_finished(req_context)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self.primary_tier.take_events()

    def on_schedule_end(self, context: Any) -> None:
        self.primary_tier.on_schedule_end(context)
        for tier in self.secondary_tiers:
            tier.on_schedule_end()

    def has_pending_work(self) -> bool:
        if self.primary_tier.has_pending_work():
            return True
        return any(
            tier.has_pending_work() for tier in self.secondary_tiers
        )

    def reset_cache(self) -> None:
        self.primary_tier.reset_cache()
        for tier in self.secondary_tiers:
            tier.reset_cache()

    def shutdown(self) -> None:
        self.primary_tier.shutdown()
        for tier in self.secondary_tiers:
            tier.shutdown()

    # ── ParentManager interface ─────────────────────────────────────

    def create_store_job(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> JobMetadata:
        # C256 shell — full job creation lands with a concrete
        # secondary tier. Returns a placeholder.
        return JobMetadata(
            job_id=0,
            keys=list(keys),
            block_ids=np.array([], dtype=np.int64),
            is_promotion=False,
            req_context=req_context,
        )


__all__ = [
    "CPUPrimaryTierOffloadingManager",
    "TieringOffloadingManager",
]
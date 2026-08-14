"""KV offload CPU backend — `CPUOffloadingManager`.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/manager.py.

C256: real `CPUOffloadingManager` impl. Owns:
- A pluggable `CachePolicy` (LRU by default; ARC shell registered).
- A simple `block_id` pool (block_pool: free list + allocated
  count). The manager hands out block IDs to the worker, who
  fills them with KV data.
- The `prepare_store` / `complete_store` / `prepare_load` /
  `complete_load` ref-counting protocol — eviction-protection
  for in-flight transfers.
- An `events` list (when `enable_events=True`) of
  `OffloadingEvent` records for the C252 event bus.
- An optional `counts` LRU (when `store_threshold >= 2`) that
  tracks per-key lookup frequency to filter which blocks are
  eligible for offloading.

my-vllm NOTE: the vLLM manager uses Prometheus metrics
(`OffloadingConnectorStats`) for `get_stats()`. We return None
today (Phase 5 wires the metric infrastructure); the bookkeeping
counters are kept internally so the Prometheus emission lands
without API churn.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Literal

from my_vllm.distributed.kv_events import MEDIUM_CPU
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
from my_vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from my_vllm.v1.kv_offload.cpu.policies import BlockStatus
from my_vllm.v1.kv_offload.cpu.policies import CachePolicy
from my_vllm.v1.kv_offload.cpu.policies import LRUCachePolicy
from my_vllm.v1.kv_offload.cpu.policies import ARCCachePolicy


_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """`OffloadingManager` with a pluggable `CachePolicy` (LRU or ARC).

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/manager.py:CPUOffloadingManager.

    The manager owns all shared logic — ref-counting, event
    emission, block-pool management, and the prepare/load/store
    skeletons. Policy-specific block organization + eviction
    decisions are delegated to the `CachePolicy`.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: str = MEDIUM_CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        self._num_evictable_cache_blocks: int = 0
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Lookup-count tracker for store_threshold ≥ 2 filtering.
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # ── Block pool helpers ──────────────────────────────────────────

    def _get_num_free_blocks(self) -> int:
        return (
            len(self._free_list)
            + self._num_blocks
            - self._num_allocated_blocks
        )

    def _allocate_blocks(
        self, keys: list[OffloadKey]
    ) -> list[BlockStatus]:
        num_fresh = min(
            len(keys), self._num_blocks - self._num_allocated_blocks
        )
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused, (
            f"free list underflow: have {len(self._free_list)}, "
            f"need {num_reused}"
        )

        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # ── OffloadingManager interface ─────────────────────────────────

    def on_new_request(
        self, req_context: ReqContext
    ) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    def lookup(
        self, key: OffloadKey, req_context: ReqContext
    ) -> LookupResult:
        del req_context
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        del req_context
        blocks: list[BlockStatus] = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, (
                f"Block {key!r} is not ready for reading "
                f"(ref_cnt={block.ref_cnt})"
            )
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def touch(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        self._policy.touch(keys, req_context)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        del req_context
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, (
                f"Block {key!r} ref_cnt is already 0"
            )
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1
                self._policy.mark_evictable(key)

    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        del req_context
        if self.counts is not None:
            num_keys = len(keys)
            keys = [
                k for k in keys
                if self.counts.get(k, 0) >= self.store_threshold
            ]
            self.stores_skipped_in_current_batch += num_keys - len(keys)

        # Skip blocks that are already stored.
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = (
            len(keys_to_store) - self._get_num_free_blocks()
        )

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # Protect the original input keys — a block that's
            # already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # Cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        del req_context
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def reset_cache(self) -> None:
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0
        self._free_list.clear()
        self._num_allocated_blocks = 0
        if self.counts is not None:
            self.counts.clear()
        self.allocation_sizes_in_current_batch.clear()
        self.stores_skipped_in_current_batch = 0
        if self.events is not None:
            self.events.clear()

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> None:
        # Phase 5: emit Prometheus metrics via OffloadingConnectorStats.
        # For now the counters are kept internally (see allocation_sizes +
        # stores_skipped) so the Prometheus path lands without API churn.
        return None


__all__ = ["CPUOffloadingManager"]
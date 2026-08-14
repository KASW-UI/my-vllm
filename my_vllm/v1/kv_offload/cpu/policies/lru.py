"""KV offload CPU backend — LRU eviction policy.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/lru.py.

C256: real `LRUCachePolicy` impl. Maintains two structures:
- `blocks`: full dict of `OffloadKey -> BlockStatus` for fast
  lookup.
- `evictable_blocks`: OrderedDict of evictable blocks (ref_cnt
  == 0) in LRU order — head is the LRU candidate, tail is MRU.
  Insertion / deletion / move-to-end are O(1).

A "use" is recorded when:
1. The key is first inserted (store).
2. A load job completes (ref_cnt transitions 1 → 0).
3. `touch()` is called for the key.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from my_vllm.v1.kv_offload.base import OffloadKey, ReqContext
from my_vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class LRUCachePolicy(CachePolicy):
    """LRU eviction policy with a dedicated evictable list.

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/lru.py:LRUCachePolicy.
    """

    def __init__(self, cache_capacity: int):
        del cache_capacity  # Capacity is enforced by the manager; the
        # policy itself is unbounded (manager caps the underlying
        # block pool).
        # Blocks with ref_cnt == 0 (idle), ordered LRU→MRU.
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        # All blocks (including in-flight).
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    def touch(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        del req_context
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)
            # Active blocks (ref_cnt > 0) are not in `evictable_blocks`
            # so they're untouched here — they reach the tail when
            # their loads complete (ref_cnt → 0 → mark_evictable).

    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()

    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, _ in self.evictable_blocks.items():
            if key in protected:
                continue
            block = self.blocks[key]
            assert block.ref_cnt == 0, (
                f"Evictable block {key!r} has non-zero ref_cnt={block.ref_cnt}"
            )
            candidates.append((key, block))
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None

        for key, _ in candidates:
            del self.evictable_blocks[key]
            del self.blocks[key]
        return candidates

    def mark_evictable(self, key: OffloadKey) -> None:
        # Called when ref_cnt transitions to 0 (store completes or
        # all loads complete).
        self.evictable_blocks[key] = None

    def mark_non_evictable(self, key: OffloadKey) -> None:
        # Called when ref_cnt transitions from 0 → 1.
        del self.evictable_blocks[key]


__all__ = ["LRUCachePolicy"]
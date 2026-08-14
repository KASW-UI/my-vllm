"""KV offload CPU backend — CachePolicy ABC + BlockStatus.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/base.py.

C256: `BlockStatus` (ctypes.Structure carrying ref_cnt + block_id)
+ `CachePolicy` ABC.

`ref_cnt = -1` marks a block as "not yet ready" (a write is
in-flight, so reads must wait for `complete_store`).

The `evict()` method must be atomic — returning None means no
state changed.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from my_vllm.v1.kv_offload.base import OffloadKey, ReqContext


class BlockStatus(ctypes.Structure):
    """Offloading status for one CPU block.

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/base.py:BlockStatus.

    Fields:
    - `ref_cnt` (c_int32): current reference count for in-flight
      transfers. `-1` means "not yet ready" (write-pending).
    - `block_id` (c_int64): physical CPU buffer slot index.
    """

    _fields_ = [
        ("ref_cnt", ctypes.c_int32),
        ("block_id", ctypes.c_int64),
    ]

    def __init__(self, block_id: int) -> None:
        super().__init__()
        # Initialize as "not ready" (ref_cnt = -1).
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """True iff the block is ready to be read."""
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """Abstract base — block organization + replacement decisions.

    Encapsulates both data structures and eviction logic. LRU and
    ARC differ in both dimensions (ARC's ghost lists + adaptive
    target_t1_size live at the intersection of storage and
    eviction, so they cannot be separated cleanly).
    """

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...

    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        """Find a block by key. None if not present."""

    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists."""

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (cleanup after a failed store)."""

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """Mark blocks as recently used (LRU bump)."""

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """Evict exactly n blocks, skipping any in `protected`.

        Returns the (key, block) pairs for the evicted blocks, or
        None if n evictions can't be satisfied. Atomic: on None,
        no state changes.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all blocks (regardless of ref_cnt) + reset ghost lists."""

    def mark_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions to 0."""
        return None

    def mark_non_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions from 0."""
        return None


__all__ = ["BlockStatus", "CachePolicy"]
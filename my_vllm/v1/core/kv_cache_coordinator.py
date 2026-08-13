"""KVCacheCoordinator — top-level coordinator across KV cache groups.

Mirrors vLLM V1 v1/core/kv_cache_coordinator.py:60.

Coordinator hierarchy (P0-3):
                KVCacheManager (scheduler-facing facade)
                          │
                          ▼
                KVCacheCoordinator (ABC)
                          │
                          ▼
               UnitaryKVCacheCoordinator  (single group)
                          │
                          ▼
                  SingleTypeKVCacheManager
                          │
                          ▼
                  FullAttentionManager

vLLM provides three concrete coordinators:
  UnitaryKVCacheCoordinator  — single KV cache group (the only case
                               my-vllm supports).
  HybridKVCacheCoordinator   — multiple groups (e.g. full-attn +
                               Mamba for hybrid models).
  KVCacheCoordinatorNoPrefixCache — disables prefix caching.

C134: minimal subset — `UnitaryKVCacheCoordinator` only. Hybrid
(no-Mamba / no-cross-attention / no-MLA) and NoPrefixCache are
out of scope per docs/alignment-plan.md.

The coordinator reuses the scheduler-facing flat `KVCacheBlocks` from
`kv_cache_manager.py` (matching vLLM, which also avoids defining a
second blocks type at the coordinator layer).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from my_vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheBlocks,
    SingleTypeKVCacheManager,
)

if TYPE_CHECKING:
    from my_vllm.v1.core.block_pool import BlockPool
    from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from my_vllm.v1.request import Request


class KVCacheCoordinator(ABC):
    """Coordinate the KV cache across all KV cache groups.

    Mirrors vLLM V1 v1/core/kv_cache_coordinator.py:60.
    """

    def __init__(
        self,
        block_pool: "BlockPool",
        block_size: int,
        enable_caching: bool,
    ):
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = enable_caching

    # ── Pure coordinator API (group-agnostic) ─────────────────────────

    @abstractmethod
    def get_num_blocks_to_allocate(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> int:
        """Total blocks needed across all groups for `num_new_tokens`."""

    @abstractmethod
    def allocate_new_blocks(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> KVCacheBlocks | None:
        """Allocate new blocks across all groups. None on failure."""

    @abstractmethod
    def get_computed_blocks(
        self, request: "Request"
    ) -> tuple[KVCacheBlocks, int]:
        """Find consecutive prefix cache hits. Returns (blocks, num_computed)."""

    @abstractmethod
    def free(self, request_id: str) -> None:
        """Release all blocks this coordinator owns for `request_id`."""

    @abstractmethod
    def pop_blocks_for_free(
        self, request_id: str
    ) -> list | None:
        """Remove blocks without freeing (deferred)."""

    @abstractmethod
    def release_deferred_blocks(self, blocks) -> None:
        """Release previously deferred blocks back to the pool."""

    @abstractmethod
    def get_block_ids(self, request_id: str) -> list[list[int]]:
        """Per-group block IDs for `request_id`."""

    @abstractmethod
    def take_kv_cache_block_copies(self):
        """Drain pending CoW copies for the Worker."""

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free


class UnitaryKVCacheCoordinator(KVCacheCoordinator):
    """Single-group coordinator — the only variant my-vllm supports.

    Mirrors vLLM V1 v1/core/kv_cache_coordinator.py:435.
    Wraps a single `SingleTypeKVCacheManager` (typically
    `FullAttentionManager` for OPT).
    """

    def __init__(
        self,
        block_pool: "BlockPool",
        block_size: int,
        enable_caching: bool,
        manager: SingleTypeKVCacheManager | None = None,
    ):
        super().__init__(block_pool, block_size, enable_caching)
        # Default to FullAttentionManager — the only concrete subclass
        # my-vllm currently uses. Tests can pass a different manager.
        self.manager: SingleTypeKVCacheManager = (
            manager
            if manager is not None
            else FullAttentionManager(
                block_pool=block_pool,
                block_size=block_size,
                enable_caching=enable_caching,
                kv_cache_group_id=0,
            )
        )

    def get_num_blocks_to_allocate(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> int:
        return self.manager.get_num_blocks_to_allocate(request, num_new_tokens)

    def allocate_new_blocks(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> "KVCacheBlocks | None":
        group_blocks = self.manager.allocate_new_blocks(request, num_new_tokens)
        if group_blocks is None:
            return None
        # Convert per-group SingleTypeKVCacheBlocks into the flat
        # scheduler-facing KVCacheBlocks. For Unitary (single group),
        # this is a straight flatten.
        from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: PLC0415
        return KVCacheBlocks(block_ids=list(group_blocks.block_ids))

    def get_computed_blocks(
        self, request: "Request"
    ) -> "tuple[KVCacheBlocks, int]":
        from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks  # noqa: PLC0415
        if not self.enable_caching:
            return KVCacheBlocks.empty(), 0
        group_blocks = self.manager.get_cached_blocks(request)
        if group_blocks is None:
            return KVCacheBlocks.empty(), 0
        num_computed = self.manager.get_num_cached_tokens(request)
        return KVCacheBlocks(block_ids=list(group_blocks.block_ids)), num_computed

    def free(self, request_id: str) -> None:
        self.manager.free(request_id)

    def pop_blocks_for_free(self, request_id: str):
        return self.manager.pop_blocks_for_free(request_id)

    def release_deferred_blocks(self, blocks) -> None:
        if blocks:
            self.manager.release_deferred_blocks(blocks)

    def get_block_ids(self, request_id: str) -> list[list[int]]:
        return [self.manager.get_block_ids(request_id)]

    def take_kv_cache_block_copies(self):
        return self.manager.take_kv_cache_block_copies()

    def cache_blocks(
        self,
        request: "Request",
        num_tokens: int,
    ) -> None:
        """Promote `request`'s blocks into the local prefix-cache map.

        Mirrors vLLM V1 v1/core/kv_cache_coordinator.py:
        UnitaryKVCacheCoordinator.cache_blocks (forwarded to
        the underlying SingleTypeKVCacheManager.cache_blocks).

        C143: distributed KV cache ↔ block-hash integration.
        Called after a successful prefill or after the
        `KVConnectorBase_V1` reports a prefix-cache hit (via
        `get_num_new_matched_tokens`).
        """
        self.manager.cache_blocks(request, num_tokens)
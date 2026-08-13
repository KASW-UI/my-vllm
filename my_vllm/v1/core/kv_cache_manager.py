"""KVCacheManager — KV cache lifecycle manager (scheduler-facing facade).

Mirrors vLLM V1 v1/core/kv_cache_manager.py.

C018: allocate_slots() replaces allocate(). Manages per-request block
tables and provides block ID lookups. No longer creates KVCache objects
(HF past_key_values) — the GPUModelRunner handles actual KV storage.

C036: CoW protection on shared blocks — clones shared blocks (ref_cnt>1)
and records KVCacheBlockCopy for the Worker to execute.

C134: re-exposes the same per-request block-ID list as before
(`get_block_ids` returns a flat list of ints for the single-group
case) but delegates to the new `UnitaryKVCacheCoordinator` →
`FullAttentionManager` hierarchy:
    KVCacheManager (this file, scheduler-facing facade)
        → UnitaryKVCacheCoordinator (kv_cache_coordinator.py)
        → FullAttentionManager (single_type_kv_cache_manager.py)
        → BlockPool (block_pool.py)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_coordinator import (
    UnitaryKVCacheCoordinator,
)
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from my_vllm.v1.request import Request

if TYPE_CHECKING:
    from my_vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector


@dataclass
class KVCacheBlocks:
    """Allocation result bridging Scheduler ↔ KVCacheManager.

    Mirrors vLLM V1 kv_cache_manager.py:33.
    Flat list of block IDs (single-group case). Hides internal block
    representation from the Scheduler.
    """

    block_ids: list[int]

    def __len__(self) -> int:
        return len(self.block_ids)

    @classmethod
    def empty(cls) -> "KVCacheBlocks":
        return cls(block_ids=[])


class KVCacheManager:
    """KV cache lifecycle manager — scheduler-facing facade.

    C018: tracks per-request block ownership via request_blocks dict.
    C036: CoW for shared blocks — allocates new block, records copy.

    C134: delegates to UnitaryKVCacheCoordinator. The public API
    (allocate_slots, get_computed_blocks, free, get_block_ids) is
    preserved unchanged so existing callers (scheduler, EngineCore,
    output processor) need no modification.
    """

    def __init__(
        self,
        block_pool: BlockPool | None = None,
        block_size: int = 16,
        num_blocks: int = 512,
    ):
        self.block_pool = block_pool or BlockPool(num_blocks, block_size)
        self.block_size = self.block_pool.block_size
        self.coordinator = UnitaryKVCacheCoordinator(
            block_pool=self.block_pool,
            block_size=self.block_size,
            enable_caching=self.block_pool.enable_caching,
        )
        # Legacy per-request bookkeeping. Maintained for callers that
        # still read `request_blocks` directly.
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        self._pending_cow_copies: list[KVCacheBlockCopy] = []

    # ── Public API (legacy-compatible) ─────────────────────────────────

    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> KVCacheBlocks | None:
        """Allocate blocks for num_new_tokens. Returns flat block IDs or None."""
        group_blocks = self.coordinator.allocate_new_blocks(
            request, num_new_tokens
        )
        if group_blocks is None:
            return None
        # Mirror allocation into the legacy dict so callers reading
        # `request_blocks` see the same data.
        flat_ids = group_blocks.block_ids
        self.request_blocks[request.request_id] = [
            self.block_pool.blocks[i] for i in flat_ids
        ]
        return KVCacheBlocks(block_ids=list(flat_ids))

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int]:
        """Find consecutive prefix cache hits for a request.

        C029: walks request.block_hashes from the start. C036: CoW —
        bumps ref_cnt on reuse, clones shared blocks (ref_cnt>1) to
        prevent cross-request KV cache pollution, and records KVCacheBlockCopy.
        """
        if not self.block_pool.enable_caching:
            return KVCacheBlocks.empty(), 0

        cached_blocks, num_computed = self.coordinator.get_computed_blocks(request)
        if cached_blocks.block_ids:
            flat_ids = cached_blocks.block_ids
            self.request_blocks[request.request_id] = [
                self.block_pool.blocks[i] for i in flat_ids
            ]
            return KVCacheBlocks(block_ids=list(flat_ids)), num_computed
        return KVCacheBlocks.empty(), 0

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending CoW copies for the Worker to execute.

        Returns copies and the list of blocks to retain.
        """
        return self.coordinator.take_kv_cache_block_copies()

    def get_block_ids(self, request_id: str) -> list[int]:
        # Flat: single group, so collapse the 2-level structure.
        return self.coordinator.get_block_ids(request_id)[0]

    def cache_blocks(
        self,
        request: Request,
        num_tokens: int,
    ) -> None:
        """Promote `request`'s blocks into the local prefix-cache map.

        Mirrors vLLM V1 kv_cache_manager.py:KVCacheManager.cache_blocks.

        C143: distributed KV cache ↔ block-hash integration.
        Called after a successful prefill or after a
        connector-reported prefix-cache hit (via
        `get_num_new_matched_tokens`). Delegates to the
        coordinator → SingleTypeKVCacheManager.
        """
        self.coordinator.cache_blocks(request, num_tokens)

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.request_blocks.get(request_id, [])

    def free(self, request_id: str):
        self.request_blocks.pop(request_id, None)
        self.coordinator.free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock] | None:
        """Remove blocks from tracking without freeing them.

        Used by defer_block_free: the caller takes ownership of the blocks
        and decides when to release them back to the pool.
        """
        self.request_blocks.pop(request_id, None)
        return self.coordinator.pop_blocks_for_free(request_id)

    def release_deferred_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Release previously deferred blocks back to the pool."""
        if blocks:
            self.coordinator.release_deferred_blocks(blocks)

    @property
    def num_free_blocks(self) -> int:
        return self.coordinator.num_free_blocks

    # ── C036 legacy proxy (CoW unit tests) ─────────────────────────────

    def _apply_cow(self, block: KVCacheBlock) -> KVCacheBlock:
        """Apply Copy-on-Write on a single block.

        Proxy to the underlying `FullAttentionManager._apply_cow`. Kept
        for legacy C036 unit tests that exercise the CoW path directly
        on the manager; new code should call `get_computed_blocks` which
        invokes CoW automatically.
        """
        return self.coordinator.manager._apply_cow(block)

    # --- backward-compat methods (C014 tests) ---

    def allocate(self, request: Request) -> bool:
        blocks = self.allocate_slots(request, len(request.prompt_token_ids))
        return blocks is not None

    def get_cache(self, request_id: str):
        blocks = self.request_blocks.get(request_id)
        if blocks:
            return [b.block_id for b in blocks]
        return None
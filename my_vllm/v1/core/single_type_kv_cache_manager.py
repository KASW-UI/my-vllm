"""SingleTypeKVCacheManager — per-KV-cache-type manager.

Mirrors vLLM V1 v1/core/single_type_kv_cache_manager.py:36.

vLLM provides one subclass per KV cache spec type:
  FullAttentionManager, SlidingWindowManager, ChunkedLocalAttentionManager,
  MambaManager, CrossAttentionManager, RSWAManager, SinkFullAttentionManager,
  ...

C134: minimal subset — `FullAttentionManager` only. my-vllm supports
a single dense-attention OPT model; sliding-window / chunked-local /
Mamba / cross-attention are out of scope per docs/alignment-plan.md
(they exist in vLLM for non-Dense architectures, all of which are
Delete-tier).

The abstract `SingleTypeKVCacheManager` is provided so future expansion
to other attention types requires only subclassing, not refactoring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy

if TYPE_CHECKING:
    from my_vllm.v1.core.block_pool import BlockPool
    from my_vllm.v1.request import Request


@dataclass
class SingleTypeKVCacheBlocks:
    """Allocation result bridging Coordinator ↔ SingleTypeKVCacheManager.

    Mirrors vLLM V1 single_type_kv_cache_manager.py:SingleTypeKVCacheBlocks.
    Hides internal block representation from the Coordinator.
    """

    block_ids: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.block_ids)

    @classmethod
    def empty(cls) -> "SingleTypeKVCacheBlocks":
        return cls(block_ids=[])


class SingleTypeKVCacheManager(ABC):
    """Abstract base — handles one KV cache group of one spec type.

    Mirrors vLLM V1 v1/core/single_type_kv_cache_manager.py:36.
    """

    supports_fine_grained_hash_lookup: bool = False

    def __init__(
        self,
        block_pool: "BlockPool",
        block_size: int,
        enable_caching: bool,
        kv_cache_group_id: int = 0,
    ):
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = enable_caching
        self.kv_cache_group_id = kv_cache_group_id

        # request_id → list of blocks owned by this group.
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = defaultdict(list)

        # # blocks cached for each request (prefix-cache hit count).
        self.num_cached_blocks: dict[str, int] = {}

        # Pending CoW copies this group needs to publish to the Worker.
        self._pending_cow_copies: list[KVCacheBlockCopy] = []

    # ── Public API ─────────────────────────────────────────────────────

    @abstractmethod
    def get_num_blocks_to_allocate(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> int:
        """How many blocks this group needs to allocate for `num_new_tokens`."""

    def allocate_new_blocks(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> SingleTypeKVCacheBlocks | None:
        """Allocate new blocks for `num_new_tokens` tokens.

        Returns `SingleTypeKVCacheBlocks` describing the new + existing
        block IDs, or `None` if the pool cannot satisfy the allocation.
        """
        n_blocks_needed = self.get_num_blocks_to_allocate(request, num_new_tokens)
        existing = self.req_to_blocks[request.request_id]
        new_blocks_needed = max(0, n_blocks_needed - len(existing))
        if new_blocks_needed == 0:
            return SingleTypeKVCacheBlocks(
                block_ids=[b.block_id for b in existing]
            )

        new_blocks = self.block_pool.get_new_blocks(new_blocks_needed)
        if new_blocks is None:
            return None
        all_blocks = existing + new_blocks
        self.req_to_blocks[request.request_id] = all_blocks
        return SingleTypeKVCacheBlocks(
            block_ids=[b.block_id for b in all_blocks]
        )

    def get_num_cached_tokens(self, request: "Request") -> int:
        """Returns the number of tokens that have cached prefix blocks."""
        return self.num_cached_blocks.get(request.request_id, 0)

    def get_cached_blocks(
        self, request: "Request"
    ) -> SingleTypeKVCacheBlocks | None:
        """Walk request.block_hashes looking for consecutive prefix hits.

        Returns a `SingleTypeKVCacheBlocks` (with the cached blocks) and
        updates `num_cached_blocks` for the request. Returns `None` if
        caching is disabled.
        """
        if not self.enable_caching:
            return None

        block_size = self.block_size
        max_cache_hit_length = max(0, request.num_tokens - 1)
        request.compute_block_hashes(block_size)
        cached_blocks, num_computed = self._walk_cached_prefix(
            request, max_cache_hit_length, block_size,
        )

        if cached_blocks:
            self.req_to_blocks[request.request_id] = cached_blocks
            self.num_cached_blocks[request.request_id] = num_computed
            return SingleTypeKVCacheBlocks(
                block_ids=[b.block_id for b in cached_blocks]
            )
        return None

    def _walk_cached_prefix(
        self,
        request: "Request",
        max_cache_hit_length: int,
        block_size: int,
    ) -> tuple[list[KVCacheBlock], int]:
        """Walk request.block_hashes, returning cached (and CoW'd) blocks."""
        cached_blocks: list[KVCacheBlock] = []
        num_computed = 0
        for block_hash in request.block_hashes:
            if not block_hash:
                break
            if num_computed >= max_cache_hit_length:
                break
            cached = self.block_pool.get_cached_block(block_hash)
            if cached is None:
                break
            cached.ref_cnt += 1
            private = self._apply_cow(cached)
            cached_blocks.append(private)
            num_computed += min(
                block_size,
                request.num_tokens - num_computed,
            )
        return cached_blocks, num_computed

    def free(self, request_id: str) -> None:
        """Release the blocks this group owns for `request_id`."""
        if request_id in self.req_to_blocks:
            blocks = self.req_to_blocks.pop(request_id)
            self.block_pool.free_blocks(blocks)
        self.num_cached_blocks.pop(request_id, None)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock] | None:
        """Remove blocks from tracking without freeing them (deferred)."""
        blocks = self.req_to_blocks.pop(request_id, None)
        self.num_cached_blocks.pop(request_id, None)
        return blocks

    def release_deferred_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Release previously deferred blocks back to the pool."""
        if blocks:
            self.block_pool.free_blocks(blocks)

    def get_block_ids(self, request_id: str) -> list[int]:
        return [b.block_id for b in self.req_to_blocks.get(request_id, [])]

    def _apply_cow(self, block: KVCacheBlock) -> KVCacheBlock:
        """Copy-on-write: clone shared block so writes don't pollute.

        Aligned with vLLM SingleTypeKVCacheManager._apply_cow.
        """
        if block.ref_cnt <= 1:
            return block
        new_blocks = self.block_pool.get_new_blocks(1)
        if new_blocks is None:
            return block
        cow_block = new_blocks[0]
        self._pending_cow_copies.append(
            KVCacheBlockCopy(
                src_block_id=block.block_id,
                dst_block_id=cow_block.block_id,
            )
        )
        block.ref_cnt -= 1
        return cow_block

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending CoW copies for the Worker to execute.

        Returns (copies, blocks_to_retain). The Worker must keep
        `blocks_to_retain` alive while the copy is in flight.
        """
        copies = self._pending_cow_copies
        self._pending_cow_copies = []
        retained: list[KVCacheBlock] = []
        for c in copies:
            retained.append(self.block_pool.blocks[c.src_block_id])
            retained.append(self.block_pool.blocks[c.dst_block_id])
        return copies, retained


class FullAttentionManager(SingleTypeKVCacheManager):
    """SingleTypeKVCacheManager for full-attention layers (OPT-style).

    C134: the only concrete subclass. All OPT layers are full-attention
    with the same block size, so `get_num_blocks_to_allocate` is a
    straight ceiling division.
    """

    def get_num_blocks_to_allocate(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> int:
        """Ceiling-division based on `block_size`.

        Returns at least 1 — a request always needs at least one block
        even if `num_new_tokens` is small (matches vLLM
        FullAttentionManager.get_num_blocks_to_allocate).
        """
        return max(1, (num_new_tokens + self.block_size - 1) // self.block_size)
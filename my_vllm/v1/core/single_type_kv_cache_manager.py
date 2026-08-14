"""SingleTypeKVCacheManager — per-KV-cache-type manager.

Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py (subset).

vLLM provides one subclass per KV cache spec type:
  FullAttentionManager, SlidingWindowManager, ChunkedLocalAttentionManager,
  MambaManager, CrossAttentionManager, RSWAManager, SinkFullAttentionManager,
  ...

C134: minimal subset — `FullAttentionManager` only. my-vllm supports
a single dense-attention OPT model; sliding-window / chunked-local /
Mamba / cross-attention are out of scope per docs/alignment-plan.md
(they exist in vLLM for non-Dense architectures, all of which are
Delete-tier).

C254: extends the abstract base + `FullAttentionManager` with the full
vLLM V1 surface so the new `KVCacheCoordinator` (multi-group, hybrid)
can dispatch through the registry:
  - `find_longest_cache_hit` (classmethod)
  - `add_local_computed_blocks` (two-phase prefix-cache integration)
  - `allocate_external_computed_blocks` (KV-connector source blocks)
  - `allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)`
  - `get_num_blocks_to_allocate(request_id, num_tokens, ...)`
  - `get_num_common_prefix_blocks` (running-prefix detection)
  - `remove_skipped_blocks` (sliding-window / Mamba gap freeing)
  - `new_step_starts` (manager step-boundary hook)
  - `get_blocks(request_id)`
  - `cache_blocks` (with `retention_interval` knob for sparse retention)

Also adds:
  - `get_manager_for_kv_cache_spec(...)` dispatcher (registry lookup)
  - `register_all_kvcache_specs()` (one-shot default registration)

Backward compatibility:
  - The legacy C134 2-arg `get_num_blocks_to_allocate_simple(request,
    num_new_tokens)` and `allocate_new_blocks_simple(request,
    num_new_tokens)` are preserved so the existing
    `UnitaryKVCacheCoordinator` / `KVCacheManager` keep working
    unchanged. The legacy `KVCacheManager` test path (C018/C134/C143)
    is the only path that exercises these; the new multi-spec
    coordinator uses the vLLM-signature methods instead.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import ceil
from typing import TYPE_CHECKING, ClassVar

from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from my_vllm.v1.kv_cache_interface import FullAttentionSpec

if TYPE_CHECKING:
    from my_vllm.v1.core.block_pool import BlockPool
    from my_vllm.v1.kv_cache_interface import KVCacheSpec
    from my_vllm.v1.request import Request


# ---------------------------------------------------------------------------
# Block container
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SingleTypeKVCacheManager(ABC):
    """Abstract base — handles one KV cache group of one spec type.

    Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:36.

    Carries the full vLLM V1 surface; the legacy C134 2-arg methods
    live in the two ``_simple`` wrappers so the existing
    `UnitaryKVCacheCoordinator` keeps working.
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def __init__(
        self,
        block_pool: "BlockPool",
        block_size: int,
        enable_caching: bool,
        kv_cache_group_id: int = 0,
        kv_cache_spec: "KVCacheSpec | None" = None,
        scheduler_block_size: int | None = None,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ):
        self.block_pool = block_pool
        # Effective block size (may be multiplied by DCP for sharded specs).
        self.block_size = block_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        self.enable_caching = enable_caching
        self.kv_cache_group_id = kv_cache_group_id
        self.kv_cache_spec = kv_cache_spec
        # Scheduling granularity. Defaults to block_size for the single-
        # group Dense path; multi-group coordinators pass the LCM of all
        # group block sizes (per vLLM).
        self.scheduler_block_size = (
            scheduler_block_size
            if scheduler_block_size is not None
            else self.block_size
        )
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        self._max_admission_blocks_per_request = max_admission_blocks_per_request
        # C054/C149: mamba layers + mixed-precision caches need freshly
        # allocated blocks zeroed (state is read before written; precision
        # reinterpretation of stale bytes → NaN/Inf). my-vllm doesn't
        # zero today (no Mamba + no mixed-precision in scope) but the
        # bookkeeping slot is preserved so future Mamba wiring picks it up.
        self._record_new_block_ids = needs_kv_cache_zeroing and bool(kv_cache_spec)
        self.new_block_ids: list[int] = []

        # Per-request block ownership for this group.
        self.req_to_blocks: dict[str, list[KVCacheBlock]] = defaultdict(list)
        # Per-request prefix-cache hit count (full blocks).
        self.num_cached_block: dict[str, int] = {}
        # Null block marker (don't free — ref_cnt not maintained).
        self._null_block = block_pool.null_block
        # EAGLE/MTP lookahead drop — set by HybridKVCacheCoordinator after
        # grouping. Defaults False.
        self.use_eagle: bool = False
        # Partial-hit CoW bookkeeping (populated by `add_local_computed_blocks`
        # when a prefix hit ends inside a cache block).
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []

    # ── Legacy C134 API (kept for UnitaryKVCacheCoordinator) ──────────────

    def get_num_blocks_to_allocate_simple(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> int:
        """C134: simple ceiling-division based on `block_size`.

        Returns at least 1 — a request always needs at least one block
        even if `num_new_tokens` is small (matches vLLM
        FullAttentionManager.get_num_blocks_to_allocate legacy path).
        """
        return max(1, (num_new_tokens + self.block_size - 1) // self.block_size)

    def allocate_new_blocks_simple(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> SingleTypeKVCacheBlocks | None:
        """C134: legacy 2-arg alloc returning `SingleTypeKVCacheBlocks`.

        Returns `SingleTypeKVCacheBlocks` describing the new + existing
        block IDs, or `None` if the pool cannot satisfy the allocation.
        """
        n_blocks_needed = self.get_num_blocks_to_allocate_simple(
            request, num_new_tokens
        )
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
        """Returns the number of tokens that have cached prefix blocks.

        Mirrors vLLM V1 legacy path (C134-compatible).
        """
        n_blocks = self.num_cached_block.get(request.request_id, 0)
        return n_blocks * self.block_size

    def get_cached_blocks(
        self, request: "Request"
    ) -> SingleTypeKVCacheBlocks | None:
        """Walk request.block_hashes looking for consecutive prefix hits.

        Returns a `SingleTypeKVCacheBlocks` (with the cached blocks) and
        updates `num_cached_block` for the request. Returns `None` if
        caching is disabled.
        """
        if not self.enable_caching:
            return None

        max_cache_hit_length = max(0, request.num_tokens - 1)
        request.compute_block_hashes(self.block_size)
        cached_blocks, num_computed = self._walk_cached_prefix(
            request, max_cache_hit_length, self.block_size,
        )

        if cached_blocks:
            self.req_to_blocks[request.request_id] = cached_blocks
            self.num_cached_block[request.request_id] = num_computed // self.block_size
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
        """Walk request.block_hashes, returning cached (and CoW'd) blocks.

        C143/C036: bumps ref_cnt on reuse; clones shared blocks
        (ref_cnt>1) via `_apply_cow` to prevent cross-request KV cache
        pollution, recording the CoW for the Worker to execute.
        """
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
            private = self._apply_cow_simple(cached)
            cached_blocks.append(private)
            num_computed += min(
                block_size,
                request.num_tokens - num_computed,
            )
        return cached_blocks, num_computed

    def _apply_cow_simple(self, block: KVCacheBlock) -> KVCacheBlock:
        """C036/C134: copy-on-write clone shared block so writes don't pollute.

        Aligned with vLLM SingleTypeKVCacheManager._apply_cow.
        """
        if block.ref_cnt <= 1:
            return block
        new_blocks = self.block_pool.get_new_blocks(1)
        if new_blocks is None:
            return block
        cow_block = new_blocks[0]
        self._pending_cow_copies.append(
            (block, cow_block)
        )
        self._pending_cow_block_copies_simple.append(
            KVCacheBlockCopy(
                src_block_id=block.block_id,
                dst_block_id=cow_block.block_id,
            )
        )
        block.ref_cnt -= 1
        return cow_block

    def free(self, request_id: str) -> None:
        """Release the blocks this group owns for `request_id`."""
        if request_id in self.req_to_blocks:
            blocks = self.req_to_blocks.pop(request_id)
            self.block_pool.free_blocks(blocks)
        self.num_cached_block.pop(request_id, None)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock] | None:
        """Remove blocks from tracking without freeing them (deferred)."""
        blocks = self.req_to_blocks.pop(request_id, None)
        self.num_cached_block.pop(request_id, None)
        return blocks

    def release_deferred_blocks(self, blocks: list[KVCacheBlock]) -> None:
        """Release previously deferred blocks back to the pool."""
        if blocks:
            self.block_pool.free_blocks(blocks)

    def get_block_ids(self, request_id: str) -> list[int]:
        return [b.block_id for b in self.req_to_blocks.get(request_id, [])]

    # Old single-name alias for legacy `_pending_cow_copies` C036 path —
    # tests (test_C036, test_C212) consume this. The new path uses
    # `_pending_cow_copies` for the vLLM two-tuple form.
    @property
    def _pending_cow_block_copies_simple(self) -> list[KVCacheBlockCopy]:
        """Lazy list of CoW copies in the legacy (NamedTuple) form.

        The legacy `_apply_cow_simple` appends here; the
        `take_kv_cache_block_copies` API consumes this.
        """
        # Backed by the same storage as the vLLM-format list for tests
        # that read either form.
        if not hasattr(self, "_legacy_cow_copies"):
            self._legacy_cow_copies: list[KVCacheBlockCopy] = []
        return self._legacy_cow_copies

    def cache_blocks_simple(
        self,
        request: "Request",
        num_tokens: int,
    ) -> None:
        """C143: register `request`'s blocks into the local prefix-cache map.

        Calls through to the vLLM-signature `cache_blocks` with
        ``retention_interval=None`` (dense mode).
        """
        self.cache_blocks(request, num_tokens, retention_interval=None)

    # ── Distributed KV cache ↔ prefix-cache integration ─────────────

    def cache_blocks(
        self,
        request: "Request",
        num_tokens: int,
        retention_interval: int | None = None,
    ) -> None:
        """Register `request`'s blocks into the local prefix-cache map.

        Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
        SingleTypeKVCacheManager.cache_blocks (L403).

        Called after a successful prefill (or after a connector-
        reported prefix-cache hit) so that future requests with
        the same hash prefix find the blocks in the local cache
        via `block_pool.get_cached_block(hash)`.

        C254: extends the C143 logic with `retention_interval` knob.
        Sparse-retention managers (SWA) override the per-block mask
        via `reachable_block_mask`; full-attention caches every
        block (the base policy).
        """
        num_cached = self.num_cached_block.get(request.request_id, 0)
        num_full = num_tokens // self.block_size
        if num_cached >= num_full:
            return
        if request.request_id in self.req_to_blocks:
            blocks = self.req_to_blocks[request.request_id][
                num_cached : min(num_full, len(self.req_to_blocks[request.request_id]))
            ]
            if blocks:
                # Block-pool's existing cache_full_blocks helper computes
                # the per-block hashes + registers them in the cache map.
                # For multi-group: pass `kv_cache_group_id` so the per-
                # group registry entries don't collide.
                self.block_pool.cache_full_blocks(request, blocks)
        # Promote cached count even if `cache_full_blocks` no-op'd.
        self.num_cached_block[request.request_id] = num_full

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending CoW copies for the Worker to execute (legacy path).

        Returns (copies, blocks_to_retain). The Worker must keep
        `blocks_to_retain` alive while the copy is in flight.
        """
        copies = self._pending_cow_block_copies_simple
        # Wipe and re-create so the lazy property returns a fresh list.
        object.__setattr__(self, "_legacy_cow_copies", [])
        retained: list[KVCacheBlock] = []
        for c in copies:
            retained.append(self.block_pool.blocks[c.src_block_id])
            retained.append(self.block_pool.blocks[c.dst_block_id])
        return copies, retained

    # ── Full vLLM surface (C254) ─────────────────────────────────────

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """How many blocks this group needs to allocate for `num_tokens`.

        Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
        get_num_blocks_to_allocate (L135).

        The new (vLLM) signature takes request_id + computed-blocks
        sequences so the per-group math can account for sliding-window
        skip / partial-hit CoW / admission caps. The legacy C134
        `get_num_blocks_to_allocate_simple` is kept for the single-
        group path.

        Base implementation: ceiling division (full-attention semantics).
        Subclasses with sparse hit semantics override.
        """
        del (
            request_id,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap,
        )
        return ceil(num_tokens / self.block_size)

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Touch the locally-cached (prefix-hit) blocks for `request_id`.

        Mirrors vLLM V1 single_type_kv_cache_manager.py:add_local_computed_blocks.

        Two-phase protocol (issue #33775): first every group touches its
        local cache-hit blocks, then any group allocates external (KV-
        connector-sourced) blocks. This guarantees an earlier group's
        external `get_new_blocks` cannot evict a later group's not-yet-
        touched cache-hit blocks.

        Base implementation: touch each non-skipped block, extend
        `req_to_blocks` with `[null] * skipped + new_computed_blocks`.
        """
        # Coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no
        # blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0, (
            "add_local_computed_blocks called for already-allocated request"
        )
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        # Full attention: no skipped tokens. Sliding-window subclass
        # would override `get_num_skipped_tokens` and trim.
        num_skipped_blocks = self.get_num_skipped_tokens(
            num_total_computed_tokens
        ) // self.block_size
        if num_skipped_blocks > 0:
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(list(new_computed_blocks))
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )
        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        req_blocks.extend(new_computed_blocks)
        # All cached hits (including skipped nulls) are already cached;
        # mark them so `cache_blocks` will not try to re-cache blocks
        # that already have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)
        # Partial-tail handling: cache_hit ends inside one of this
        # manager's blocks → record CoW redirect.
        if self._has_partial_local_hit(
            list(new_computed_blocks), num_local_computed_tokens
        ):
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (
                block_idx, new_computed_blocks[-1]
            )
            # Cap at the full blocks so `cache_blocks` re-caches the
            # private copy once full.
            self.num_cached_block[request_id] = block_idx

    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        """True iff the local hit ends inside one of this group's blocks."""
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Allocate new blocks for external (KV-connector) computed tokens.

        Mirrors vLLM V1 single_type_kv_cache_manager.py:
        allocate_external_computed_blocks. Must run after every group's
        local cache-hit blocks have been touched (issue #33775).
        """
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        allocated_blocks = self.block_pool.get_new_blocks(
            ceil(num_total_computed_tokens / self.block_size) - len(req_blocks)
        )
        if allocated_blocks is None:
            # Pool is exhausted; the coordinator's allocate_new_blocks will
            # surface the failure on its next call.
            return
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
    ) -> list[KVCacheBlock]:
        """Allocate new blocks for `num_tokens` (vLLM signature).

        Mirrors vLLM V1 single_type_kv_cache_manager.py:allocate_new_blocks.

        Returns a list of newly allocated blocks (cow_blocks first, then
        the gap-closing new blocks). Coordinator extends its tuple with
        this list.
        """
        del num_tokens_main_model  # Used by subclasses (EAGLE).
        cow_blocks: list[KVCacheBlock] = []
        # Partial hit: redirect the shared tail to a private CoW block.
        if request_id in self._partial_hit_reqs:
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            if self._record_new_block_ids:
                self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = ceil(num_tokens / self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return cow_blocks
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in new_blocks)
        return cow_blocks + new_blocks

    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        return self._record_new_block_ids

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def take_pending_cow_copies(
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain pending CoW source and destination block pairs."""
        pending = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending

    def _apply_cow(
        self,
        request_id: str,
        block_idx: int,
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.

        Mirrors vLLM V1 SingleTypeKVCacheManager._apply_cow.
        """
        req_blocks = self.req_to_blocks[request_id]
        assert block_idx < len(req_blocks)
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1

    @classmethod
    def reachable_block_mask(
        cls,
        start_block: int,
        end_block: int,
        alignment_tokens: int | None,
        kv_cache_spec: "KVCacheSpec",
        use_eagle: bool,
        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
        """Per-block mask for `cache_full_blocks`. None = cache every block.

        Subclasses with sparse hit semantics (SWA / Mamba) override to
        skip blocks that can never serve a hit. Full attention uses the
        default (None = cache everything).
        """
        del (
            start_block,
            end_block,
            alignment_tokens,
            kv_cache_spec,
            use_eagle,
            retention_interval,
            reachable_boundaries,
        )
        return None

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        """Return the per-group block list for `request_id`."""
        return list(self.req_to_blocks.get(request_id, []))

    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in ``[first_block, last_block)`` and replace with null."""
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove blocks that are no longer needed (sliding-window gap)."""
        del num_prompt_tokens
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # Full attention: nothing to free.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """How many tokens are skipped by this manager's attention window.

        Default (full attention): 0.
        """
        del num_computed_tokens
        return 0

    def new_step_starts(self) -> None:
        """Step-boundary hook (used by SWA / Mamba for per-step state)."""
        return None

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: "BlockPool",
        kv_cache_spec: "KVCacheSpec",
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Longest cache-hit prefix common to all groups.

        Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
        find_longest_cache_hit.

        Returns a tuple (per-group cached block lists, hit_length).
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_common_prefix_blocks(
        self, running_request_id: str
    ) -> int:
        """Number of prefix blocks shared with all other running requests."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete subclass: FullAttentionManager (the only one exercised)
# ---------------------------------------------------------------------------


class FullAttentionManager(SingleTypeKVCacheManager):
    """SingleTypeKVCacheManager for full-attention layers (OPT-style).

    C134: the only concrete subclass. All OPT layers are full-attention
    with the same block size.

    C254: extends with:
    - `find_longest_cache_hit` (full two-phase implementation, with
      eagle-block drop + alignment rounding).
    - `get_num_common_prefix_blocks` (ref-count based common-prefix
      detection for `cache_blocks` / scheduling decisions).
    """

    supports_fine_grained_hash_lookup: ClassVar[bool] = True

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: list,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: "BlockPool",
        kv_cache_spec: "KVCacheSpec",
        drop_eagle_block: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Full-attention longest-cache-hit.

        Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
        FullAttentionManager.find_longest_cache_hit (L658).

        Phase 1: walk `block_hashes` from the start, looking up each
        hash in the block pool. A miss breaks the chain.
        Phase 2 (EAGLE only): drop the last matched block so the engine
        recomputes the tokens right before the generation point.
        """
        del pcp_world_size
        block_size = kv_cache_spec.block_size
        if dcp_world_size > 1:
            block_size *= dcp_world_size

        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        # Phase 1: longest run of cached full blocks from the start.
        max_blocks = max_length // block_size
        for i in range(min(len(block_hashes), max_blocks)):
            block_hash = block_hashes[i]
            if not block_hash:
                break
            cached_block = block_pool.get_cached_block(block_hash)
            if cached_block is None:
                break
            for computed in computed_blocks:
                computed.append(cached_block)
        hit_length = len(computed_blocks[0]) * block_size

        # EAGLE/MTP drop: subtract one hash unit (alignment_tokens) when
        # fine-grained, else one cache block.
        if drop_eagle_block and hit_length > 0:
            hit_length -= min(alignment_tokens, block_size)
        # Round down to alignment; a no-op when alignment == block_size.
        hit_length -= hit_length % alignment_tokens
        num_blocks = ceil(hit_length / block_size) if hit_length > 0 else 0
        for computed in computed_blocks:
            del computed[num_blocks:]
        return computed_blocks, hit_length

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """Count prefix blocks shared with all other running requests.

        A block is "common" when its ref_cnt equals the number of
        running requests that own blocks in this group (i.e. every
        running request has it in its block table at the same index).

        Mirrors vLLM V1 FullAttentionManager.get_num_common_prefix_blocks.
        """
        blocks = self.req_to_blocks[running_request_id]
        if not blocks:
            return 0
        # Number of running requests = number of entries in req_to_blocks
        # (one block-list per running request).
        num_running = len(self.req_to_blocks)
        num_common = 0
        for block in blocks:
            if block.ref_cnt == num_running:
                num_common += 1
            else:
                break
        return num_common

    # `cache_blocks` is inherited from `SingleTypeKVCacheManager` (the
    # base default-impl caches every block, which is correct for full
    # attention). Partial-block / DCP-sharded caching would override
    # here when MLA / DCP land.

    # `get_num_blocks_to_allocate` is inherited — full attention's
    # ceiling-division is correct.


# ---------------------------------------------------------------------------
# Spec-class shells (C254 — declared for multi-spec routing but no body)
# ---------------------------------------------------------------------------
#
# These classes are not exercised in my-vllm today (Dense-only project).
# They're declared with the field names vLLM uses so the
# `KVCacheSpecRegistry` can route them when future Llama (SWA) /
# DeepSeek (MLA / Mamba) integration lands.
# ---------------------------------------------------------------------------


class SlidingWindowSpec:  # type: ignore[no-redef]
    """Spec shell for sliding-window attention (Llama Mistral etc.).

    C254 shell — mirrors vLLM SlidingWindowSpec field shape; body
    deliberately not vendored since sliding-window requires a
    non-Dense model (Llama-family) which is out of scope.

    C257: gains ``sliding_window`` field so SimpleCPUOffloadScheduler
    can compute the per-request offload target
    (``cdiv(sliding_window, block_size) + 1`` blocks per group).

    Use ``register_kv_cache_spec(SlidingWindowSpec, SlidingWindowManager,
    uniform_type_base_spec=SlidingWindowSpec)`` to plug into the
    registry.
    """

    def __init__(
        self,
        block_size: int = 16,
        sliding_window: int | None = None,
        **_kwargs,
    ) -> None:
        self.block_size = block_size
        self.sliding_window = sliding_window


class ChunkedLocalAttentionSpec:  # type: ignore[no-redef]
    """Spec shell for chunked-local attention (Llama-3.x fine-grained).

    C254 shell — declared for routing.
    """

    def __init__(self, block_size: int = 16, **_kwargs) -> None:
        self.block_size = block_size


class MambaSpec:  # type: ignore[no-redef]
    """Spec shell for Mamba (state-space) layers.

    C254 shell — declared for routing. Real Mamba layers require a
    non-Transformer model (Mamba / Jamba) which is Delete-tier per
    docs/alignment-plan.md.
    """

    def __init__(self, block_size: int = 16, **_kwargs) -> None:
        self.block_size = block_size


class CrossAttentionSpec:  # type: ignore[no-redef]
    """Spec shell for cross-attention (encoder-decoder) layers.

    C254 shell — declared for routing. Encoder-decoder models are
    Delete-tier.
    """

    def __init__(self, block_size: int = 16, **_kwargs) -> None:
        self.block_size = block_size


class MLAAttentionSpec(FullAttentionSpec):  # type: ignore[no-redef]
    """Spec shell for MLA (multi-head latent attention, DeepSeek-V3).

    C254 shell — declared for routing. MLA requires a non-Dense
    model; Phase 5 (C266) will vendor the real spec.

    Inherits from `FullAttentionSpec` so the registry can route MLA
    groups through `FullAttentionManager` until the real MLA manager
    lands in Phase 5. The `MLAAttentionSpec` class name is preserved
    so the `KVCacheSpecRegistry.get_manager_class` lookup returns
    `FullAttentionManager` (the placeholder).
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        # MLA shells forward through FullAttentionSpec so the math
        # works identically for now. Phase 5 will add MLA-specific
        # overrides (e.g. ``real_page_size_bytes`` for the
        # compressed kv_lora layout).
        super().__init__(*args, **kwargs)


class Shells:  # type: ignore[no-redef]
    """Container so callers can import all shells from one place.

    >>> from my_vllm.v1.core.single_type_kv_cache_manager import Shells
    >>> Shells.SlidingWindowSpec
    """


# Populate the shells container.
Shells.SlidingWindowSpec = SlidingWindowSpec
Shells.ChunkedLocalAttentionSpec = ChunkedLocalAttentionSpec
Shells.MambaSpec = MambaSpec
Shells.CrossAttentionSpec = CrossAttentionSpec
Shells.MLAAttentionSpec = MLAAttentionSpec


# ---------------------------------------------------------------------------
# Manager shells (C254 — declared for multi-spec routing)
# ---------------------------------------------------------------------------


class SlidingWindowManager(SingleTypeKVCacheManager):  # type: ignore[no-redef]
    """Shell — sliding-window manager. Real impl requires Llama (Phase 5)."""

    supports_fine_grained_hash_lookup: ClassVar[bool] = False

    def find_longest_cache_hit(cls, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "SlidingWindowManager.find_longest_cache_hit: "
            "shell — real impl lands with Llama-family wiring (Phase 5)."
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        raise NotImplementedError(
            "SlidingWindowManager.get_num_common_prefix_blocks: "
            "shell — real impl lands with Llama-family wiring (Phase 5)."
        )


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):  # type: ignore[no-redef]
    """Shell — chunked-local manager. Real impl requires Llama (Phase 5)."""

    def find_longest_cache_hit(cls, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "ChunkedLocalAttentionManager.find_longest_cache_hit: shell."
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        raise NotImplementedError(
            "ChunkedLocalAttentionManager.get_num_common_prefix_blocks: shell."
        )


class MambaManager(SingleTypeKVCacheManager):  # type: ignore[no-redef]
    """Shell — Mamba manager. Real impl requires Mamba model (Phase 5)."""

    def find_longest_cache_hit(cls, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "MambaManager.find_longest_cache_hit: shell."
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        raise NotImplementedError(
            "MambaManager.get_num_common_prefix_blocks: shell."
        )


class CrossAttentionManager(SingleTypeKVCacheManager):  # type: ignore[no-redef]
    """Shell — cross-attention manager. Real impl requires encoder-decoder."""

    def find_longest_cache_hit(cls, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError(
            "CrossAttentionManager.find_longest_cache_hit: shell."
        )

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        raise NotImplementedError(
            "CrossAttentionManager.get_num_common_prefix_blocks: shell."
        )


# ---------------------------------------------------------------------------
# Spec → manager dispatcher
# ---------------------------------------------------------------------------


def get_manager_for_kv_cache_spec(
    kv_cache_spec: "KVCacheSpec",
    block_pool: "BlockPool",
    enable_caching: bool,
    kv_cache_group_id: int = 0,
    scheduler_block_size: int | None = None,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    needs_kv_cache_zeroing: bool = False,
    max_admission_blocks_per_request: int | None = None,
) -> SingleTypeKVCacheManager:
    """Instantiate the right SingleTypeKVCacheManager for `kv_cache_spec`.

    Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
    get_manager_for_kv_cache_spec (L1776).

    Looks up the manager class via the registry (lazily bootstrapped).
    Raises if no manager is registered for the spec type.
    """
    from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec).__name__}"
    )
    return manager_class(
        block_pool=block_pool,
        block_size=kv_cache_spec.block_size,
        enable_caching=enable_caching,
        kv_cache_group_id=kv_cache_group_id,
        kv_cache_spec=kv_cache_spec,
        scheduler_block_size=scheduler_block_size,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        needs_kv_cache_zeroing=needs_kv_cache_zeroing,
        max_admission_blocks_per_request=max_admission_blocks_per_request,
    )


def register_all_kvcache_specs() -> None:
    """Built-in spec → manager registration.

    Mirrors vLLM V1 vllm/v1/core/single_type_kv_cache_manager.py:
    register_all_kvcache_specs (L1821).

    Registers ``FullAttentionSpec`` → ``FullAttentionManager`` (the only
    pair that has a real body in my-vllm today) plus the spec shells
    (SlidingWindow / ChunkedLocal / Mamba / CrossAttention / MLA) so
    the registry can route them when their real impls land in Phase 5.

    Safe to call multiple times — re-registration with the same
    metadata is a no-op; re-registration with different metadata is a
    hard error.
    """
    from my_vllm.v1.kv_cache_interface import FullAttentionSpec
    from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    # Default: dense attention → FullAttentionManager.
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    # Shells — register with shell manager classes so the registry
    # lookup succeeds; the shell managers raise NotImplementedError on
    # actual use (until Phase 5 lands the real bodies).
    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        MambaSpec,
        MambaManager,
        uniform_type_base_spec=MambaSpec,
    )
    KVCacheSpecRegistry.register(
        CrossAttentionSpec,
        CrossAttentionManager,
        uniform_type_base_spec=CrossAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        MLAAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
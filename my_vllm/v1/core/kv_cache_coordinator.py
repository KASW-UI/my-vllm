"""KVCacheCoordinator — top-level coordinator across KV cache groups.

Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py (subset).

Two parallel coordinator hierarchies live here:

1. **Legacy C134 path** — `UnitaryKVCacheCoordinator` (the simple
   wrapper at the bottom of this file). Used by `KVCacheManager` and
   the existing C134/C143/C212 tests. Wraps a single
   `SingleTypeKVCacheManager` directly with a flat
   `KVCacheBlocks`-in / `KVCacheBlocks`-out interface.

2. **C254 multi-spec path** — `KVCacheCoordinator` (ABC) +
   `KVCacheCoordinatorNoPrefixCache` + `UnitaryKVCacheCoordinator`
   (full vLLM-compatible variant) + `HybridKVCacheCoordinator` +
   `SpecGroup` + `get_kv_cache_coordinator()` factory. Operates on a
   `KVCacheConfig` (multi-group aware) and a `single_type_managers`
   tuple, dispatching to per-group managers via the
   `KVCacheSpecRegistry`.

C254 scope: build the multi-spec routing surface. Tests exercise it
directly with synthetic `KVCacheConfig` instances; the legacy
`UnitaryKVCacheCoordinator` (legacy) remains the production path
since OPT is Dense (single FullAttention group).

Wiring the new `KVCacheCoordinator` into `KVCacheManager` is deferred
to Phase 5 — that's when Llama (GQA) or DeepSeek (MLA) lands and the
single-group assumption breaks.

Coordinator hierarchy:

                          KVCacheCoordinator (ABC)
                                    │
            ┌───────────────────────┼────────────────────────────────┐
            ▼                       ▼                                ▼
  KVCacheCoordinatorNoPrefixCache  UnitaryKVCacheCoordinator       HybridKVCacheCoordinator
        (no caching)               (single group)                  (multi-group, fixed-point)
                                          │                                │
                                          ▼                                ▼
                              SingleTypeKVCacheManager ──registry──→ FullAttentionManager / MambaManager / …
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from typing import TYPE_CHECKING, NamedTuple

from my_vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SingleTypeKVCacheBlocks,
    SingleTypeKVCacheManager,
    get_manager_for_kv_cache_spec,
)

if TYPE_CHECKING:
    from my_vllm.v1.core.block_pool import BlockPool
    from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from my_vllm.v1.kv_cache_metrics import KVCacheMetricsCollector
    from my_vllm.v1.kv_cache_utils import BlockHash, KVCacheBlock
    from my_vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from my_vllm.v1.request import Request


# ===========================================================================
# C254 multi-spec path
# ===========================================================================


class SpecGroup(NamedTuple):
    """KV cache groups that share one spec, batched for cache-hit lookup.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:SpecGroup.

    Members sharing a spec are cached + looked up jointly, so the EAGLE
    last-block drop is decided for the whole spec group.
    """

    spec: "KVCacheSpec"
    group_ids: list[int]
    manager_cls: type[SingleTypeKVCacheManager]
    use_eagle: bool


class KVCacheCoordinator(ABC):
    """Coordinate the KV cache across all KV cache groups.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:KVCacheCoordinator.

    Operates on a `KVCacheConfig` and builds a `single_type_managers`
    tuple (one manager per group, dispatched via the
    `KVCacheSpecRegistry`). Subclasses override `find_longest_cache_hit`
    to encode the cross-group hit-reconciliation strategy:

    - ``UnitaryKVCacheCoordinator`` (single group): single lookup.
    - ``HybridKVCacheCoordinator`` (≥ 2 groups): fixed-point iteration
      with EAGLE-block verification.
    - ``KVCacheCoordinatorNoPrefixCache`` (caching off): returns empty.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        metrics_collector: "KVCacheMetricsCollector | None" = None,
        block_pool: "BlockPool | None" = None,
    ):
        self.kv_cache_config = kv_cache_config
        self.max_model_len = max_model_len
        self.enable_caching = enable_caching
        # Scheduling granularity — LCM of all group block sizes. For
        # single-group Dense (my-vllm today) this equals block_size.
        if scheduler_block_size is None:
            scheduler_block_size = (
                kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            )
        # hash_block_size defaults to scheduler_block_size for single-
        # group Dense. For multi-group it may differ — pass explicitly.
        if hash_block_size is None:
            hash_block_size = scheduler_block_size
        self.hash_block_size = hash_block_size
        self.scheduler_block_size = scheduler_block_size
        # Validate LCM constraints (per-group block_size divides it).
        assert scheduler_block_size % hash_block_size == 0, (
            "scheduler_block_size must be a multiple of hash_block_size"
        )
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        ), (
            "scheduler_block_size must be a multiple of every group's block_size"
        )

        # Block pool — use the injected one if given, else create a
        # default. Tests inject so they can wire in custom block sizes.
        if block_pool is None:
            from my_vllm.v1.core.block_pool import BlockPool
            block_pool = BlockPool(
                num_blocks=kv_cache_config.num_blocks,
                block_size=hash_block_size,
                enable_caching=enable_caching,
                metrics_collector=metrics_collector,
            )
        self.block_pool = block_pool

        # EAGLE/MTP group ids — used to drop the last block at every
        # cache-hit boundary (so the engine recomputes the lookahead
        # tail).
        self.eagle_group_ids: set[int] = {
            i
            for i, g in enumerate(kv_cache_config.kv_cache_groups)
            if g.is_eagle_group
        }
        if use_eagle and not self.eagle_group_ids:
            # Conservatively flag all groups when no group is flagged.
            self.eagle_group_ids = set(
                range(len(kv_cache_config.kv_cache_groups))
            )

        # One manager per group, dispatched via the registry.
        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=group.kv_cache_spec,
                block_pool=self.block_pool,
                enable_caching=enable_caching,
                kv_cache_group_id=i,
                scheduler_block_size=self.scheduler_block_size,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                needs_kv_cache_zeroing=(
                    kv_cache_config.needs_kv_cache_zeroing
                ),
            )
            for i, group in enumerate(kv_cache_config.kv_cache_groups)
        )
        # Propagate the EAGLE flag to each manager (Hybrid overrides
        # to set per-spec-group rather than per-group).
        for i, manager in enumerate(self.single_type_managers):
            manager.use_eagle = i in self.eagle_group_ids

    # ── Group-agnostic API ────────────────────────────────────────────

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple,
        num_encoder_tokens: int = 0,
        total_computed_tokens: int = 0,
        num_local_computed_tokens: int = 0,
        num_tokens_main_model: int | None = None,
        apply_admission_cap: bool = False,
    ) -> int:
        """Total blocks needed across all groups for `num_tokens`.

        Mirrors vLLM V1 KVCacheCoordinator.get_num_blocks_to_allocate.
        """
        if num_tokens_main_model is None:
            num_tokens_main_model = num_tokens
        n = 0
        for i, manager in enumerate(self.single_type_managers):
            # Cross-attention shell: a single static allocation of
            # blocks based on encoder length. We treat any manager that
            # is not a FullAttentionManager as encoder-coupled and
            # route accordingly.
            if type(manager).__name__ == "CrossAttentionManager":
                n += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    (),
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                n += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return n

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Two-phase cache-hit bookkeeping (issue #33775).

        Phase 1: every group touches its local cache-hit blocks.
        Phase 2: every group allocates external (KV-connector-sourced)
                 blocks.

        This guarantees an earlier group's external `get_new_blocks`
        cannot evict a later group's not-yet-touched cache-hit blocks.
        """
        # Running request short-circuit: a request already tracked in
        # `num_cached_block` won't have new prefix-cache hits.
        if any(
            request_id in mgr.num_cached_block
            for mgr in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens > 0:
            for manager in self.single_type_managers:
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple:
        """Allocate new blocks across all groups.

        Returns a tuple of per-group block lists.
        """
        return tuple(
            mgr.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if type(mgr).__name__ == "CrossAttentionManager"
                else num_tokens,
                num_tokens_main_model,
            )
            for mgr in self.single_type_managers
        )

    def cache_blocks(self, request: "Request", num_computed_tokens: int) -> None:
        """Promote `request`'s blocks into the local prefix-cache map."""
        for manager in self.single_type_managers:
            manager.cache_blocks(request, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """Free `request_id`'s blocks across all groups."""
        for manager in self.single_type_managers:
            manager.free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list["KVCacheBlock"]:
        """Pop all groups' bookkeeping, returning combined block list."""
        blocks: list["KVCacheBlock"] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id) or [])
        return blocks

    def get_num_common_prefix_blocks(
        self, running_request_id: str
    ) -> list[int]:
        """Number of common prefix blocks per group."""
        return [
            mgr.get_num_common_prefix_blocks(running_request_id)
            for mgr in self.single_type_managers
        ]

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove blocks no longer needed (sliding-window / Mamba gap)."""
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    def get_blocks(
        self, request_id: str
    ) -> tuple[list["KVCacheBlock"], ...]:
        """Return per-group block lists for `request_id`."""
        return tuple(
            list(mgr.req_to_blocks.get(request_id, []))
            for mgr in self.single_type_managers
        )

    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        for manager in self.single_type_managers:
            manager.new_step_starts()

    @abstractmethod
    def find_longest_cache_hit(
        self,
        block_hashes: list["BlockHash"],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list["KVCacheBlock"], ...], int, int]:
        """Find the longest prefix-cache hit common to all groups.

        Returns (per-group hit blocks, hit_length,
        num_uncached_common_prefix_tokens).
        """


# ---------------------------------------------------------------------------
# No-prefix-cache coordinator (caching disabled)
# ---------------------------------------------------------------------------


class KVCacheCoordinatorNoPrefixCache(KVCacheCoordinator):
    """Coordinator used when prefix caching is disabled.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:
    KVCacheCoordinatorNoPrefixCache (L385).

    Supports arbitrary numbers of KV cache groups (including 0 groups
    via an empty `kv_cache_groups` list — my-vllm would never hit
    that, but the constructor allows it for parity with vLLM's wider
    support).
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        metrics_collector: "KVCacheMetricsCollector | None" = None,
        block_pool: "BlockPool | None" = None,
    ):
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=False,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            block_pool=block_pool,
        )
        self.num_single_type_manager = len(self.single_type_managers)

    def get_num_common_prefix_blocks(
        self, running_request_id: str
    ) -> list[int]:
        return [0] * self.num_single_type_manager

    def find_longest_cache_hit(
        self,
        block_hashes: list["BlockHash"],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list["KVCacheBlock"], ...], int, int]:
        blocks: tuple[list["KVCacheBlock"], ...] = tuple(
            [] for _ in range(self.num_single_type_manager)
        )
        return blocks, 0, 0


# ---------------------------------------------------------------------------
# Unitary (single-group) coordinator — the only case my-vllm exercises
# ---------------------------------------------------------------------------


class UnitaryKVCacheCoordinatorFull(KVCacheCoordinator):
    """Coordinator for models with a single KV cache group.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:
    UnitaryKVCacheCoordinator (L435).

    Suffix "Full" distinguishes from the legacy C134
    ``UnitaryKVCacheCoordinator`` at the bottom of this module — that
    legacy class wraps a single manager directly with a flat
    `KVCacheBlocks` interface for `KVCacheManager`. The full vLLM-
    compatible variant here operates on `KVCacheConfig` and is what
    the `get_kv_cache_coordinator` factory produces.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        metrics_collector: "KVCacheMetricsCollector | None" = None,
        block_pool: "BlockPool | None" = None,
    ):
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            block_pool=block_pool,
        )
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "UnitaryKVCacheCoordinatorFull requires exactly 1 group; "
            f"got {len(kv_cache_config.kv_cache_groups)}"
        )
        self.kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = self.kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
        # Single group: hash block size must equal effective block size.
        # Use the resolved value from super() (which may have applied
        # the default).
        effective_hash_block_size = (
            hash_block_size if hash_block_size is not None
            else self.scheduler_block_size
        )
        assert not enable_caching or effective_hash_block_size == self.block_size, (
            "UnitaryKVCacheCoordinatorFull assumes hash_block_size == block_size; "
            f"got hash_block_size={effective_hash_block_size}, block_size={self.block_size}"
        )
        # EAGLE propagation (idempotent if already set in super).
        self.single_type_managers[0].use_eagle = 0 in self.eagle_group_ids

    def find_longest_cache_hit(
        self,
        block_hashes: list["BlockHash"],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list["KVCacheBlock"], ...], int, int]:
        hit_blocks, hit_length = self.single_type_managers[
            0
        ].find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_cache_hit_length,
            kv_cache_group_ids=[0],
            block_pool=self.block_pool,
            kv_cache_spec=self.kv_cache_spec,
            drop_eagle_block=0 in self.eagle_group_ids,
            alignment_tokens=self.block_size,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
        )
        # Single group: nothing "uncached common" — no other group to lag it.
        return hit_blocks, hit_length, 0


# ---------------------------------------------------------------------------
# Hybrid (multi-group) coordinator — fixed-point reconciliation
# ---------------------------------------------------------------------------


class HybridKVCacheCoordinator(KVCacheCoordinator):
    """Coordinator for hybrid models with multiple KV cache types.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:
    HybridKVCacheCoordinator (L521).

    Uses an iterative fixed-point algorithm: each attention type either
    accepts the current candidate length or reduces it. If any type
    reduces the length, restart checks over all types. Converges
    because length monotonically decreases and is bounded below by 0.
    """

    def __init__(
        self,
        kv_cache_config: "KVCacheConfig",
        max_model_len: int,
        max_in_flight_tokens: int,
        use_eagle: bool,
        enable_caching: bool,
        enable_kv_cache_events: bool,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        scheduler_block_size: int | None = None,
        hash_block_size: int | None = None,
        metrics_collector: "KVCacheMetricsCollector | None" = None,
        block_pool: "BlockPool | None" = None,
    ):
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            block_pool=block_pool,
        )
        assert pcp_world_size == 1, (
            "PCP not supported with hybrid attn now."
        )
        assert dcp_world_size == 1, (
            "DCP with hybrid KV layouts only supports full-attention / "
            "Mamba groups; my-vllm ships full-attention only."
        )
        # Build spec groups: groups sharing a spec merge into one
        # SpecGroup so they're cached + looked up jointly.
        self.verify_and_split_kv_cache_groups()

    def verify_and_split_kv_cache_groups(self) -> None:
        """Group KV cache groups by their spec type for efficient lookup."""
        self.attention_groups: list[SpecGroup] = []
        for i, group in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = type(self.single_type_managers[i])
            spec = group.kv_cache_spec
            use_eagle = i in self.eagle_group_ids

            # Try to find an existing group with the same spec.
            merged = False
            for idx, sg in enumerate(self.attention_groups):
                if sg.spec == spec:
                    assert manager_cls is sg.manager_cls, (
                        "Expected same manager class for identical KV cache specs."
                    )
                    sg.group_ids.append(i)
                    if use_eagle and not sg.use_eagle:
                        # Promote the SpecGroup to EAGLE.
                        self.attention_groups[idx] = sg._replace(use_eagle=True)
                    merged = True
                    break
            if not merged:
                self.attention_groups.append(
                    SpecGroup(spec, [i], manager_cls, use_eagle)
                )

        assert len(self.attention_groups) > 1, (
            "HybridKVCacheCoordinator requires at least two attention groups; "
            f"got {len(self.attention_groups)}"
        )
        # Sort full-attention first — its downward-closed scan provides a
        # tighter initial bound, reducing work for subsequent groups.
        self.attention_groups.sort(
            key=lambda g: not isinstance(g.spec, FullAttentionSpec)
            if isinstance(g.spec, type) and issubclass(g.spec, FullAttentionSpec)
            else True
        )
        # Propagate the eagle bit to each manager.
        for sg in self.attention_groups:
            if sg.use_eagle:
                for gid in sg.group_ids:
                    self.single_type_managers[gid].use_eagle = True

    def find_longest_cache_hit(
        self,
        block_hashes: list["BlockHash"],
        max_cache_hit_length: int,
    ) -> tuple[tuple[list["KVCacheBlock"], ...], int, int]:
        """Iterative fixed-point cache-hit reconciliation.

        Mirrors vLLM V1 HybridKVCacheCoordinator.find_longest_cache_hit.
        """
        num_groups = len(self.kv_cache_config.kv_cache_groups)
        hit_length = max_cache_hit_length
        hit_blocks_by_group: list[list["KVCacheBlock"] | None] = [
            None
        ] * num_groups
        hit_length_by_group: list[int] = [0] * num_groups

        # Tracks EAGLE/MTP groups whose drop was already verified at
        # the current candidate length (avoid re-dropping in the same
        # fixed-point iteration).
        eagle_verified: set[int] = set()

        # Simple hybrid short-circuit (1 full-attn + 1 other): one
        # iteration suffices.
        is_simple_hybrid = len(self.attention_groups) == 2 and (
            isinstance(self.attention_groups[0].spec, type)
            and issubclass(self.attention_groups[0].spec, FullAttentionSpec)
        )

        while True:
            curr_hit_length = hit_length
            for idx, (spec, group_ids, manager_cls, use_eagle) in enumerate(
                self.attention_groups
            ):
                first_group_id = group_ids[0]
                cached_blocks = hit_blocks_by_group[first_group_id]
                if (
                    isinstance(spec, type)
                    and issubclass(spec, FullAttentionSpec)
                    and cached_blocks is not None
                ):
                    # Full attention: downward-closed → just trim to the
                    # current (reduced) hit length.
                    curr_hit_length = min(
                        curr_hit_length, hit_length_by_group[first_group_id]
                    )
                    continue

                drop_eagle_block = use_eagle and idx not in eagle_verified
                _max_length = curr_hit_length
                if drop_eagle_block and not (
                    isinstance(spec, type) and spec.__name__ == "MambaSpec"
                ):
                    # Mamba: no margin (draft models have no mamba).
                    _max_length = min(
                        curr_hit_length + self.block_size
                        if hasattr(self, "block_size")
                        else curr_hit_length + self.scheduler_block_size,
                        max_cache_hit_length,
                    )
                hit_blocks, _new_hit_length = manager_cls.find_longest_cache_hit(
                    block_hashes=block_hashes,
                    max_length=_max_length,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=drop_eagle_block,
                    alignment_tokens=self.scheduler_block_size,
                    dcp_world_size=self.dcp_world_size,
                )
                if drop_eagle_block:
                    eagle_verified.add(idx)
                elif _new_hit_length < curr_hit_length:
                    # Length shrunk → invalidate prior eagle verifications.
                    eagle_verified.clear()
                curr_hit_length = _new_hit_length
                for group_id, blocks in zip(group_ids, hit_blocks):
                    hit_blocks_by_group[group_id] = blocks
                    hit_length_by_group[group_id] = _new_hit_length
            if curr_hit_length >= hit_length:
                break
            hit_length = curr_hit_length
            if is_simple_hybrid:
                break

        # Truncate full-attention blocks to the final hit_length.
        first_group = self.attention_groups[0]
        if isinstance(first_group.spec, type) and issubclass(
            first_group.spec, FullAttentionSpec
        ):
            group_block_size = self.single_type_managers[
                first_group.group_ids[0]
            ].block_size
            num_blocks = ceil(hit_length / group_block_size)
            for group_id in first_group.group_ids:
                if (blks := hit_blocks_by_group[group_id]) is not None:
                    del blks[num_blocks:]
                    hit_length_by_group[group_id] = hit_length

        # Uncached shared-prefix detection (0 unless hybrid).
        cache_hit_blocks = tuple(
            blocks if blocks is not None else []
            for blocks in hit_blocks_by_group
        )
        return cache_hit_blocks, hit_length, 0


# ---------------------------------------------------------------------------
# Coordinator factory
# ---------------------------------------------------------------------------


def get_kv_cache_coordinator(
    kv_cache_config: "KVCacheConfig",
    max_model_len: int,
    max_in_flight_tokens: int,
    use_eagle: bool,
    enable_caching: bool,
    enable_kv_cache_events: bool,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
    scheduler_block_size: int | None = None,
    hash_block_size: int | None = None,
    metrics_collector: "KVCacheMetricsCollector | None" = None,
    block_pool: "BlockPool | None" = None,
) -> KVCacheCoordinator:
    """Pick the right coordinator based on caching + group count.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:
    get_kv_cache_coordinator (L851).

    Dispatch:
    - Caching disabled → ``KVCacheCoordinatorNoPrefixCache``
    - Caching enabled + 1 group → ``UnitaryKVCacheCoordinatorFull``
    - Caching enabled + ≥ 2 groups → ``HybridKVCacheCoordinator``
    """
    if not enable_caching:
        return KVCacheCoordinatorNoPrefixCache(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            block_pool=block_pool,
        )
    if len(kv_cache_config.kv_cache_groups) == 1:
        return UnitaryKVCacheCoordinatorFull(
            kv_cache_config,
            max_model_len,
            max_in_flight_tokens,
            use_eagle,
            enable_caching,
            enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            metrics_collector=metrics_collector,
            block_pool=block_pool,
        )
    return HybridKVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        use_eagle,
        enable_caching,
        enable_kv_cache_events,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        hash_block_size=hash_block_size,
        metrics_collector=metrics_collector,
        block_pool=block_pool,
    )


# ===========================================================================
# Legacy C134 path — `UnitaryKVCacheCoordinator` (simple wrapper)
# ===========================================================================


class UnitaryKVCacheCoordinator:
    """Single-group coordinator — the only variant my-vllm supports.

    Mirrors vLLM V1 vllm/v1/core/kv_cache_coordinator.py:435 (the
    simple wrapper that `KVCacheManager` consumes in C134).

    Wraps a single `SingleTypeKVCacheManager` (typically
    `FullAttentionManager` for OPT). The flat `KVCacheBlocks` API is
    preserved so `KVCacheManager.allocate_slots` /
    `get_computed_blocks` / `free` / `get_block_ids` keep working
    unchanged.

    This class is **deliberately separate** from the new vLLM-
    compatible `UnitaryKVCacheCoordinatorFull` above. The full
    variant operates on a `KVCacheConfig` and is wired through the
    `get_kv_cache_coordinator` factory; this legacy variant is what
    the production `KVCacheManager` uses today.
    """

    def __init__(
        self,
        block_pool: "BlockPool",
        block_size: int,
        enable_caching: bool,
        manager: SingleTypeKVCacheManager | None = None,
    ):
        self.block_pool = block_pool
        self.block_size = block_size
        self.enable_caching = enable_caching
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
        return self.manager.get_num_blocks_to_allocate_simple(
            request, num_new_tokens
        )

    def allocate_new_blocks(
        self,
        request: "Request",
        num_new_tokens: int,
    ) -> "KVCacheBlocks | None":
        group_blocks = self.manager.allocate_new_blocks_simple(
            request, num_new_tokens
        )
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
        the underlying SingleTypeKVCacheManager).

        C143: distributed KV cache ↔ block-hash integration.
        Called after a successful prefill or after the
        `KVConnectorBase_V1` reports a prefix-cache hit (via
        `get_num_new_matched_tokens`).
        """
        self.manager.cache_blocks_simple(request, num_tokens)

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free
"""KVCacheCoordinator — dispatches KV cache operations across groups.

Mirrors vLLM V1 v1/core/kv_cache_coordinator.py.

C034: Coordinates multiple KVCacheManager instances, one per KV cache group.
      For OPT-125m (single FullAttention group), wraps a single KVCacheManager.
      For mixed-attention models, routes each group's operations independently.
"""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock
from my_vllm.v1.kv_cache_interface import KVCacheConfig
from my_vllm.v1.request import Request


class KVCacheCoordinator:
    """Coordinates KV cache operations across multiple KV cache groups.

    Each KV cache group (e.g. FullAttention, SlidingWindow) has its own
    KVCacheManager instance.  The coordinator routes block allocation,
    prefix cache lookups, and freeing to the appropriate group(s).

    Mirrors vLLM V1 v1/core/kv_cache_coordinator.py: KVCacheCoordinator.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        block_pool: BlockPool,
        enable_caching: bool = True,
    ):
        self.kv_cache_config = kv_cache_config
        self.block_pool = block_pool
        self.enable_caching = enable_caching

        self.managers: tuple[KVCacheManager, ...] = tuple(
            KVCacheManager(block_pool=block_pool)
            for _ in kv_cache_config.kv_cache_groups
        )

    def find_longest_cache_hit(
        self,
        block_hashes: list[bytes],
        max_cache_hit_length: int | None = None,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int]:
        """Find longest prefix cache hit across all KV cache groups.

        For each group, walks block_hashes sequentially and checks the
        BlockPool cache.  Returns the per-group hit blocks and the minimum
        hit length (tokens) across all groups — the intersection defines
        what is actually reusable.

        Args:
            block_hashes: consecutive block hashes for the request's prompt.
            max_cache_hit_length: optional cap on the hit length (tokens).

        Returns:
            (per_group_blocks, hit_tokens) where per_group_blocks is one
            list[KVCacheBlock] per KV cache group.
        """
        per_group: list[list[KVCacheBlock]] = []
        hit_lengths: list[int] = []

        for mgr in self.managers:
            cached = mgr.get_computed_blocks_mgr(block_hashes)
            per_group.append(cached)
            # Number of tokens covered by these blocks
            tokens = len(cached) * self.block_pool.block_size
            hit_lengths.append(tokens)

        if not per_group:
            return tuple(), 0

        # The common prefix across all groups is bounded by the group
        # with the fewest cache hits.  Read the last hit block(s) may
        # be partial, but the block alignment guarantees full block reuse.
        min_hit = min(hit_lengths) if hit_lengths else 0

        if max_cache_hit_length is not None:
            min_hit = min(min_hit, max_cache_hit_length)

        return tuple(per_group), min_hit

    def allocate_new_blocks(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> tuple[KVCacheBlocks, ...] | None:
        """Allocate blocks for num_new_tokens across all KV cache groups.

        Returns None if any group cannot satisfy the allocation (OOM).
        """
        results: list[KVCacheBlocks] = []
        for mgr in self.managers:
            blocks = mgr.allocate_slots(request, num_new_tokens)
            if blocks is None:
                # Rollback: caller should not use partial allocations
                return None
            results.append(blocks)
        return tuple(results)

    def free(self, request_id: str) -> None:
        """Free all blocks owned by request across all groups."""
        for mgr in self.managers:
            mgr.free(request_id)

    # --- backward-compat / convenience methods (used by Scheduler) ---

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int]:
        """Backward-compat: delegate to first manager (single-group case).

        Returns (KVCacheBlocks, num_cached_tokens) for the primary group.
        """
        if not self.managers:
            return KVCacheBlocks.empty(), 0
        return self.managers[0].get_computed_blocks(request)

    def allocate_slots(
        self, request: Request, num_tokens: int
    ) -> KVCacheBlocks | None:
        """Backward-compat: delegate to first manager (single-group case)."""
        if not self.managers:
            return None
        return self.managers[0].allocate_slots(request, num_tokens)

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        """Get blocks for primary group (single-group compat)."""
        if not self.managers:
            return []
        return self.managers[0].get_blocks(request_id)

    @property
    def num_free_blocks(self) -> int:
        if not self.managers:
            return 0
        return self.managers[0].num_free_blocks

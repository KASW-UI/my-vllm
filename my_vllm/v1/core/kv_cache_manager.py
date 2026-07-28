"""KVCacheManager — KV cache lifecycle manager with allocate_slots interface.

Mirrors vLLM V1 v1/core/kv_cache_manager.py.

C018: allocate_slots() replaces allocate().  Manages per-request block
tables and provides block ID lookups.  No longer creates KVCache objects
(HF past_key_values) — the GPUModelRunner handles actual KV storage.
"""

from dataclasses import dataclass

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock
from my_vllm.v1.request import Request


@dataclass
class KVCacheBlocks:
    """Allocation result bridging Scheduler ↔ KVCacheManager.

    Mirrors vLLM V1 kv_cache_manager.py:33.
    Hides internal block representation from the Scheduler.
    """

    block_ids: list[int]

    def __len__(self) -> int:
        return len(self.block_ids)

    @classmethod
    def empty(cls) -> "KVCacheBlocks":
        return cls(block_ids=[])


class KVCacheManager:
    """KV cache lifecycle manager with allocate_slots interface.

    C018: tracks per-request block ownership via request_blocks dict.
    Uses BlockPool (with FreeKVCacheBlockQueue + LRU eviction) for
    allocation and freeing.  No longer manages KVCache (HF past_kv)
    objects directly.
    """

    def __init__(
        self,
        block_pool: BlockPool | None = None,
        block_size: int = 16,
        num_blocks: int = 512,
    ):
        self.block_pool = block_pool or BlockPool(num_blocks, block_size)
        self.block_size = self.block_pool.block_size
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}

    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> KVCacheBlocks | None:
        """Allocate blocks for num_new_tokens. Returns block IDs or None.

        C029: respects pre-existing blocks from get_computed_blocks()
        (prefix cache hits).  Only allocates additional blocks beyond
        the already-cached portion.
        """
        total_blocks_needed = max(
            1, (num_new_tokens + self.block_size - 1) // self.block_size
        )

        existing = self.request_blocks.get(request.request_id, [])
        current_blocks = len(existing)
        new_blocks_needed = max(0, total_blocks_needed - current_blocks)

        new_blocks: list[KVCacheBlock] = []
        if new_blocks_needed > 0:
            new_blocks = self.block_pool.get_new_blocks(new_blocks_needed)
            if new_blocks is None:
                return None

        all_blocks = existing + new_blocks
        self.request_blocks[request.request_id] = all_blocks

        return KVCacheBlocks(
            block_ids=[b.block_id for b in all_blocks]
        )

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int]:
        """Find consecutive prefix cache hits for a request.

        C029: walks request.block_hashes from the start,
        checking BlockPool.get_cached_block() for each.
        Returns cached blocks + number of already-computed tokens.

        Args:
            request: the request to check for prefix cache hits.

        Returns:
            (KVCacheBlocks, num_new_computed_tokens) where:
            - KVCacheBlocks: cached blocks (may be empty)
            - num_new_computed_tokens: tokens covered by cache hits
        """
        if not self.block_pool.enable_caching:
            return KVCacheBlocks.empty(), 0

        block_size = self.block_pool.block_size
        request.compute_block_hashes(block_size)
        cached_blocks: list[KVCacheBlock] = []
        num_computed = 0

        for block_hash in request.block_hashes:
            if not block_hash:
                break
            cached = self.block_pool.get_cached_block(block_hash)
            if cached is None:
                break
            cached_blocks.append(cached)
            num_computed += min(
                block_size,
                len(request.prompt_token_ids) - num_computed,
            )

        if cached_blocks:
            self.request_blocks[request.request_id] = cached_blocks
            return KVCacheBlocks(
                block_ids=[b.block_id for b in cached_blocks]
            ), num_computed

        return KVCacheBlocks.empty(), 0

    def get_block_ids(self, request_id: str) -> list[int]:
        bt = self.request_blocks.get(request_id, [])
        return [b.block_id for b in bt]

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.request_blocks.get(request_id, [])

    def free(self, request_id: str):
        bt = self.request_blocks.pop(request_id, None)
        if bt:
            self.block_pool.free_blocks(bt)

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free

    # --- backward-compat methods (C014 tests) ---

    def allocate(self, request: Request) -> bool:
        """Compat: allocate blocks for a request's prompt tokens."""
        blocks = self.allocate_slots(request, len(request.prompt_token_ids))
        return blocks is not None

    def get_cache(self, request_id: str):
        """Compat: returns block list (truthy) instead of KVCache object."""
        blocks = self.request_blocks.get(request_id)
        return blocks if blocks else None

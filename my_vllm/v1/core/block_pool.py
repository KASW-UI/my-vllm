"""BlockPool — fixed-size KV cache block allocator.

Mirrors vLLM V1 v1/core/block_pool.py:143.

C018: Enhanced with FreeKVCacheBlockQueue (LRU eviction),
cached_block_hash_to_block (prefix cache lookup), touch(),
evict_blocks(), and ref_cnt-aware allocation.
"""

from my_vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from my_vllm.v1.request import Request


class BlockPool:
    """Fixed-size block allocator with LRU eviction.

    Mirrors vLLM V1 v1/core/block_pool.py:143.
    FreeKVCacheBlockQueue replaces the old _free_ids list.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        enable_caching: bool = False,
    ):
        assert num_blocks > 0
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching

        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=idx) for idx in range(num_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        self.cached_block_hash_to_block: dict[bytes, KVCacheBlock] = {}

    def get_new_blocks(self, n: int) -> list[KVCacheBlock] | None:
        """Allocate n blocks from the free queue. Returns None if insufficient."""
        if self.free_block_queue.num_free_blocks < n:
            return None
        blocks = list(self.free_block_queue.popleft_n(n))
        for b in blocks:
            b._is_free = False
            b.ref_cnt = 1
        return blocks

    def free_blocks(self, blocks: list[KVCacheBlock]):
        """Return blocks to the free queue (appended at tail, LRU order)."""
        for b in blocks:
            b.ref_cnt = 0
            b._is_free = True
            # C029: remove from prefix cache when freeing
            h = b.reset_hash()
            if h and h in self.cached_block_hash_to_block:
                del self.cached_block_hash_to_block[h]
            self.free_block_queue.append(b)

    def touch(self, blocks: list[KVCacheBlock]):
        """Refresh eviction position (move to tail)."""
        for b in blocks:
            self.free_block_queue.remove(b)
        for b in blocks:
            self.free_block_queue.append(b)

    def get_cached_block(self, block_hash: bytes) -> KVCacheBlock | None:
        """Look up a block by its hash (prefix cache)."""
        if not self.enable_caching:
            return None
        return self.cached_block_hash_to_block.get(block_hash)

    def cache_full_blocks(
        self, request: Request, blocks: list[KVCacheBlock]
    ):
        """Register block hashes for prefix caching."""
        if not self.enable_caching:
            return
        request.compute_block_hashes(self.block_size)
        for b, h in zip(blocks, request.block_hashes):
            if h:
                b.set_block_hash(h)
                self.cached_block_hash_to_block[h] = b

    @property
    def num_free(self) -> int:
        return self.free_block_queue.num_free_blocks

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

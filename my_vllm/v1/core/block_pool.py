"""BlockPool — fixed-size KV cache block allocator.

Mirrors vLLM V1 v1/core/block_pool.py:143.

C018: Enhanced with FreeKVCacheBlockQueue (LRU eviction),
cached_block_hash_to_block (prefix cache lookup), touch(),
evict_blocks(), and ref_cnt-aware allocation.

C035: BlockHashToBlockMap (hash→list[block] for collision handling),
ref_cnt cascading release in free_blocks() for GC.
"""

from typing import Any

from my_vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from my_vllm.v1.request import Request


class BlockHashToBlockMap:
    """Hash → list[block] mapping with collision support.

    Mirrors vLLM V1 v1/core/block_pool.py BlockHashToBlockMap.
    C032 had hash→block (1:1 dict). C035 upgrades to production:
    when multiple blocks share a hash (collision or dedup),
    the map stores them all via a dict[int, KVCacheBlock].

    Optimization: single block stored directly as KVCacheBlock
    (common case); promoted to dict on collision.
    """

    def __init__(self):
        self._cache: dict[
            bytes, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: bytes) -> KVCacheBlock | None:
        """Return any cached block for the given hash key."""
        blocks = self._cache.get(key)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            return blocks
        if isinstance(blocks, dict):
            return next(iter(blocks.values()))
        self._unexpected_type(blocks)
        return None

    def contain(self, key: bytes, block_id: int) -> bool:
        """Check whether key maps to the given block_id."""
        blocks = self._cache.get(key)
        if blocks is None:
            return False
        if isinstance(blocks, KVCacheBlock):
            return blocks.block_id == block_id
        if isinstance(blocks, dict):
            return block_id in blocks
        self._unexpected_type(blocks)
        return False

    def insert(self, key: bytes, block: KVCacheBlock) -> None:
        """Insert a block into the cache under key."""
        blocks = self._cache.get(key)
        if blocks is None:
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            blocks[block.block_id] = block
        else:
            self._unexpected_type(blocks)

    def pop(self, key: bytes, block_id: int) -> KVCacheBlock | None:
        """Remove and return the block with matching block_id under key."""
        blocks = self._cache.pop(key, None)
        if blocks is None:
            return None
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


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

        self.cached_block_hash_to_block = BlockHashToBlockMap()

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
        """Return blocks to the free queue with ref_cnt cascading release.

        C035: decrements ref_cnt instead of setting to 0.
        Blocks are only truly freed when ref_cnt reaches 0.
        Blocks with a hash stay in the prefix cache as LRU eviction
        candidates (like vLLM) — they may be re-used by other requests
        until actually evicted from the free queue.
        """
        for b in blocks:
            if b.ref_cnt > 0:
                b.ref_cnt -= 1
            if b.ref_cnt == 0:
                b._is_free = True
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
        return self.cached_block_hash_to_block.get_one_block(block_hash)

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
                self.cached_block_hash_to_block.insert(h, b)

    @property
    def num_free(self) -> int:
        return self.free_block_queue.num_free_blocks

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

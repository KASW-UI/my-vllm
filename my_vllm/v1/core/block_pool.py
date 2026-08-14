"""BlockPool — fixed-size KV cache block allocator.

Mirrors vLLM V1 v1/core/block_pool.py:143.

C018: Enhanced with FreeKVCacheBlockQueue (LRU eviction),
cached_block_hash_to_block (prefix cache lookup), touch(),
evict_blocks(), and ref_cnt-aware allocation.

C035: BlockHashToBlockMap (hash→list[block] for collision handling),
ref_cnt cascading release in free_blocks() for GC.

C054: KVCacheMetricsCollector integration — on_block_allocated/accessed/evicted.
"""

from typing import Any, TYPE_CHECKING

from my_vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock
from my_vllm.v1.request import Request

if TYPE_CHECKING:
    from my_vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector


class BlockHashToBlockMap:
    """Hash → list[block] mapping with collision support.

    Mirrors vLLM V1 v1/core/block_pool.py BlockHashToBlockMap.
    C032 had hash→block (1:1 dict). C035 upgrades to production:
    when multiple blocks share a hash (collision or dedup),
    the map stores them all via a dict[int, KVCacheBlock].

    Optimization: single block stored directly as KVCacheBlock
    (common case); promoted to dict on collision.

    C123: the two-branch (single-block vs dict-of-blocks) handling
    is consolidated through the `_unwrap_entry` helper, so the
    public methods stay readable.
    """

    def __init__(self):
        self._cache: dict[
            bytes, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: bytes) -> KVCacheBlock | None:
        """Return any cached block for the given hash key."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if isinstance(entry, KVCacheBlock):
            return entry
        if isinstance(entry, dict):
            return next(iter(entry.values()))
        raise AssertionError(f"Invalid KV cache block type {type(entry)}")

    def contain(self, key: bytes, block_id: int) -> bool:
        """Check whether key maps to the given block_id."""
        entry = self._cache.get(key)
        if entry is None:
            return False
        if isinstance(entry, KVCacheBlock):
            return entry.block_id == block_id
        if isinstance(entry, dict):
            return block_id in entry
        raise AssertionError(f"Invalid KV cache block type {type(entry)}")

    def insert(self, key: bytes, block: KVCacheBlock) -> None:
        """Insert a block into the cache under key."""
        entry = self._cache.get(key)
        if entry is None:
            self._cache[key] = block
        elif isinstance(entry, KVCacheBlock):
            self._cache[key] = {entry.block_id: entry, block.block_id: block}
        elif isinstance(entry, dict):
            entry[block.block_id] = block
        else:
            raise AssertionError(
                f"Invalid KV cache block type {type(entry)}",
            )

    def pop(self, key: bytes, block_id: int) -> KVCacheBlock | None:
        """Remove and return the block with matching block_id under key."""
        entry = self._cache.pop(key, None)
        if entry is None:
            return None
        if isinstance(entry, KVCacheBlock):
            if entry.block_id == block_id:
                return entry
            # Different block — restore.
            self._cache[key] = entry
            return None
        if isinstance(entry, dict):
            block = entry.pop(block_id, None)
            if len(entry) > 0:
                self._cache[key] = entry
            return block
        raise AssertionError(f"Invalid KV cache block type {type(entry)}")

    def __len__(self) -> int:
        return len(self._cache)


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
        metrics_collector: "KVCacheMetricsCollector | None" = None,
    ):
        assert num_blocks > 0
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.metrics_collector = metrics_collector

        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=idx) for idx in range(num_blocks)
        ]
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # Null block: a permanent sentinel placeholder (ref_cnt not
        # maintained, never freed). Used by sliding-window / Mamba
        # managers to pad skipped positions so block-table indices
        # remain stable across skipped regions.
        #
        # C254 deviation from vLLM (L190): we don't pop a real block
        # from the pool for this. Instead we synthesize a KVCacheBlock
        # with a sentinel block_id that's not in `self.blocks`. Tests
        # (C008 / C018) expect block IDs to start at 0 with no offset.
        # The sentinel is identity-equality unique (`is null_block` /
        # `== null_block`) but never participates in pool accounting.
        self.null_block = KVCacheBlock(block_id=-1)
        self.null_block.is_null = True

        self.cached_block_hash_to_block = BlockHashToBlockMap()

    def get_new_blocks(self, n: int) -> list[KVCacheBlock] | None:
        """Allocate n blocks from the free queue. Returns None if insufficient."""
        if self.free_block_queue.num_free_blocks < n:
            return None
        blocks = list(self.free_block_queue.popleft_n(n))
        for b in blocks:
            self._maybe_evict_cached_block(b)
            b._is_free = False
            b.ref_cnt = 1
            if self.metrics_collector is not None:
                self.metrics_collector.on_block_allocated(b)
        return blocks

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> None:
        """Evict stale hash entries when a block is re-allocated."""
        if not self.enable_caching:
            return
        old_hash = block.block_hash
        if old_hash is not None:
            self.cached_block_hash_to_block.pop(old_hash, block.block_id)
            block.reset_hash()

    def free_blocks(self, blocks: list[KVCacheBlock]):
        """Return blocks to free queue with eviction-order split.

        Mirrors vLLM V1 BlockPool.free_blocks(): blocks without hash
        are prepended (evicted first), blocks with hash are appended
        (evicted last, preserved for prefix cache reuse).
        """
        with_hash, without_hash = self._split_blocks_for_free(blocks)
        self.free_block_queue.prepend_n(without_hash)
        for b in with_hash:
            self.free_block_queue.append(b)

    def _split_blocks_for_free(
        self, blocks: list[KVCacheBlock],
    ) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        """Decrement ref_cnt; partition into (with_hash, without_hash).

        Only blocks whose ref_cnt dropped to 0 (and aren't null) are
        returned for queue insertion. Each list preserves input order.
        """
        with_hash: list[KVCacheBlock] = []
        without_hash: list[KVCacheBlock] = []
        for b in blocks:
            if b.ref_cnt > 0:
                b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                b._is_free = True
                if b.block_hash is not None:
                    with_hash.append(b)
                else:
                    without_hash.append(b)
        return with_hash, without_hash

    def touch(self, blocks: list[KVCacheBlock]):
        """Pin cached blocks by removing from free list and incrementing ref_cnt.

        Mirrors vLLM V1 BlockPool.touch(): only operates on free blocks
        (ref_cnt == 0), increments ref_cnt to prevent eviction.
        """
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free_block_queue.remove(b)
                b.ref_cnt += 1
                if self.metrics_collector is not None:
                    self.metrics_collector.on_block_accessed(b)

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

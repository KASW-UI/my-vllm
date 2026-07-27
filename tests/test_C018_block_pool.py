"""C018: tests for enhanced BlockPool (v1/core/block_pool.py)."""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock
from my_vllm.v1.request import Request


class TestBlockPool:
    def test_allocate_free(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        assert pool.num_free == 10
        assert pool.num_used == 0

        blocks = pool.get_new_blocks(3)
        assert blocks is not None
        assert len(blocks) == 3
        assert pool.num_free == 7
        assert pool.num_used == 3
        assert all(not b.is_free for b in blocks)
        assert all(b.ref_cnt == 1 for b in blocks)

        pool.free_blocks(blocks)
        assert pool.num_free == 10
        assert pool.num_used == 0
        assert all(b.is_free for b in blocks)
        assert all(b.ref_cnt == 0 for b in blocks)
        assert all(b.block_hash is None for b in blocks)

    def test_exhaust_and_reuse(self):
        pool = BlockPool(num_blocks=3, block_size=16)
        a = pool.get_new_blocks(3)
        assert pool.num_free == 0
        pool.free_blocks(a)
        assert pool.num_free == 3
        b = pool.get_new_blocks(3)
        assert b is not None
        assert len(b) == 3

    def test_insufficient_blocks(self):
        pool = BlockPool(num_blocks=5, block_size=16)
        assert pool.get_new_blocks(6) is None
        assert pool.num_free == 5

    def test_block_id_ordering(self):
        """Initial allocation returns blocks in increasing order."""
        pool = BlockPool(num_blocks=10, block_size=16)
        blocks = pool.get_new_blocks(4)
        assert [b.block_id for b in blocks] == [0, 1, 2, 3]

    def test_fragmentation_resistant(self):
        pool = BlockPool(num_blocks=5, block_size=16)
        a = pool.get_new_blocks(3)
        b = pool.get_new_blocks(2)
        assert pool.num_free == 0
        pool.free_blocks(a)
        assert pool.num_free == 3
        c = pool.get_new_blocks(3)
        assert c is not None
        # After freeing a[0,1,2], they're at the END of free queue
        # b[3,4] was allocated at the start. After freeing a,
        # free queue: [3,4 re-???] — but b was never freed...
        # Actually after poping b[3,4], free queue is empty.
        # After free(a) appends [0,1,2], free queue = [0,1,2]
        # So get_new_blocks(3) → [0,1,2]
        assert [blk.block_id for blk in c] == [0, 1, 2]

    def test_block_hash_cache(self):
        pool = BlockPool(num_blocks=10, block_size=16, enable_caching=True)
        assert pool.enable_caching
        # No hash cached initially
        assert pool.get_cached_block(b"xyz") is None

    def test_block_hash_cache_disabled(self):
        pool = BlockPool(num_blocks=10, block_size=16, enable_caching=False)
        assert not pool.enable_caching
        assert pool.get_cached_block(b"xyz") is None

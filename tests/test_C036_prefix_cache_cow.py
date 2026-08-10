"""C036: tests for PrefixCache Copy-on-Write protection."""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from my_vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from my_vllm.v1.core.prefix_cache import PrefixCache
from my_vllm.v1.request import Request


def _make_request(rid="r1", tokens=None):
    return Request(
        request_id=rid,
        prompt_token_ids=(tokens if tokens is not None else [1, 2, 3, 4]),
    )


class TestPrefixCacheCoW:
    def test_block_not_shared_no_cow(self):
        """Single-owner block: get_block_for_write returns the same block."""
        pool = BlockPool(32, enable_caching=True)
        pc = PrefixCache(pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        assert b.ref_cnt == 1
        result = pc.get_block_for_write(b)
        assert result is b
        assert b.ref_cnt == 1
        assert len(pc.get_copies()) == 0

    def test_shared_block_triggers_cow(self):
        """Shared block (ref_cnt>1) → CoW allocates new block, returns it."""
        pool = BlockPool(32, enable_caching=True)
        pc = PrefixCache(pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.ref_cnt = 2  # simulate shared
        result = pc.get_block_for_write(b)
        assert result is not b
        assert result.block_id != b.block_id
        assert b.ref_cnt == 1  # decremented
        assert result.ref_cnt == 1  # new block starts ref_cnt=1
        copies = pc.get_copies()
        assert len(copies) == 1
        assert copies[0] == KVCacheBlockCopy(
            src_block_id=b.block_id, dst_block_id=result.block_id
        )

    def test_cow_when_free_blocks_exhausted(self):
        """CoW when pool is exhausted falls back to original block."""
        pool = BlockPool(1, enable_caching=True)
        pc = PrefixCache(pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.ref_cnt = 2
        result = pc.get_block_for_write(b)
        assert result is b  # fallback, no copy possible

    def test_multiple_copies(self):
        """Multiple shared blocks each get their own CoW clone."""
        pool = BlockPool(32, enable_caching=True)
        pc = PrefixCache(pool)
        blks_a = pool.get_new_blocks(2)
        b0, b1 = blks_a
        b0.ref_cnt = 2
        b1.ref_cnt = 3
        r0 = pc.get_block_for_write(b0)
        r1 = pc.get_block_for_write(b1)
        assert r0.block_id != b0.block_id
        assert r1.block_id != b1.block_id
        assert r0.block_id != r1.block_id
        assert b0.ref_cnt == 1
        assert b1.ref_cnt == 2
        copies = pc.get_copies()
        assert len(copies) == 2

    def test_clear_copies(self):
        pool = BlockPool(32, enable_caching=True)
        pc = PrefixCache(pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.ref_cnt = 2
        pc.get_block_for_write(b)
        assert len(pc.get_copies()) == 1
        pc.clear_copies()
        assert len(pc.get_copies()) == 0


class TestKVCacheManagerCoW:
    def test_computed_blocks_triggers_cow_on_shared(self):
        """get_computed_blocks clones shared blocks via PrefixCache."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        # Request A: allocate + cache its blocks
        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))
        blk_a = mgr.get_blocks("a")[0]
        assert blk_a.ref_cnt == 1

        # Request B: same tokens → get_computed_blocks should return
        # the same block (but with ref_cnt=2 now since cached_block
        # already had ref_cnt=1 from get_new_blocks on A)
        req_b = _make_request("b", tokens=list(range(16)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16
        assert len(blks_b) == 1

        # After get_computed_blocks, both requests have private copies
        # (CoW cloned B's block because the cached block was shared)
        blk_b = mgr.get_blocks("b")[0]
        # B has a different block than A (CoW happened)
        assert blk_b.block_id != blk_a.block_id
        # Original block ref_cnt went back to 1 (shared→CoW→decremented)
        assert blk_a.ref_cnt == 1

    def test_computed_blocks_cow_records_copy(self):
        """CoW via KVCacheManager records KVCacheBlockCopy."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        req_b = _make_request("b", tokens=list(range(16)))
        mgr.get_computed_blocks(req_b)
        copies = mgr.prefix_cache.get_copies()
        assert len(copies) == 1
        assert copies[0].src_block_id == mgr.get_blocks("a")[0].block_id
        assert copies[0].dst_block_id == mgr.get_blocks("b")[0].block_id

    def test_no_cow_when_not_shared(self):
        """get_computed_blocks without prior cached request: no CoW."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        # No previously cached blocks
        req = _make_request("x", tokens=list(range(16)))
        blks, num = mgr.get_computed_blocks(req)
        assert num == 0  # no cache hit
        assert len(blks) == 0
        assert len(mgr.prefix_cache.get_copies()) == 0

    def test_partial_hit_cow(self):
        """Partial cache hit: shared blocks get CoW, miss blocks allocate new."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        # Cache the first 16 tokens (1 block) from request A
        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        # Request B: 32 tokens (2 blocks), first block matches
        req_b = _make_request("b", tokens=list(range(32)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16
        assert len(blks_b) == 1

        # The first block should be a CoW clone, not the original
        blk_b = mgr.get_blocks("b")[0]
        blk_a = mgr.get_blocks("a")[0]
        assert blk_b.block_id != blk_a.block_id
        copies = mgr.prefix_cache.get_copies()
        assert len(copies) == 1

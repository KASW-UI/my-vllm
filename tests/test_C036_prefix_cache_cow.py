"""C036: tests for Copy-on-Write protection on shared KV cache blocks."""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from my_vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from my_vllm.v1.request import Request


def _make_request(rid="r1", tokens=None):
    return Request(
        request_id=rid,
        prompt_token_ids=(tokens if tokens is not None else [1, 2, 3, 4]),
    )


class TestKVCacheManagerCoWApply:
    def test_block_not_shared_no_cow(self):
        """Single-owner block: no CoW, returned as-is."""
        pool = BlockPool(32, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        assert b.ref_cnt == 1
        result = mgr._apply_cow(b)
        assert result is b
        assert b.ref_cnt == 1
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 0

    def test_shared_block_triggers_cow(self):
        """Shared block (ref_cnt>1) → CoW allocates new block."""
        pool = BlockPool(32, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.ref_cnt = 2
        result = mgr._apply_cow(b)
        assert result is not b
        assert result.block_id != b.block_id
        assert b.ref_cnt == 1
        assert result.ref_cnt == 1
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 1
        assert copies[0] == KVCacheBlockCopy(
            src_block_id=b.block_id, dst_block_id=result.block_id
        )

    def test_cow_when_free_blocks_exhausted(self):
        """CoW when pool exhausted falls back to original."""
        pool = BlockPool(1, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.ref_cnt = 2
        result = mgr._apply_cow(b)
        assert result is b

    def test_multiple_copies(self):
        """Multiple shared blocks each get their own CoW clone."""
        pool = BlockPool(32, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)
        blks_a = pool.get_new_blocks(2)
        b0, b1 = blks_a
        b0.ref_cnt = 2
        b1.ref_cnt = 3
        r0 = mgr._apply_cow(b0)
        r1 = mgr._apply_cow(b1)
        assert r0.block_id != b0.block_id
        assert r1.block_id != b1.block_id
        assert r0.block_id != r1.block_id
        assert b0.ref_cnt == 1
        assert b1.ref_cnt == 2
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 2


class TestKVCacheManagerCoWIntegration:
    def test_computed_blocks_triggers_cow_on_shared(self):
        """get_computed_blocks clones shared blocks."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))
        blk_a = mgr.get_blocks("a")[0]
        assert blk_a.ref_cnt == 1

        req_b = _make_request("b", tokens=list(range(16)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16
        assert len(blks_b) == 1

        blk_b = mgr.get_blocks("b")[0]
        assert blk_b.block_id != blk_a.block_id
        assert blk_a.ref_cnt == 1

    def test_computed_blocks_cow_records_copy(self):
        """CoW records KVCacheBlockCopy for Worker."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        req_b = _make_request("b", tokens=list(range(16)))
        mgr.get_computed_blocks(req_b)
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 1
        assert copies[0].src_block_id == mgr.get_blocks("a")[0].block_id
        assert copies[0].dst_block_id == mgr.get_blocks("b")[0].block_id

    def test_no_cow_when_not_shared(self):
        """No prior cached requests means no CoW."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        req = _make_request("x", tokens=list(range(16)))
        blks, num = mgr.get_computed_blocks(req)
        assert num == 0
        assert len(blks) == 0
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 0

    def test_partial_hit_cow(self):
        """Partial cache hit: shared blocks get CoW."""
        pool = BlockPool(64, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        req_b = _make_request("b", tokens=list(range(32)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16
        assert len(blks_b) == 1

        blk_b = mgr.get_blocks("b")[0]
        blk_a = mgr.get_blocks("a")[0]
        assert blk_b.block_id != blk_a.block_id
        copies, _ = mgr.take_kv_cache_block_copies()
        assert len(copies) == 1

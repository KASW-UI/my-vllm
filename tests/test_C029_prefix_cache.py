"""C029: tests for prefix caching (BlockPool hash cache + KVCacheManager)."""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from my_vllm.v1.request import Request


def _make_request(rid="r1", tokens=None):
    req = Request(
        request_id=rid,
        prompt_token_ids=(tokens if tokens is not None else [1, 2, 3, 4]),
    )
    return req


class TestBlockHashCompute:
    def test_block_hashes_computed(self):
        req = _make_request(tokens=list(range(16)))
        hashes = req.compute_block_hashes()
        assert len(hashes) == 1
        assert isinstance(hashes[0], bytes)
        assert len(hashes[0]) == 8

    def test_multi_block_hashes(self):
        req = _make_request(tokens=list(range(32)))
        hashes = req.compute_block_hashes()
        assert len(hashes) == 2
        assert hashes[0] != hashes[1]

    def test_same_prompt_same_hash(self):
        r1 = _make_request(tokens=list(range(16)))
        r2 = _make_request(tokens=list(range(16)))
        assert r1.compute_block_hashes()[0] == r2.compute_block_hashes()[0]

    def test_empty_prompt(self):
        req = _make_request(tokens=[])
        assert req.compute_block_hashes() == []


class TestBlockPoolCache:
    def test_get_cached_block_disabled(self):
        pool = BlockPool(32, enable_caching=False)
        assert pool.get_cached_block(b"abc") is None

    def test_cache_and_retrieve(self):
        pool = BlockPool(32, enable_caching=True)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.set_block_hash(b"test")
        pool.cached_block_hash_to_block.insert(b"test", b)
        assert pool.get_cached_block(b"test") is b

    def test_free_removes_from_cache(self):
        pool = BlockPool(32, enable_caching=True)
        blocks = pool.get_new_blocks(1)
        b = blocks[0]
        b.set_block_hash(b"myhash")
        pool.cached_block_hash_to_block.insert(b"myhash", b)
        assert pool.get_cached_block(b"myhash") is b
        pool.free_blocks([b])
        assert pool.get_cached_block(b"myhash") is None

    def test_cache_full_blocks(self):
        pool = BlockPool(32, enable_caching=True)
        blocks = pool.get_new_blocks(2)
        req = _make_request(tokens=list(range(32)))
        req.compute_block_hashes()
        pool.cache_full_blocks(req, blocks)
        assert pool.get_cached_block(req.block_hashes[0]) is blocks[0]
        assert pool.get_cached_block(req.block_hashes[1]) is blocks[1]


class TestKVCacheManagerComputed:
    def test_get_computed_blocks_no_cache(self):
        pool = BlockPool(32, enable_caching=False)
        mgr = KVCacheManager(block_pool=pool)
        req = _make_request(tokens=list(range(16)))
        blocks, num = mgr.get_computed_blocks(req)
        assert len(blocks) == 0
        assert num == 0

    def test_get_computed_blocks_hit(self):
        pool = BlockPool(32, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        # First: allocate and cache for request A
        req_a = _make_request("a", tokens=list(range(16)))
        blks_a = mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        # Second: request B with same tokens should get cache hit
        req_b = _make_request("b", tokens=list(range(16)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16
        assert len(blks_b) == 1
        assert blks_b.block_ids == blks_a.block_ids  # same physical blocks

    def test_get_computed_blocks_partial_hit(self):
        pool = BlockPool(32, enable_caching=True)
        mgr = KVCacheManager(block_pool=pool)

        # Request A: 16 tokens (1 block) — cache it
        req_a = _make_request("a", tokens=list(range(16)))
        mgr.allocate_slots(req_a, 16)
        pool.cache_full_blocks(req_a, mgr.get_blocks("a"))

        # Request B: 32 tokens (2 blocks), first block matches
        req_b = _make_request("b", tokens=list(range(32)))
        blks_b, num = mgr.get_computed_blocks(req_b)
        assert num == 16  # only first block cached
        assert len(blks_b) == 1


class TestSchedulerPrefixCache:
    def test_schedule_with_cache_hit(self):
        from my_vllm.v1.core.block_pool import BlockPool as BP
        from my_vllm.v1.core.kv_cache_manager import KVCacheManager as KM
        from my_vllm.v1.core.sched.scheduler import Scheduler

        pool = BP(64, enable_caching=True)
        mgr = KM(block_pool=pool)
        sched = Scheduler(max_running=8, kv_cache_manager=mgr)

        # First request — allocate and finish (cache its blocks)
        req_a = _make_request("a", tokens=list(range(16)))
        sched.add_request(req_a)
        sched.schedule()
        assert req_a.status.name == "RUNNING"
        sched.finish(req_a)

        # Second request — same tokens, should get cache hit
        req_b = _make_request("b", tokens=list(range(16)))
        sched.add_request(req_b)
        output = sched.schedule()
        assert req_b.num_computed_tokens == 16
        assert output.scheduled_new_reqs[0].num_computed_tokens == 16

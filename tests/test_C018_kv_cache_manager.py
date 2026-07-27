"""C018: tests for KVCacheManager allocate_slots() (v1/core/kv_cache_manager.py)."""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_manager import KVCacheManager, KVCacheBlocks
from my_vllm.v1.request import Request


class TestKVCacheBlocks:
    def test_empty(self):
        kb = KVCacheBlocks.empty()
        assert kb.block_ids == []
        assert len(kb) == 0

    def test_full(self):
        kb = KVCacheBlocks(block_ids=[1, 2, 3])
        assert kb.block_ids == [1, 2, 3]
        assert len(kb) == 3


class TestKVCacheManager:
    def test_allocate_slots_new_request(self):
        pool = BlockPool(num_blocks=50, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("t1")
        req.prompt_token_ids = list(range(32))  # 2 blocks
        result = mgr.allocate_slots(req, len(req.prompt_token_ids))
        assert result is not None
        assert isinstance(result, KVCacheBlocks)
        assert len(result.block_ids) == 2
        assert result.block_ids == [0, 1]
        assert mgr.num_free_blocks == 48

    def test_allocate_slots_insufficient(self):
        pool = BlockPool(num_blocks=3, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("big")
        req.prompt_token_ids = list(range(100))  # needs 7 blocks
        result = mgr.allocate_slots(req, len(req.prompt_token_ids))
        assert result is None

    def test_free(self):
        pool = BlockPool(num_blocks=20, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("f1")
        req.prompt_token_ids = list(range(32))
        mgr.allocate_slots(req, len(req.prompt_token_ids))
        assert mgr.num_free_blocks == 18

        mgr.free(req.request_id)
        assert mgr.num_free_blocks == 20
        assert mgr.get_block_ids(req.request_id) == []
        assert mgr.get_blocks(req.request_id) == []
        assert mgr.get_cache(req.request_id) is None

    def test_get_block_ids(self):
        pool = BlockPool(num_blocks=20, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("g1")
        req.prompt_token_ids = list(range(48))  # 3 blocks
        mgr.allocate_slots(req, len(req.prompt_token_ids))
        assert mgr.get_block_ids(req.request_id) == [0, 1, 2]

    def test_allocate_compat(self):
        """Backward compat: allocate() returns bool."""
        pool = BlockPool(num_blocks=50, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("c1")
        req.prompt_token_ids = list(range(32))
        assert mgr.allocate(req) is True
        assert mgr.get_cache(req.request_id) is not None

    def test_allocate_compat_insufficient(self):
        pool = BlockPool(num_blocks=2, block_size=16)
        mgr = KVCacheManager(block_pool=pool)

        req = Request("c2")
        req.prompt_token_ids = list(range(100))
        assert mgr.allocate(req) is False

    def test_num_free_blocks(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = KVCacheManager(block_pool=pool)
        assert mgr.num_free_blocks == 10

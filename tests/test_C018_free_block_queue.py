"""C018: tests for FreeKVCacheBlockQueue (v1/core/kv_cache_utils.py)."""

import pytest

from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, FreeKVCacheBlockQueue


class TestFreeKVCacheBlockQueue:
    @staticmethod
    def _make_blocks(n: int) -> list[KVCacheBlock]:
        return [KVCacheBlock(block_id=i) for i in range(n)]

    def test_init(self):
        blocks = self._make_blocks(5)
        q = FreeKVCacheBlockQueue(blocks)
        assert q.num_free_blocks == 5
        assert not q.is_empty

    def test_init_empty(self):
        q = FreeKVCacheBlockQueue([])
        assert q.num_free_blocks == 0
        assert q.is_empty

    def test_popleft(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(3))
        b = q.popleft()
        assert b.block_id == 0
        assert q.num_free_blocks == 2

    def test_popleft_empty_raises(self):
        q = FreeKVCacheBlockQueue([])
        with pytest.raises(ValueError, match="No free blocks"):
            q.popleft()

    def test_popleft_n(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(5))
        blocks = q.popleft_n(3)
        assert len(blocks) == 3
        assert [b.block_id for b in blocks] == [0, 1, 2]
        assert q.num_free_blocks == 2

    def test_popleft_n_exhaust(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(2))
        blocks = q.popleft_n(2)
        assert [b.block_id for b in blocks] == [0, 1]
        assert q.num_free_blocks == 0

    def test_popleft_n_zero(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(3))
        blocks = q.popleft_n(0)
        assert blocks == []
        assert q.num_free_blocks == 3

    def test_append_restores_count(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(5))
        b = q.popleft()
        assert q.num_free_blocks == 4
        q.append(b)
        assert q.num_free_blocks == 5

    def test_lru_order_after_recycle(self):
        """Blocks cycled through pop-append return in LRU order."""
        q = FreeKVCacheBlockQueue(self._make_blocks(4))
        b0 = q.popleft()  # pop 0
        b1 = q.popleft()  # pop 1
        # remaining: [2, 3]
        q.append(b0)  # append 0 at tail → [2, 3, 0]
        q.append(b1)  # append 1 at tail → [2, 3, 0, 1]
        popped = q.popleft_n(4)
        assert [b.block_id for b in popped] == [2, 3, 0, 1]

    def test_remove_middle(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(5))
        # pop 0,1, append them back
        a = q.popleft_n(2)  # [0, 1]
        q.append(a[0])  # append 0
        q.append(a[1])  # append 1
        # free list: [2, 3, 4, 0, 1]
        # pop to keep refs, then remove block 3
        all_refs = q.popleft_n(5)
        # put them all back except block 3
        for b in all_refs:
            if b.block_id != 3:
                q.append(b)
        assert q.num_free_blocks == 4
        # remaining: [2, 4, 0, 1]
        final = q.popleft_n(4)
        assert [b.block_id for b in final] == [2, 4, 0, 1]

    def test_clear(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(5))
        q.clear()
        assert q.num_free_blocks == 0
        assert q.is_empty

    def test_consecutive_alloc_free(self):
        q = FreeKVCacheBlockQueue(self._make_blocks(100))
        used = q.popleft_n(80)
        assert q.num_free_blocks == 20
        for b in used[:40]:
            q.append(b)
        assert q.num_free_blocks == 60
        more = q.popleft_n(30)
        assert len(more) == 30
        assert q.num_free_blocks == 30

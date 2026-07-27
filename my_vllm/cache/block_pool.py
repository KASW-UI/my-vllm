from my_vllm.cache.block import KVCacheBlock


class BlockPool:
    """Fixed-size block allocator — solves the fragmentation of C008.

    Maintains a pool of KVCacheBlock instances with a free-block
    queue.  Each block holds `block_size` tokens of KV cache.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self._free_ids: list[int] = list(range(num_blocks))

    def get_new_blocks(self, n: int) -> list[KVCacheBlock] | None:
        if len(self._free_ids) < n:
            return None
        ids = self._free_ids[:n]
        del self._free_ids[:n]
        blocks = [self.blocks[i] for i in ids]
        for b in blocks:
            b.is_free = False
        return blocks

    def free_blocks(self, blocks: list[KVCacheBlock]):
        for b in blocks:
            b.is_free = True
            self._free_ids.append(b.block_id)

    @property
    def num_free(self) -> int:
        return len(self._free_ids)

    @property
    def num_used(self) -> int:
        return self.num_blocks - self.num_free

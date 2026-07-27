from my_vllm.cache.kv_cache import KVCache
from my_vllm.cache.block_pool import BlockPool
from my_vllm.cache.block_table import BlockTable
from my_vllm.engine.request import Request


class KVCacheManager:
    """KV cache lifecycle manager with block-based allocation.

    C010: execution state (_steps/_next_token) moved to ModelRunner.
    KVCacheManager now purely manages blocks and KVCache instances.
    """

    def __init__(
        self,
        block_pool: BlockPool | None = None,
        block_size: int = 16,
        num_blocks: int = 512,
    ):
        self.block_pool = block_pool or BlockPool(num_blocks, block_size)
        self.block_size = self.block_pool.block_size
        self._caches: dict[str, KVCache] = {}
        self._block_tables: dict[str, BlockTable] = {}

    def allocate(self, request: Request) -> bool:
        num_tokens = len(request.prompt_token_ids)
        num_blocks = max(1, (num_tokens + self.block_size - 1)
                         // self.block_size)
        blocks = self.block_pool.get_new_blocks(num_blocks)
        if blocks is None:
            return False
        block_table = BlockTable(self.block_size)
        block_table.append(blocks)
        self._block_tables[request.request_id] = block_table
        self._caches[request.request_id] = KVCache()
        return True

    def get_cache(self, request_id: str) -> KVCache | None:
        return self._caches.get(request_id)

    def get_block_ids(self, request_id: str) -> list[int]:
        bt = self._block_tables.get(request_id)
        return bt.block_ids if bt else []

    def free(self, request_id: str):
        bt = self._block_tables.pop(request_id, None)
        if bt is not None and bt.blocks:
            self.block_pool.free_blocks(list(bt.blocks))
        self._caches.pop(request_id, None)

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free

from my_vllm.cache.kv_cache import KVCache
from my_vllm.cache.block import KVCacheBlock
from my_vllm.cache.block_pool import BlockPool
from my_vllm.cache.block_table import BlockTable
from my_vllm.engine.request import Request


class KVCacheManager:
    """KV cache lifecycle manager with block-based allocation.

    C009: uses BlockPool + BlockTable instead of continuous
    allocator, eliminating fragmentation.
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
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

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

    def get_step(self, request_id: str) -> int:
        return self._steps.get(request_id, 0)

    def set_step(self, request_id: str, step: int):
        self._steps[request_id] = step

    def get_next_token(self, request_id: str) -> int | None:
        return self._next_token.get(request_id)

    def set_next_token(self, request_id: str, token: int):
        self._next_token[request_id] = token

    def free(self, request_id: str):
        bt = self._block_tables.pop(request_id, None)
        if bt is not None and bt.blocks:
            self.block_pool.free_blocks(list(bt.blocks))
        self._caches.pop(request_id, None)
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free

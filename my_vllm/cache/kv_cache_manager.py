from my_vllm.cache.kv_cache import KVCache
from my_vllm.cache.allocator import KVCacheAllocator
from my_vllm.engine.request import Request


class KVCacheManager:
    """Central KV cache lifecycle manager.

    Owns the request_id → KVCache mapping and coordinates
    with the allocator for memory budgeting.
    Scheduler queries can_allocate() before promoting requests.
    """

    def __init__(self, allocator: KVCacheAllocator | None = None):
        self.allocator = allocator or KVCacheAllocator()
        self._caches: dict[str, KVCache] = {}
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        return self.allocator.can_allocate(num_tokens)

    def allocate(self, request: Request) -> bool:
        num_tokens = len(request.prompt_token_ids) + 50
        if not self.allocator.can_allocate(num_tokens):
            return False
        self.allocator.reserve(request.request_id, num_tokens)
        self._caches[request.request_id] = KVCache()
        return True

    def get_cache(self, request_id: str) -> KVCache | None:
        return self._caches.get(request_id)

    def get_step(self, request_id: str) -> int:
        return self._steps.get(request_id, 0)

    def set_step(self, request_id: str, step: int):
        self._steps[request_id] = step

    def get_next_token(self, request_id: str) -> int | None:
        return self._next_token.get(request_id)

    def set_next_token(self, request_id: str, token: int):
        self._next_token[request_id] = token

    def free(self, request_id: str):
        self.allocator.release(request_id)
        self._caches.pop(request_id, None)
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)

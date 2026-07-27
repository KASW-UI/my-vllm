class KVCacheAllocator:
    """Token-budget tracker for continuous KV cache memory.

    Pre-paged (C008): allocates by token count, not blocks.
    Continuous allocation causes fragmentation — fixed in C009.
    """

    def __init__(self, max_total_tokens: int = 8192):
        self.max_total_tokens = max_total_tokens
        self._reserved: dict[str, int] = {}

    def can_allocate(self, num_tokens: int) -> bool:
        total = sum(self._reserved.values())
        return total + num_tokens <= self.max_total_tokens

    def reserve(self, request_id: str, num_tokens: int):
        self._reserved[request_id] = num_tokens

    def release(self, request_id: str):
        self._reserved.pop(request_id, None)

    def used(self) -> int:
        return sum(self._reserved.values())

    def free(self) -> int:
        return self.max_total_tokens - self.used()

from my_vllm.engine.request import Request, RequestStatus
from my_vllm.cache.kv_cache_manager import KVCacheManager


class Scheduler:
    """FCFS scheduler with max batch size and KV cache awareness.

    C008: now queries KVCacheManager.can_allocate() before
    promoting requests from waiting → running.
    """

    def __init__(
        self,
        max_running: int = 8,
        kv_cache_manager: KVCacheManager | None = None,
    ):
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.max_running = max_running
        self.kv_cache_manager = kv_cache_manager

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def schedule(self) -> list[Request]:
        new_requests: list[Request] = []
        while len(self.running) < self.max_running and self.waiting:
            req = self.waiting.pop(0)

            if self.kv_cache_manager is not None:
                if not self.kv_cache_manager.allocate(req):
                    self.waiting.insert(0, req)
                    break

            req.status = RequestStatus.RUNNING
            self.running.append(req)
            new_requests.append(req)
        return new_requests

    def finish(self, request: Request):
        if request in self.running:
            self.running.remove(request)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

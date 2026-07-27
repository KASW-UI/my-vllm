from my_vllm.engine.request import Request, RequestStatus


class Scheduler:
    """FCFS scheduler with max batch size.

    Extracted from StateManager in C007.  Owns the waiting/running
    queues and the scheduling policy.  Does NOT know about KV cache
    resources (defect left for C008: KVCacheManager).
    """

    def __init__(self, max_running: int = 8):
        self.waiting: list[Request] = []
        self.running: list[Request] = []
        self.max_running = max_running

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    def schedule(self) -> list[Request]:
        """Promote waiting → running (FCFS) and return the new ones."""
        new_requests: list[Request] = []
        while len(self.running) < self.max_running and self.waiting:
            req = self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            new_requests.append(req)
        return new_requests

    def finish(self, request: Request):
        if request in self.running:
            self.running.remove(request)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

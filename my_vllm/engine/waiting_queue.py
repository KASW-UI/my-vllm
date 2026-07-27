from my_vllm.engine.request import Request, RequestStatus


class WaitingQueue:
    def __init__(self):
        self._queue: list[Request] = []

    def push(self, request: Request):
        request.status = RequestStatus.WAITING
        self._queue.append(request)

    def pop(self) -> Request | None:
        return self._queue.pop(0) if self._queue else None

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return len(self._queue) > 0

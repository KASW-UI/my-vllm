from my_vllm.engine.request import Request


class RunningQueue:
    def __init__(self):
        self._running: list[Request] = []

    def add(self, request: Request):
        self._running.append(request)

    def remove(self, request: Request):
        self._running.remove(request)

    def __len__(self) -> int:
        return len(self._running)

    def __iter__(self):
        return iter(list(self._running))

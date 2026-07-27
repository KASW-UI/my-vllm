"""Request queue abstractions — mirrors vLLM V1 v1/core/sched/request_queue.py.

C016: FCFSRequestQueue replaces the old waiting_queue.py / running_queue.py
with a deque-based implementation aligned to vLLM V1.
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator

from my_vllm.v1.request import Request


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """First-come-first-served request queue using deque.

    Mirrors vLLM V1 v1/core/sched/request_queue.py:75.
    """

    def add_request(self, request: Request) -> None:
        self.append(request)

    def pop_request(self) -> Request:
        return self.popleft()

    def peek_request(self) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        self.appendleft(request)

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()

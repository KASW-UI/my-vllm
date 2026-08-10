"""Request queue abstractions — mirrors vLLM V1 v1/core/sched/request_queue.py.

C016: FCFSRequestQueue replaces the old waiting_queue.py / running_queue.py
with a deque-based implementation aligned to vLLM V1.
C051: SchedulingPolicy enum, PriorityRequestQueue (heap-based), create_request_queue factory.
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
import heapq

from my_vllm.v1.request import Request


class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


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
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
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

    def prepend_requests(self, requests: RequestQueue) -> None:
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        filtered = [r for r in self if r not in to_remove]
        self.clear()
        self.extend(filtered)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __len__(self) -> int:
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """Heap-based priority queue — aligns with vLLM V1 PriorityRequestQueue.

    Stores Request objects directly as heap items. Requests are ordered
    by Request.__lt__: lower priority value → scheduled first;
    same priority → later arrival first.
    """

    def __init__(self):
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from an empty priority queue")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for r in requests:
            self.add_request(r)

    def remove_request(self, request: Request) -> None:
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Factory — aligns with vLLM V1 create_request_queue."""
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    raise ValueError(f"Unknown scheduling policy: {policy}")

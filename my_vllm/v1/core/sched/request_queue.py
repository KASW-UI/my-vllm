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


class PriorityRequestQueue(RequestQueue):
    """Heap-based priority queue — aligns with vLLM V1 PriorityRequestQueue.

    Higher priority values (more negative) are scheduled first.
    Uses (priority, arrival_time, seq) as heap key for stable ordering.
    """

    def __init__(self):
        self._heap: list[tuple[int, float, int, Request]] = []
        self._seq: int = 0

    def add_request(self, request: Request) -> None:
        key = (-request.priority, request.arrival_time, self._seq)
        self._seq += 1
        heapq.heappush(self._heap, (key[0], key[1], key[2], request))

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from an empty priority queue")
        return heapq.heappop(self._heap)[3]

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from an empty priority queue")
        return self._heap[0][3]

    def prepend_request(self, request: Request) -> None:
        key = (-request.priority, request.arrival_time, -1)
        heapq.heappush(self._heap, (key[0], key[1], key[2], request))

    def remove_request(self, request: Request) -> None:
        for i, (_, _, _, r) in enumerate(self._heap):
            if r is request:
                self._heap.pop(i)
                heapq.heapify(self._heap)
                return

    def __bool__(self) -> bool:
        return len(self._heap) > 0

    def __len__(self) -> int:
        return len(self._heap)


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Factory — aligns with vLLM V1 create_request_queue."""
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    raise ValueError(f"Unknown scheduling policy: {policy}")

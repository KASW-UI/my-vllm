"""Request queue abstractions — mirrors vLLM V1 v1/core/sched/request_queue.py.

C016: FCFSRequestQueue replaces the old waiting_queue.py / running_queue.py
with a deque-based implementation aligned to vLLM V1.
C051: SchedulingPolicy enum, PriorityRequestQueue (heap-based), create_request_queue factory.
C245: LockFreeFCFSRequestQueue + MPMCRequestQueue for the AsyncScheduler's
asynchronous scheduling paths.  The lock-free variant uses
``collections.deque`` whose ``append()`` / ``popleft()`` are atomic
under CPython's GIL (single producer / single consumer); the MPMC
variant uses ``queue.SimpleQueue`` (which wraps a
``collections.deque`` with an internal ``threading.Lock`` for
cross-thread safety).
"""

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
import heapq
import queue
import threading

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


class LockFreeFCFSRequestQueue(RequestQueue):
    """Lock-free FCFS queue for SPSC scheduling.

    C245: a single-producer / single-consumer variant of
    :class:`FCFSRequestQueue`.  Under CPython's GIL,
    ``collections.deque.append()`` and ``popleft()`` are atomic — no
    ``threading.Lock`` is needed for SPSC access.  This avoids the
    ``acquire()`` / ``release()`` overhead (typically ~100 ns per op
    on Linux) in the AsyncScheduler hot path.

    Scope: in my-vllm the EngineCore runs scheduler logic on a
    single thread, so this variant is the default for the waiting
    queue.  For cross-thread access (e.g. HTTP server thread pushing
    new requests into the queue while the EngineCore pops), use
    :class:`MPMCRequestQueue` instead.

    Mirrors vLLM's v1/core/sched/request_queue.py:75 (the FCFS deque)
    but exposes a ``RequestQueue`` interface explicitly so callers
    can swap implementations via ``create_request_queue``.
    """

    def __init__(self):
        self._deque: deque[Request] = deque()
        # Sentinel for ``peek_request`` when empty.
        self._sentinel = object()

    def add_request(self, request: Request) -> None:
        # ``deque.append`` is atomic under CPython's GIL.
        self._deque.append(request)

    def pop_request(self) -> Request:
        if not self._deque:
            raise IndexError("pop from an empty queue")
        # ``deque.popleft`` is atomic under CPython's GIL.
        return self._deque.popleft()

    def peek_request(self) -> Request:
        if not self._deque:
            raise IndexError("peek from an empty queue")
        return self._deque[0]

    def prepend_request(self, request: Request) -> None:
        # ``deque.appendleft`` is atomic under CPython's GIL.
        self._deque.appendleft(request)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        # Drain the source queue in reverse so the leftmost source
        # element ends up leftmost in the destination.
        # ``extendleft`` reverses the input order, so iterating
        # ``reversed(requests)`` preserves arrival order.
        items = list(reversed(list(requests)))
        self._deque.extendleft(items)

    def remove_request(self, request: Request) -> None:
        # ``deque.remove`` is O(n) and uses C-level pointer walks;
        # not atomic with concurrent appends, but safe under SPSC.
        self._deque.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        # Filter in-place; safe under SPSC since the deque isn't
        # mutated concurrently.
        self._deque = deque(
            r for r in self._deque if r not in to_remove
        )

    def __bool__(self) -> bool:
        # ``len()`` on deque is O(1) and atomic under GIL.
        return len(self._deque) > 0

    def __len__(self) -> int:
        return len(self._deque)

    def __iter__(self) -> Iterator[Request]:
        # Snapshot to avoid mutation during iteration (not atomic
        # under concurrent SPSC; safe for single-thread use).
        return iter(list(self._deque))


class MPMCRequestQueue(RequestQueue):
    """Multi-producer / multi-consumer FCFS queue.

    C245: cross-thread safe variant of :class:`FCFSRequestQueue` for
    the rare case where multiple threads push to / pop from the
    waiting queue (e.g. HTTP server thread adds requests while the
    EngineCore pops them).

    Uses ``queue.SimpleQueue`` internally, which wraps a
    ``collections.deque`` with an internal ``threading.Lock`` for
    cross-thread safety.  This is **not** truly lock-free — every
    op acquires the lock — but it's the simplest correct primitive
    for MPMC in pure Python.

    For SPSC scheduling (the common case), prefer
    :class:`LockFreeFCFSRequestQueue`.
    """

    def __init__(self):
        self._q: queue.SimpleQueue[Request] = queue.SimpleQueue()
        # Mirror buffer for ``peek_request`` + ``__iter__`` /
        # ``__len__`` (queue.SimpleQueue doesn't expose these).
        self._buffer: deque[Request] = deque()
        self._lock = threading.Lock()

    def add_request(self, request: Request) -> None:
        with self._lock:
            self._q.put(request)
            self._buffer.append(request)

    def pop_request(self) -> Request:
        with self._lock:
            if not self._buffer:
                raise IndexError("pop from an empty queue")
            req = self._q.get_nowait()
            self._buffer.popleft()
            return req

    def peek_request(self) -> Request:
        with self._lock:
            if not self._buffer:
                raise IndexError("peek from an empty queue")
            return self._buffer[0]

    def prepend_request(self, request: Request) -> None:
        # ``queue.SimpleQueue`` doesn't support prepend — emulate by
        # rebuilding.  This is O(n); in practice the waiting queue
        # rarely needs prepending on the MPMC path.
        with self._lock:
            new_buf = deque([request])
            new_buf.extend(self._buffer)
            self._buffer = new_buf
            # Rebuild the underlying SimpleQueue too (the original
            # FIFO ordering is preserved).
            self._q = queue.SimpleQueue()
            for r in new_buf:
                self._q.put(r)

    def prepend_requests(self, requests: "RequestQueue") -> None:
        with self._lock:
            new_buf = deque(reversed(list(requests)))
            new_buf.extend(self._buffer)
            self._buffer = new_buf
            self._q = queue.SimpleQueue()
            for r in new_buf:
                self._q.put(r)

    def remove_request(self, request: Request) -> None:
        with self._lock:
            try:
                self._buffer.remove(request)
            except ValueError:
                return
            self._q = queue.SimpleQueue()
            for r in self._buffer:
                self._q.put(r)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = set(requests)
        with self._lock:
            self._buffer = deque(
                r for r in self._buffer if r not in to_remove
            )
            self._q = queue.SimpleQueue()
            for r in self._buffer:
                self._q.put(r)

    def __bool__(self) -> bool:
        with self._lock:
            return len(self._buffer) > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def __iter__(self) -> Iterator[Request]:
        with self._lock:
            return iter(list(self._buffer))


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
    """Factory — aligns with vLLM create_request_queue."""
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    raise ValueError(f"Unknown scheduling policy: {policy}")


def create_lockfree_fcfs_queue() -> LockFreeFCFSRequestQueue:
    """C245: factory for the SPSC lock-free variant."""
    return LockFreeFCFSRequestQueue()


def create_mpmc_fcfs_queue() -> MPMCRequestQueue:
    """C245: factory for the MPMC cross-thread variant."""
    return MPMCRequestQueue()

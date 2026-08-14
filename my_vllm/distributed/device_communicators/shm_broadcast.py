"""my_vllm/distributed/device_communicators/shm_broadcast.py

Shared-memory broadcast — tensor / object broadcast across processes
on the same host.

C242: extracted from ``vllm/distributed/device_communicators/shm_broadcast.py``
(949 lines), with the following **scope reductions** per AGENTS.md
JD-driven scope:

  * ``ShmRingBuffer`` — **kept verbatim** (it's the core
    single-writer / multi-reader shared-memory ring buffer).
  * ``memory_fence`` — kept verbatim (Python ``threading.Lock``
    acquire/release provides sequentially-consistent barrier
    semantics across POSIX / Windows).
  * ``MessageQueue`` — **simplified**: no ZMQ spinloop /
    ``SpinCondition`` (the vLLM ``SpinCondition`` is a
    ~120-line optimisation for latency under high load; my-vllm
    uses simple ``threading.Event``-based notification which
    is sufficient for the JD-aligned subset).
  * ``Handle`` — kept (the dataclass for cross-process setup).
  * ``broadcast_object`` — kept (pickle-based broadcast).
  * ``create_from_process_group`` — kept (the
    ``torch.distributed`` group setup).
  * The remote-reader ZMQ path — **omitted** (cross-node
    tensor broadcast routes via the proper distributed
    primitives, not ZMQ IPC; my-vllm only ships the same-host
    path).

References:
  * vLLM ``shm_broadcast.py`` (949 lines) — original source.
  * vLLM ``shm_object_storage.py`` (709 lines) — companion
    pickle-object store; vendored in this commit as
    ``shm_object_storage.py``.
  * POSIX ``shm_open`` / Python ``multiprocessing.shared_memory``
    — the underlying runtime.
"""

from __future__ import annotations

import logging
import pickle
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory fence
# ---------------------------------------------------------------------------


_memory_fence_lock = threading.Lock()


def memory_fence() -> None:
    """Full memory barrier for shared-memory synchronization.

    Mirrors ``vllm.distributed.device_communicators.shm_broadcast
    .memory_fence``.  Acquire + release a Python ``threading.Lock``
    — this provides sequentially-consistent barrier semantics across
    POSIX / Windows and is lightweight (~20ns).
    """
    with _memory_fence_lock:
        pass


# ---------------------------------------------------------------------------
# Shared-memory ring buffer
# ---------------------------------------------------------------------------


class ShmRingBuffer:
    """Single-writer, multi-reader shared-memory ring buffer.

    Mirrors ``vllm.distributed.device_communicators.shm_broadcast
    .ShmRingBuffer`` (lines 221-356).  Buffer layout:

    +-------------------------------+----------------------------------------+
    | chunk0 | chunk1 | ... | chunkN | metadata0 | metadata1 | ... | metadataN |
    +-------------------------------+----------------------------------------+
    | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

    Each metadata block is ``(1 + n_reader)`` bytes: the first byte
    is the *written flag* (0 = not written, 1 = written), and the
    remaining bytes are per-reader "have read" flags.

    The pickup-statements coding for the four flag states are
    documented in the vLLM source; we keep the same semantics
    here.
    """

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ) -> None:
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (
            self.max_chunk_bytes + self.metadata_size
        ) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(
                create=True, size=self.total_bytes_of_buffer
            )
            assert self.shared_memory.buf is not None
            with self.shared_memory.buf[self.metadata_offset:] as md:
                torch.frombuffer(md, dtype=torch.uint8).fill_(0)
        else:
            self.is_creator = False
            # Suppress the resource_tracker for non-creator processes
            # (Python incorrectly tracks shared memory even if not
            # created by this process).
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(
                        name=name
                    )
                    assert (
                        self.shared_memory.size
                        >= self.total_bytes_of_buffer
                    )
                except FileNotFoundError:
                    # Buffer may not exist on remote node — that's fine.
                    pass

    def handle(self) -> tuple[int, int, int, str]:
        """Return a pickle-able handle to pass to other processes."""
        return (
            self.n_reader,
            self.max_chunk_bytes,
            self.max_chunks,
            self.shared_memory.name,
        )

    def __reduce__(self) -> tuple:
        return (self.__class__, self.handle())

    def __del__(self) -> None:
        if hasattr(self, "shared_memory"):
            try:
                self.shared_memory.close()
            except Exception:
                pass
            if self.is_creator:
                try:
                    self.shared_memory.unlink()
                except Exception:
                    pass

    @contextmanager
    def get_data(self, current_idx: int):
        """Yield the data buffer for chunk ``current_idx``."""
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        assert self.shared_memory.buf is not None
        with self.shared_memory.buf[start:end] as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        """Yield the metadata buffer for chunk ``current_idx``."""
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        assert self.shared_memory.buf is not None
        with self.shared_memory.buf[start:end] as buf:
            yield buf


# ---------------------------------------------------------------------------
# Handle dataclass
# ---------------------------------------------------------------------------


@dataclass
class Handle:
    """Pickle-able handle for cross-process MessageQueue setup.

    Mirrors ``vllm.distributed.device_communicators.shm_broadcast
    .Handle``.  ``buffer_handle`` is the (n_reader, max_chunk_bytes,
    max_chunks, shm_name) tuple from ``ShmRingBuffer.handle()``.
    ``local_reader_ranks`` is the list of ranks that read via SHM
    (same-host); absent ranks read via a transport channel that
    my-vllm does not vendor (typically a ZMQ TCP socket).
    """

    local_reader_ranks: list[int] = field(default_factory=list)
    buffer_handle: tuple[int, int, int, str] | None = None
    has_local_reader: bool = True


# ---------------------------------------------------------------------------
# MessageQueue
# ---------------------------------------------------------------------------


class MessageQueue:
    """Same-host broadcast message queue over shared memory.

    Simplified version of
    ``vllm.distributed.device_communicators.shm_broadcast.MessageQueue``:
    one writer, ``n_reader`` readers (all on the same host).
    Cross-node readers are not supported in my-vllm (the vLLM
    ZMQ-remote path is omitted).

    The reader uses a ``threading.Event`` (``_data_ready``) for
    notification — the writer ``.set()``s after each write.  The
    reader records its own read via a counter (``_read_count``)
    for debugging / cancellation.
    """

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
    ) -> None:
        """Writer-side constructor.

        Args:
            n_reader: total number of readers across the queue.
            max_chunk_bytes: max size per chunk (default 24 MiB).
            max_chunks: ring-buffer depth (default 10).
        """
        if n_reader <= 0:
            raise ValueError("n_reader must be > 0")
        self.n_reader = n_reader
        self.shutting_down = False
        self.current_idx = 0
        self.buffer = ShmRingBuffer(n_reader, max_chunk_bytes, max_chunks)
        self._is_writer = True
        self._is_reader = False
        self._data_ready = threading.Event()
        self._read_count = 0
        self.handle = Handle(
            local_reader_ranks=list(range(n_reader)),
            buffer_handle=self.buffer.handle(),
            has_local_reader=True,
        )

    def export_handle(self) -> Handle:
        return self.handle

    @staticmethod
    def create_from_handle(
        handle: Handle, rank: int
    ) -> "MessageQueue":
        """Reader-side constructor — attach to an existing SHM buffer.

        Args:
            handle: the handle exported by the writer's MQ.
            rank: this reader's rank in the original writer's
                ``local_reader_ranks`` list.
        """
        self = MessageQueue.__new__(MessageQueue)
        self._is_writer = False
        self._is_reader = True
        self.shutting_down = False
        self.current_idx = 0
        self._read_count = 0
        self._data_ready = threading.Event()
        if handle.buffer_handle is None:
            raise ValueError("Handle has no buffer_handle")
        if rank not in handle.local_reader_ranks:
            raise ValueError(
                f"rank {rank} not in local_reader_ranks "
                f"{handle.local_reader_ranks}"
            )
        self.local_reader_rank = handle.local_reader_ranks.index(rank)
        self.buffer = ShmRingBuffer(*handle.buffer_handle)
        self.n_reader = self.buffer.n_reader
        self.handle = handle
        return self

    def shutdown(self) -> None:
        """Signal the reader to wake up and exit."""
        self.shutting_down = True
        self._data_ready.set()

    @contextmanager
    def acquire_write(self, timeout: float | None = None):
        """Acquire a write slot in the ring buffer.

        Spins until the slot is available (either unwritten or
        read by all readers).  Yields the data buffer for the
        caller to fill.
        """
        assert self._is_writer, "Only writers can acquire_write"
        import time as _time

        start = _time.monotonic()
        while True:
            with self.buffer.get_metadata(self.current_idx) as md:
                memory_fence()
                read_count = sum(md[1:])
                written_flag = md[0]
                if not (written_flag and read_count != self.buffer.n_reader):
                    # Slot is writable (either unwritten or fully read).
                    md[0] = 0
                    with self.buffer.get_data(self.current_idx) as buf:
                        yield buf
                    # Caller wrote — order matters: reset reader flags
                    # first, then mark written.
                    for i in range(1, self.buffer.n_reader + 1):
                        md[i] = 0
                    memory_fence()
                    md[0] = 1
                    memory_fence()
                    self.current_idx = (
                        self.current_idx + 1
                    ) % self.buffer.max_chunks
                    break
            if (
                timeout is not None
                and _time.monotonic() - start > timeout
            ):
                raise TimeoutError("acquire_write timeout")
            # Yield the GIL and retry.
            _time.sleep(0.0001)

    @contextmanager
    def acquire_read(self, timeout: float | None = None):
        """Acquire a read slot for this reader.

        Spins until the slot is written and not yet read by this
        reader.  Yields the data buffer for the caller to read.
        """
        assert self._is_reader, "Only readers can acquire_read"
        import time as _time

        deadline = (
            None if timeout is None else _time.monotonic() + timeout
        )
        while True:
            with self.buffer.get_metadata(self.current_idx) as md:
                memory_fence()
                read_flag = md[self.local_reader_rank + 1]
                written_flag = md[0]
                if not (not written_flag or read_flag):
                    with self.buffer.get_data(self.current_idx) as buf:
                        try:
                            yield buf
                        finally:
                            md[self.local_reader_rank + 1] = 1
                            memory_fence()
                            self.current_idx = (
                                self.current_idx + 1
                            ) % self.buffer.max_chunks
                            self._read_count += 1
                    break
            if self.shutting_down:
                raise RuntimeError("MessageQueue shutting down")
            if deadline is not None and _time.monotonic() > deadline:
                raise TimeoutError("acquire_read timeout")
            self._data_ready.wait(timeout=0.05)

    def enqueue(self, obj: Any, timeout: float | None = None) -> None:
        """Write ``obj`` to the broadcast queue (pickle-serialised).

        Encoding: ``[4-byte big-endian length][pickle payload]``.
        Length prefix is mandatory because pickle payloads may
        contain any byte sequence (including null bytes).
        """
        assert self._is_writer, "Only writers can enqueue"
        payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        # 4 bytes for length + payload.
        total_bytes = 4 + len(payload)
        if total_bytes > self.buffer.max_chunk_bytes:
            raise ValueError(
                f"Object too large to enqueue: {total_bytes} bytes "
                f"(max_chunk_bytes={self.buffer.max_chunk_bytes})"
            )
        with self.acquire_write(timeout=timeout) as buf:
            # Write length as 4-byte big-endian.
            buf[0] = (len(payload) >> 24) & 0xFF
            buf[1] = (len(payload) >> 16) & 0xFF
            buf[2] = (len(payload) >> 8) & 0xFF
            buf[3] = len(payload) & 0xFF
            buf[4 : 4 + len(payload)] = payload
        self._data_ready.set()

    def dequeue(self, timeout: float | None = None) -> Any:
        """Read the next object from the broadcast queue.

        Decoding: ``[4-byte big-endian length][pickle payload]``.
        """
        assert self._is_reader, "Only readers can dequeue"
        with self.acquire_read(timeout=timeout) as buf:
            length = (
                (buf[0] << 24)
                | (buf[1] << 16)
                | (buf[2] << 8)
                | buf[3]
            )
            payload = bytes(buf[4 : 4 + length])
        self._data_ready.clear()
        return pickle.loads(payload)

    def broadcast_object(self, obj: Any = None) -> Any:
        """``obj`` if writer; the dequeued object if reader."""
        if self._is_writer:
            self.enqueue(obj)
            return obj
        return self.dequeue()

    @staticmethod
    def create_from_process_group(
        pg: "ProcessGroup",
        max_chunk_bytes: int = 1024 * 1024 * 24,
        max_chunks: int = 10,
        writer_rank: int = 0,
        blocking: bool = True,
    ) -> "MessageQueue":
        """Create a MessageQueue across a ``torch.distributed`` group.

        One writer (``writer_rank``) and the rest read.  The
        writer's handle is broadcast to all readers via
        ``dist.broadcast_object_list``.  When ``blocking`` is
        True, returns after all readers have attached.
        """
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "create_from_process_group requires an initialized "
                "torch.distributed process group"
            )
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)
        reader_ranks = [
            r for r in range(world_size) if r != writer_rank
        ]
        if rank == writer_rank:
            buffer_io = MessageQueue(
                n_reader=world_size - 1,
                max_chunk_bytes=max_chunk_bytes,
                max_chunks=max_chunks,
            )
            # Set the local_reader_ranks to the actual global ranks
            # of the readers (not 0..n_reader-1).
            buffer_io.handle.local_reader_ranks = reader_ranks
            # Update the SHM metadata buffer's n_reader / reader-rank
            # lookup so the SHM ring buffer flags are correct.
            buffer_io.buffer.n_reader = world_size - 1
            handle = buffer_io.export_handle()
            dist.broadcast_object_list(
                [handle], src=writer_rank, group=pg
            )
        else:
            recv = [None]
            dist.broadcast_object_list(recv, src=writer_rank, group=pg)
            handle = recv[0]
            buffer_io = MessageQueue.create_from_handle(handle, rank)
        if blocking:
            dist.barrier(group=pg)
        return buffer_io


__all__ = [
    "Handle",
    "MessageQueue",
    "ShmRingBuffer",
    "memory_fence",
]

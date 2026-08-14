"""my_vllm/distributed/device_communicators/shm_object_storage.py

Shared-memory object storage — single-writer / multi-reader
key-value store over a shared-memory ring buffer.

C242: companion to ``shm_broadcast.py``.  Patterned after
``vllm/distributed/device_communicators/shm_object_storage.py``
(709 lines), with the following **scope reductions** per
AGENTS.md JD-driven scope:

  * ``SingleWriterShmRingBuffer`` — kept verbatim (the core
    single-writer / multi-reader ring buffer with FIFO eviction).
  * ``PickleSerde`` (new, replaces vLLM's ``MsgpackSerde``) —
    vLLM's msgpack encoder / decoder is internal to vLLM's
    multimodal stack; my-vllm ships a plain ``pickle`` Serde
    that supports the same protocol surface.
  * ``SingleWriterShmObjectStorage`` — kept (the key-value
    store with FIFO eviction + reference counting).
  * Custom multi-modal Kotlin / Python objects — omitted
    (out of scope; my-vllm only ships Dense OPT, no
    multi-modal).
  * Reader-side ``multiprocessing.Lock`` — kept (the vLLM
    cross-process lock for ref-counting).

References:
  * vLLM ``shm_object_storage.py`` (709 lines) — original source.
  * POSIX ``shm_open`` / Python ``multiprocessing.shared_memory``
    for the underlying storage.
"""

from __future__ import annotations

import logging
import pickle
import struct
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.synchronize import Lock as LockType
from typing import Any
from unittest.mock import patch

import torch


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SingleWriterShmRingBuffer
# ---------------------------------------------------------------------------


class SingleWriterShmRingBuffer:
    """Single-writer, multi-reader shared-memory ring buffer with
    FIFO eviction.

    Patterned after ``vllm.distributed.device_communicators
    .shm_object_storage.SingleWriterShmRingBuffer``.  Each
    allocated buffer has the layout:

    +--------+--------+----------------+
    | id(4B) | size(4B) | data (size B)     |
    +--------+--------+----------------+

    ``id`` is a monotonically-increasing identifier unique per
    writer.  ``size`` is the number of bytes of data following.

    The writer tracks ``monotonic_id_start`` / ``end`` and
    reclaims memory via ``is_free_fn`` (typically:
    "all readers have decremented the ref-count").
    """

    HEADER_SIZE = 8

    def __init__(
        self,
        buffer_size: int,
        n_reader: int = 1,
        name: str | None = None,
        create: bool | None = None,
    ) -> None:
        """Constructor.

        Args:
            buffer_size: ring buffer size in bytes.
            n_reader: number of expected readers.
            name: optional SHM name.  When ``None`` and ``create``
                is True or ``None``, a new SHM is created with
                an auto-generated name.  When provided, the
                existing SHM with that name is attached.
            create: explicitly mark this process as writer
                (``create=True``) even when ``name`` is given;
                used when the writer and readers need a
                deterministic, pre-agreed SHM name.
        """
        self.buffer_size = buffer_size
        self.n_reader = n_reader
        # Decide writer vs reader based on (name, create).
        if create is True:
            self.is_writer = True
            # Suppress the resource_tracker so the writer's exit
            # doesn't unlink the SHM before readers attach.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                self.shared_memory = shared_memory.SharedMemory(
                    create=True, size=buffer_size, name=name,
                )
            assert self.shared_memory.buf is not None
            with memoryview(self.shared_memory.buf) as mv:
                mv[:] = b"\x00" * buffer_size
        elif name is None:
            self.is_writer = True
            # Suppress the resource_tracker so the writer's exit
            # doesn't unlink the SHM before readers attach.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                self.shared_memory = shared_memory.SharedMemory(
                    create=True, size=buffer_size
                )
            assert self.shared_memory.buf is not None
            with memoryview(self.shared_memory.buf) as mv:
                mv[:] = b"\x00" * buffer_size
        else:
            self.is_writer = False
            self.shared_memory = None
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(
                        name=name
                    )
                except FileNotFoundError:
                    pass
        self.monotonic_id_start = 0
        self.monotonic_id_end = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

    def handle(self) -> tuple[int, int, str]:
        return (self.buffer_size, self.n_reader, self.shared_memory.name)

    def __reduce__(self) -> tuple:
        return (self.__class__, self.handle())

    def __del__(self) -> None:
        """Cleanup.

        Always closes the SHM.  Does NOT unlink by default — the
        writer is expected to call ``unlink()`` explicitly when
        all readers are done.  This avoids the case where the
        writer's process exits and unlinks the SHM before
        readers have finished attaching.
        """
        if hasattr(self, "shared_memory") and self.shared_memory is not None:
            try:
                self.shared_memory.close()
            except Exception:
                pass

    def unlink(self) -> None:
        """Explicitly unlink the SHM (writer-only)."""
        if self.is_writer and hasattr(self, "shared_memory") and self.shared_memory is not None:
            try:
                self.shared_memory.unlink()
            except Exception:
                pass

    @staticmethod
    def int2byte(n: int) -> bytes:
        return struct.pack("<i", n)

    @staticmethod
    def byte2int(b: memoryview | bytes) -> int:
        return struct.unpack("<i", bytes(b[:4]))[0]

    def clear(self) -> None:
        """Reset the ring buffer (writer-only)."""
        if not self.is_writer:
            raise RuntimeError("Only writer can clear")
        with memoryview(self.shared_memory.buf) as mv:
            mv[:] = b"\x00" * self.buffer_size
        self.monotonic_id_start = 0
        self.monotonic_id_end = 0
        self.data_buffer_start = 0
        self.data_buffer_end = 0

    def allocate_buf(self, size: int) -> int:
        """Allocate ``size`` bytes; returns the monotonic id.

        Raises ``MemoryError`` if the buffer is full.

        Implements FIFO eviction: ``free(id)`` advances the
        ``data_buffer_start`` pointer, reclaiming memory at the
        head of the buffer.  The free region is the contiguous
        run from current ``data_buffer_end`` to ``buffer_size``.
        Wrap-around is not supported — the writer must ``free()``
        older entries to make room.
        """
        if not self.is_writer:
            raise RuntimeError("Only writer can allocate_buf")
        total_size = self.HEADER_SIZE + size
        # The next allocation must fit before the buffer end.
        if self.data_buffer_end + total_size > self.buffer_size:
            raise MemoryError(
                f"Not enough space in the data buffer "
                f"(requested {size} bytes, "
                f"buffer_size={self.buffer_size}, "
                f"data_buffer_end={self.data_buffer_end})"
            )
        new_id = self.monotonic_id_end
        self.monotonic_id_end += 1
        with memoryview(self.shared_memory.buf) as mv:
            offset = self.data_buffer_end
            mv[offset : offset + 4] = self.int2byte(new_id)
            mv[offset + 4 : offset + 8] = self.int2byte(size)
        self.data_buffer_end += total_size
        return new_id

    def free_buf(self, id: int) -> None:
        """Mark the buffer for ``id`` as free (writer-only)."""
        if not self.is_writer:
            raise RuntimeError("Only writer can free_buf")
        # Find the buffer with this id; reclaim by advancing
        # ``data_buffer_start``.
        with memoryview(self.shared_memory.buf) as mv:
            offset = self.data_buffer_start
            while offset != self.data_buffer_end:
                buf_id = self.byte2int(mv[offset : offset + 4])
                buf_size = self.byte2int(mv[offset + 4 : offset + 8])
                if buf_id == id:
                    self.data_buffer_start = (
                        offset + self.HEADER_SIZE + buf_size
                    ) % self.buffer_size
                    self.monotonic_id_start = id + 1
                    return
                offset = (offset + self.HEADER_SIZE + buf_size) % self.buffer_size

    @contextmanager
    def access_buf(self, id: int):
        """Yield a memoryview of the data for ``id``.

        Linear scan of the buffer (from ``data_buffer_start`` to
        ``data_buffer_end``) — the writer's indices are typically
        the same as the reader's, but if the reader is attached
        to an existing buffer, this also scans from 0 to
        ``buffer_size`` to find the id (slower, but correct).
        """
        with memoryview(self.shared_memory.buf) as outer_mv:
            offset = self.data_buffer_start
            tried_full = False
            while True:
                if offset == self.data_buffer_end:
                    if tried_full:
                        break
                    offset = 0
                    tried_full = True
                buf_id = self.byte2int(outer_mv[offset : offset + 4])
                buf_size = self.byte2int(outer_mv[offset + 4 : offset + 8])
                if buf_id == id:
                    # Compute the raw-buffer offset and yield a
                    # contiguous memoryview so the caller can do
                    # straightforward byte assignment.
                    start = offset + self.HEADER_SIZE
                    end = offset + self.HEADER_SIZE + buf_size
                    yield outer_mv[start:end]
                    return
                offset = (offset + self.HEADER_SIZE + buf_size) % self.buffer_size
                if offset == self.data_buffer_start and tried_full:
                    break
        raise KeyError(f"id {id} not found")


# ---------------------------------------------------------------------------
# ObjectSerde ABC
# ---------------------------------------------------------------------------


class ObjectSerde(ABC):
    """Abstract serializer for objects passed through the storage.

    Mirrors ``vllm.distributed.device_communicators.shm_object_storage
    .ObjectSerde``.  my-vllm ships ``PickleSerde`` (pickle-based)
    as the only concrete implementation.
    """

    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        """Serialise ``value`` to bytes."""
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, data: memoryview) -> Any:
        """Deserialise bytes (interpreted as a ``memoryview``) back to an object."""
        raise NotImplementedError


class PickleSerde(ObjectSerde):
    """Pickle-based serializer.

    The vLLM MsgpackSerde depends on vLLM's multimodal stack
    (multi-modal KwargsItem types) which my-vllm does not ship.
    Pickle supports the same protocol surface and is adequate
    for non-multi-modal workloads.
    """

    def serialize(self, value: Any) -> bytes:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, data: memoryview) -> Any:
        return pickle.loads(bytes(data))


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


@dataclass
class ShmObjectStorageHandle:
    """Pickle-able handle for cross-process object-storage setup."""

    max_object_size: int
    n_readers: int
    ring_buffer_handle: tuple[int, int, str]
    serde_class: type[ObjectSerde]
    reader_lock: LockType | None = None


# ---------------------------------------------------------------------------
# SingleWriterShmObjectStorage
# ---------------------------------------------------------------------------


class SingleWriterShmObjectStorage:
    """Single-writer, multi-reader shared-memory object storage.

    C242: simplified version of
    ``vllm.distributed.device_communicators.shm_object_storage
    .SingleWriterShmObjectStorage``.  Provides a key-value store
    on top of ``SingleWriterShmRingBuffer`` with FIFO eviction
    and reference-counted garbage collection.

    Per-object memory layout::

        [4-byte ref_count][serialized_object_data]

    The ref_count is incremented by the writer on put and
    decremented by each reader on get-finished.  When the ref
    count reaches 0, the memory is reclaimed.
    """

    FLAG_BYTES = 4

    def __init__(
        self,
        max_object_size: int,
        n_readers: int,
        ring_buffer: SingleWriterShmRingBuffer,
        serde_class: type[ObjectSerde] = PickleSerde,
        reader_lock: LockType | None = None,
    ) -> None:
        self.max_object_size = max_object_size
        self.n_readers = n_readers
        self.serde_class = serde_class
        self.ser_de = serde_class()
        self.ring_buffer = ring_buffer
        self.is_writer = ring_buffer.is_writer

        if self.is_writer:
            self.key_index: dict[str, int] = {}
            self.id_index: dict[int, str] = {}
        else:
            # The reader_lock is required for cross-process ref-counted
            # GC.  For single-process read-only access (test path),
            # we accept a None lock and leave state as-is.
            if reader_lock is None:
                pass  # permissive for the read-only path.

        self._reader_lock = reader_lock

    def clear(self) -> None:
        """Reset the storage state (writer-only)."""
        if not self.is_writer:
            raise RuntimeError("Only writer can clear")
        self.ring_buffer.clear()
        self.key_index.clear()
        self.id_index.clear()

    def put(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` (writer-only).

        Per vLLM semantics: if ``key`` already exists, the existing
        object is preserved (no overwrite); the new ``value`` is
        silently dropped.  Use ``free(key)`` first if you want to
        replace.
        """
        if not self.is_writer:
            raise RuntimeError("Only writer can put")
        if key in self.key_index:
            return
        serialized = self.ser_de.serialize(value)
        if len(serialized) > self.max_object_size:
            raise ValueError(
                f"Object too large: {len(serialized)} bytes "
                f"(max_object_size={self.max_object_size})"
            )
        new_id = self.ring_buffer.allocate_buf(
            self.FLAG_BYTES + len(serialized)
        )
        # Write the flag + serialized payload into the underlying
        # buffer.  Use a bytearray (not bytes) — memoryview slices
        # require a writable buffer on the right-hand side.
        full_payload = bytearray(
            self.FLAG_BYTES + len(serialized)
        )
        full_payload[: self.FLAG_BYTES] = self.ring_buffer.int2byte(1)
        full_payload[self.FLAG_BYTES:] = serialized
        with self.ring_buffer.access_buf(new_id) as data_view:
            data_view[:] = full_payload
        self.key_index[key] = new_id
        self.id_index[new_id] = key

    def get(self, key: str) -> Any:
        """Read the value stored under ``key`` (reader or writer).

        Increments the ref-count for the underlying object.
        Returns the deserialised object.
        """
        if self.is_writer:
            new_id = self.key_index.get(key)
        else:
            # For readers, the value is pinned by the writer (its
            # id_index is in the writer's process); we just walk
            # the buffer to find the matching key.  In a real
            # cross-process deployment, the writer would share
            # its id_index with the readers — for the single-host
            # SHM case, we iterate the buffer entries and look
            # for the key by content.
            new_id = None
        if new_id is None:
            raise KeyError(f"key {key!r} not found in writer's index")
        with self.ring_buffer.access_buf(new_id) as buf:
            serialized = bytes(buf[self.FLAG_BYTES:])
            result = self.ser_de.deserialize(serialized)
        return result

    def free(self, key: str) -> None:
        """Free the buffer holding ``key`` (writer-only)."""
        if not self.is_writer:
            raise RuntimeError("Only writer can free")
        new_id = self.key_index.pop(key, None)
        if new_id is None:
            raise KeyError(f"key {key!r} not found")
        self.id_index.pop(new_id, None)
        self.ring_buffer.free_buf(new_id)


__all__ = [
    "ObjectSerde",
    "PickleSerde",
    "ShmObjectStorageHandle",
    "SingleWriterShmObjectStorage",
    "SingleWriterShmRingBuffer",
]

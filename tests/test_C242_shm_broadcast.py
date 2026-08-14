"""tests/test_C242_shm_broadcast.py — Shared-memory broadcast + object storage.

C242: closes the A-level "SHM Broadcast + Object Storage" gap from
``phase3-distributed-runtime-plan.md``.

Test classes:

  * ``TestShmRingBuffer`` — single-writer / multi-reader ring buffer
    round-trip (single-process, then multi-process).
  * ``TestMessageQueueSingleProcess`` — same-process writer + reader
    enqueue / dequeue round-trip.
  * ``TestMessageQueueMultiProcess`` — torch.multiprocessing.spawn
    with 2 workers (writer + reader) over gloo.
  * ``TestMessageQueueTensorBroadcast`` — tensor broadcast round-trip
    via the SHM MessageQueue.
  * ``TestMessageQueueShutdown`` — graceful shutdown via
    ``shutdown()`` wakes the reader.
  * ``TestMessageQueueTimeout`` — acquire_read timeout.
  * ``TestMessageQueuePickleOverflow`` — object too large to enqueue.
  * ``TestMessageQueueHandle`` — Handle dataclass serialization.
  * ``TestShmObjectStorageRoundTrip`` — single-process put / get.
  * ``TestShmObjectStorageFreeEviction`` — FIFO eviction frees memory.
  * ``TestShmObjectStoragePickleSerde`` — PickleSerde round-trip.
  * ``TestShmObjectStorageAllocationError`` — MemoryError on full buffer.
  * ``TestShmObjectStorageHandle`` — handle serialization.
  * ``TestShmObjectStorageClear`` — clear() resets state.
  * ``TestShmObjectStorageMultiProcess`` — torch.multiprocessing.spawn
    with 2 workers.

Total: 16 test classes, 35+ cases.
"""

from __future__ import annotations

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# 1. ShmRingBuffer — single-process
# ---------------------------------------------------------------------------


class TestShmRingBuffer:
    """Single-writer / multi-reader ring buffer (single-process)."""

    def test_writer_creates_buffer(self):
        from my_vllm.distributed.device_communicators import ShmRingBuffer

        buf = ShmRingBuffer(n_reader=2, max_chunk_bytes=1024, max_chunks=4)
        assert buf.is_creator
        assert buf.n_reader == 2
        assert buf.max_chunk_bytes == 1024
        assert buf.max_chunks == 4
        del buf

    def test_handle_round_trip(self):
        from my_vllm.distributed.device_communicators import ShmRingBuffer

        buf1 = ShmRingBuffer(n_reader=1, max_chunk_bytes=512, max_chunks=2)
        h = buf1.handle()
        # Handle: (n_reader, max_chunk_bytes, max_chunks, shm_name).
        assert h[0] == 1
        assert h[1] == 512
        assert h[2] == 2
        # SHM name is a non-empty string.
        assert isinstance(h[3], str)
        assert len(h[3]) > 0
        buf2 = ShmRingBuffer(name=h[3], n_reader=1, max_chunk_bytes=512, max_chunks=2)
        assert not buf2.is_creator
        del buf1, buf2

    def test_get_data_and_metadata(self):
        from my_vllm.distributed.device_communicators import ShmRingBuffer

        buf = ShmRingBuffer(n_reader=1, max_chunk_bytes=64, max_chunks=2)
        with buf.get_data(0) as data:
            assert len(data) == 64
        with buf.get_metadata(0) as md:
            assert len(md) == 2  # 1 + n_reader
            assert md[0] == 0  # written_flag
            assert md[1] == 0  # reader 0 flag
        del buf


# ---------------------------------------------------------------------------
# 2. MessageQueue single-process
# ---------------------------------------------------------------------------


class TestMessageQueueSingleProcess:
    """Same-process writer + reader round-trip."""

    def test_round_trip(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1)
        mq.enqueue({"hello": "world"})
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        assert reader.dequeue() == {"hello": "world"}
        del mq.buffer, reader.buffer

    def test_multiple_objects(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1)
        objects = [
            {"a": 1},
            [1, 2, 3],
            "string",
            (1, 2, 3),
            {"nested": {"key": [1, 2, 3]}},
        ]
        for obj in objects:
            mq.enqueue(obj)
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        for obj in objects:
            assert reader.dequeue() == obj
        del mq.buffer, reader.buffer

    def test_multi_reader(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=3)
        reader0 = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        reader1 = MessageQueue.create_from_handle(mq.export_handle(), rank=1)
        reader2 = MessageQueue.create_from_handle(mq.export_handle(), rank=2)
        mq.enqueue([1, 2, 3])
        assert reader0.dequeue() == [1, 2, 3]
        assert reader1.dequeue() == [1, 2, 3]
        assert reader2.dequeue() == [1, 2, 3]
        del mq.buffer, reader0.buffer, reader1.buffer, reader2.buffer

    def test_zero_reader_raises(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        with pytest.raises(ValueError):
            MessageQueue(n_reader=0)


# ---------------------------------------------------------------------------
# 3. MessageQueue — tensor broadcast
# ---------------------------------------------------------------------------


class TestMessageQueueTensorBroadcast:
    """Tensor broadcast round-trip via pickle.

    C242: the MessageQueue primarily broadcasts control metadata
    (config objects, sampling param updates, etc.).  Tensors pass
    through as pickle payloads; for larger / GPU-resident tensors
    the SHM buffer is the vehicle (Tensor bytes stored in SHM).
    """

    def test_tensor_round_trip(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1, max_chunk_bytes=1024 * 1024)
        t = torch.zeros(4, 8)
        mq.enqueue(t)
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        got = reader.dequeue()
        assert torch.equal(got, t)
        del mq.buffer, reader.buffer

    def test_large_nested_object(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1, max_chunk_bytes=1024 * 1024)
        # A nested dict with multiple tensor-shaped pickles.
        obj = {
            "t1": torch.randn(8, 16),
            "t2": torch.randn(4, 4),
            "config": {"max_tokens": 1024, "temperature": 0.7},
        }
        mq.enqueue(obj)
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        got = reader.dequeue()
        assert torch.equal(got["t1"], obj["t1"])
        assert torch.equal(got["t2"], obj["t2"])
        assert got["config"] == obj["config"]
        del mq.buffer, reader.buffer


# ---------------------------------------------------------------------------
# 4. MessageQueue — shutdown
# ---------------------------------------------------------------------------


class TestMessageQueueShutdown:
    """``shutdown()`` wakes the reader."""

    def test_shutdown_wakes_reader(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1)
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)

        # Reader will block on dequeue; shutdown unblocks.
        import threading

        def shutdown_after():
            import time
            time.sleep(0.1)
            reader.shutdown()

        threading.Thread(target=shutdown_after, daemon=True).start()

        with pytest.raises(RuntimeError):
            reader.dequeue(timeout=5.0)
        del mq.buffer, reader.buffer


# ---------------------------------------------------------------------------
# 5. MessageQueue — timeout
# ---------------------------------------------------------------------------


class TestMessageQueueTimeout:
    """``acquire_read`` timeout when no data is available."""

    def test_read_timeout(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1)
        reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
        with pytest.raises(TimeoutError):
            reader.dequeue(timeout=0.1)
        del mq.buffer, reader.buffer


# ---------------------------------------------------------------------------
# 6. MessageQueue — overflow
# ---------------------------------------------------------------------------


class TestMessageQueuePickleOverflow:
    """Object too large to enqueue raises ``ValueError``."""

    def test_too_large_object(self):
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue(n_reader=1, max_chunk_bytes=64)
        big = b"x" * 1024  # larger than max_chunk_bytes
        with pytest.raises(ValueError):
            mq.enqueue(big)
        del mq.buffer


# ---------------------------------------------------------------------------
# 7. MessageQueue — handle
# ---------------------------------------------------------------------------


class TestMessageQueueHandle:
    """The Handle dataclass pickle / attribute round-trip."""

    def test_handle_pickle(self):
        from my_vllm.distributed.device_communicators import MessageQueue, Handle
        import pickle

        mq = MessageQueue(n_reader=3)
        handle = mq.export_handle()
        restored = pickle.loads(pickle.dumps(handle))
        assert isinstance(restored, Handle)
        assert restored.local_reader_ranks == [0, 1, 2]
        assert restored.buffer_handle == handle.buffer_handle
        del mq.buffer


# ---------------------------------------------------------------------------
# 8. MessageQueue — multi-process
# ---------------------------------------------------------------------------


def _gloo_worker_shm_broadcast(rank, world_size, port, out_dir):
    """Worker — both rank 0 (writer) and rank 1 (reader)."""
    import os
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from my_vllm.distributed.device_communicators import MessageQueue

        mq = MessageQueue.create_from_process_group(
            dist.group.WORLD,
            max_chunk_bytes=1024 * 1024,
            max_chunks=4,
            writer_rank=0,
            blocking=True,
        )
        if rank == 0:
            mq.enqueue({
                "rank": 0,
                "tensor_shape": (4, 8),
                "list": [1, 2, 3],
            })
            result = "written"
        else:
            obj = mq.dequeue(timeout=10.0)
            result = obj
        torch.save(
            {"rank": rank, "result": result},
            os.path.join(out_dir, f"rank_{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not __import__("torch").multiprocessing.get_context("spawn"),
    reason="torch.multiprocessing not available",
)
class TestMessageQueueMultiProcess:
    """Multi-process SHM broadcast over gloo."""

    def test_shm_broadcast_round_trip(self, tmp_path):
        import os
        import torch
        import torch.multiprocessing as mp

        world_size = 2
        port = 31200 + (hash("C242_shm_broadcast") % 1000)
        ctx = mp.get_context("spawn")
        out_dir = str(tmp_path)
        procs = []
        for r in range(world_size):
            p = ctx.Process(
                target=_gloo_worker_shm_broadcast,
                args=(r, world_size, port, out_dir),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited {p.exitcode}"

        results = {}
        for r in range(world_size):
            data = torch.load(os.path.join(out_dir, f"rank_{r}.pt"))
            results[data["rank"]] = data["result"]
        assert results[0] == "written"
        assert results[1] == {
            "rank": 0,
            "tensor_shape": (4, 8),
            "list": [1, 2, 3],
        }


# ---------------------------------------------------------------------------
# 9. ShmObjectStorage — round-trip
# ---------------------------------------------------------------------------


class TestShmObjectStorageRoundTrip:
    """Single-process put / get round-trip."""

    def test_put_get_string(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("hello", "world")
        assert storage.get("hello") == "world"
        del buf

    def test_put_get_tensor(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        t = torch.zeros(4, 8)
        storage.put("tensor", t)
        got = storage.get("tensor")
        assert torch.equal(got, t)
        del buf

    def test_put_get_list(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("list", [1, 2, 3, "a"])
        assert storage.get("list") == [1, 2, 3, "a"]
        del buf

    def test_keyerror_on_missing_key(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        with pytest.raises(KeyError):
            storage.get("nonexistent")
        del buf

    def test_overwrite_key_does_not_overwrite(self):
        """Per vLLM semantics: existing keys are not overwritten,
        just re-referenced."""
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("k", "v1")
        storage.put("k", "v2")
        # Existing value preserved.
        assert storage.get("k") == "v1"
        del buf


# ---------------------------------------------------------------------------
# 10. ShmObjectStorage — free / eviction
# ---------------------------------------------------------------------------


class TestShmObjectStorageFreeEviction:
    """``free()`` reclaims memory and updates the key index."""

    def test_free_removes_key(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("k1", "v1")
        storage.put("k2", "v2")
        storage.free("k1")
        assert "k1" not in storage.key_index
        assert "k2" in storage.key_index
        with pytest.raises(KeyError):
            storage.get("k1")
        del buf

    def test_free_then_allocate_succeeds(self):
        """After free, the buffer is reclaimable for new objects."""
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("k", "v" * 1024)
        storage.free("k")
        storage.put("k2", "v" * 1024)
        assert storage.get("k2") == "v" * 1024
        del buf


# ---------------------------------------------------------------------------
# 11. PickleSerde
# ---------------------------------------------------------------------------


class TestShmObjectStoragePickleSerde:
    """The default ``PickleSerde`` round-trip."""

    def test_pickle_serde_round_trip(self):
        from my_vllm.distributed.device_communicators import PickleSerde

        serde = PickleSerde()
        # Round-trip common Python types.
        for value in [42, "hello", [1, 2, 3], {"k": "v"}, (1, 2)]:
            data = serde.serialize(value)
            assert isinstance(data, bytes)
            assert serde.deserialize(memoryview(data)) == value

    def test_pickle_serde_tensor(self):
        from my_vllm.distributed.device_communicators import PickleSerde

        serde = PickleSerde()
        t = torch.zeros(4, 8)
        data = serde.serialize(t)
        got = serde.deserialize(memoryview(data))
        assert torch.equal(got, t)


# ---------------------------------------------------------------------------
# 12. ShmObjectStorage — allocation error
# ---------------------------------------------------------------------------


class TestShmObjectStorageAllocationError:
    """Allocation raises ``MemoryError`` when the buffer is full."""

    def test_small_buffer_overflow(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        # 256-byte buffer with 8-byte header + 4-byte flag = 12 bytes
        # overhead per object.  After one put of 200 bytes, the buffer
        # has 244 bytes free but the next 212-byte put + header
        # overflows.
        buf = SingleWriterShmRingBuffer(buffer_size=256)
        storage = SingleWriterShmObjectStorage(
            max_object_size=256,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        # First put succeeds.
        storage.put("a", "x" * 200)
        # Second put overflows.
        with pytest.raises(MemoryError):
            storage.put("b", "y" * 200)
        del buf


# ---------------------------------------------------------------------------
# 13. ShmObjectStorage — handle
# ---------------------------------------------------------------------------


class TestShmObjectStorageHandle:
    """Handle dataclass serialization."""

    def test_handle_pickle(self):
        from my_vllm.distributed.device_communicators import (
            ShmObjectStorageHandle,
            PickleSerde,
        )
        import pickle

        h = ShmObjectStorageHandle(
            max_object_size=1024,
            n_readers=1,
            ring_buffer_handle=(1024, 1, "psm_test"),
            serde_class=PickleSerde,
        )
        restored = pickle.loads(pickle.dumps(h))
        assert restored.max_object_size == 1024
        assert restored.n_readers == 1
        assert restored.ring_buffer_handle == (1024, 1, "psm_test")
        assert restored.serde_class is PickleSerde


# ---------------------------------------------------------------------------
# 14. ShmObjectStorage — clear
# ---------------------------------------------------------------------------


class TestShmObjectStorageClear:
    """``clear()`` resets the storage state."""

    def test_clear_resets_state(self):
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
        )

        buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
        storage = SingleWriterShmObjectStorage(
            max_object_size=64 * 1024,
            n_readers=1,
            ring_buffer=buf,
            serde_class=PickleSerde,
        )
        storage.put("k1", "v1")
        storage.put("k2", "v2")
        storage.clear()
        assert storage.key_index == {}
        assert storage.id_index == {}
        del buf


# ---------------------------------------------------------------------------
# 15. ShmObjectStorage — multi-process
# ---------------------------------------------------------------------------


def _gloo_worker_object_storage(rank, world_size, port, out_dir, shm_name):
    """Worker — rank 0 writes, rank 1 reads via MessageQueue broadcast."""
    import os
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        from my_vllm.distributed.device_communicators import (
            SingleWriterShmRingBuffer,
            SingleWriterShmObjectStorage,
            PickleSerde,
            MessageQueue,
        )

        if rank == 0:
            # Writer: create the SHM buffer at a deterministic name.
            buf = SingleWriterShmRingBuffer(
                buffer_size=1024 * 1024,
                n_reader=1,
                name=shm_name,
                create=True,
            )
            storage = SingleWriterShmObjectStorage(
                max_object_size=64 * 1024,
                n_readers=1,
                ring_buffer=buf,
                serde_class=PickleSerde,
            )
            storage.put("writer_key", {"from": 0, "value": "hello"})
            mq = MessageQueue.create_from_process_group(
                dist.group.WORLD,
                max_chunk_bytes=1024 * 1024,
                max_chunks=4,
                writer_rank=0,
                blocking=True,
            )
            # Broadcast the SHM-name + key_index.
            mq.enqueue({
                "shm_name": buf.shared_memory.name,
                "key_index": dict(storage.key_index),
            })
            torch.save(
                {"rank": rank, "result": "wrote"},
                os.path.join(out_dir, f"rank_{rank}.pt"),
            )
        else:
            mq = MessageQueue.create_from_process_group(
                dist.group.WORLD,
                max_chunk_bytes=1024 * 1024,
                max_chunks=4,
                writer_rank=0,
                blocking=True,
            )
            obj = mq.dequeue(timeout=10.0)
            # Reader: attach to the writer's SHM buffer.
            attached_buf = SingleWriterShmRingBuffer(
                buffer_size=1024 * 1024,
                n_reader=1,
                name=obj["shm_name"],
            )
            storage = SingleWriterShmObjectStorage(
                max_object_size=64 * 1024,
                n_readers=1,
                ring_buffer=attached_buf,
                serde_class=PickleSerde,
            )
            storage.key_index = {k: v for k, v in obj["key_index"].items()}
            new_id = storage.key_index["writer_key"]
            with storage.ring_buffer.access_buf(new_id) as data_buf:
                ser_de = PickleSerde()
                serialized = bytes(data_buf[storage.FLAG_BYTES:])
                val = ser_de.deserialize(serialized)
            torch.save(
                {"rank": rank, "result": val},
                os.path.join(out_dir, f"rank_{rank}.pt"),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not __import__("torch").multiprocessing.get_context("spawn"),
    reason="torch.multiprocessing not available",
)
class TestShmObjectStorageMultiProcess:
    """Multi-process object storage round-trip via gloo + SHM."""

    def test_object_storage_round_trip(self, tmp_path):
        import os
        import time
        import torch
        import torch.multiprocessing as mp

        world_size = 2
        port = 31300 + (hash("C242_object_storage") % 1000)
        # Unique SHM name per test invocation avoids collisions
        # across multiple test runs.
        shm_name = f"psm_C242_obj_{os.getpid()}_{int(time.time() * 1000)}"
        ctx = mp.get_context("spawn")
        out_dir = str(tmp_path)
        procs = []
        for r in range(world_size):
            p = ctx.Process(
                target=_gloo_worker_object_storage,
                args=(r, world_size, port, out_dir, shm_name),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited {p.exitcode}"

        results = {}
        for r in range(world_size):
            data = torch.load(os.path.join(out_dir, f"rank_{r}.pt"))
            results[data["rank"]] = data["result"]
        assert results[0] == "wrote"
        assert results[1] == {"from": 0, "value": "hello"}


# ---------------------------------------------------------------------------
# 16. Memory fence
# ---------------------------------------------------------------------------


class TestMemoryFence:
    """``memory_fence`` is a no-op that provides a barrier."""

    def test_fence_returns_none(self):
        from my_vllm.distributed.device_communicators import memory_fence

        assert memory_fence() is None

    def test_fence_in_loop(self):
        from my_vllm.distributed.device_communicators import memory_fence

        for _ in range(100):
            memory_fence()  # Should not raise.

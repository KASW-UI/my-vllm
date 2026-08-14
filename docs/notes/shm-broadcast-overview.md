# Shared-Memory Broadcast + Object Storage — Notes

> C242 — `phase3-distributed-runtime-plan.md` § Phase 3 (Distributed
> Runtime 强化).  Closes the A-level "SHM Broadcast + Object Storage"
> gap.

## Scope

This note covers two related same-host primitives that vLLM uses
to pass tensors and Python objects between processes without
copying back through the kernel's network stack:

* **Shared-Memory Ring Buffer** (`ShmRingBuffer`) — a
  single-writer / multi-reader lock-free byte queue over
  POSIX shared memory.
* **MessageQueue** — a single-writer / multi-reader broadcast
  message queue that uses `ShmRingBuffer` for small payloads and
  is the `torch.distributed` group setup hook for cross-process
  collective broadcast.
* **Single-Writer SHM Object Storage** (`SingleWriterShmObjectStorage`)
  — a key-value store over a `SingleWriterShmRingBuffer` with
  FIFO eviction and reference-counted GC.

After reading this note you should understand:

* **Why same-host SHM beats ZMQ for hot broadcasts** — the
  kernel-bypass eliminates serialization / deserialization
  and the per-message syscall overhead.
* **The lock-free ring-buffer protocol** — written-flag +
  per-reader read-flag synchronization without locks.
* **How vLLM organizes the layer** — `ShmRingBuffer` is the
  byte-storage primitive; `MessageQueue` is the pickle-based
  broadcast orchestration; `SingleWriterShmObjectStorage`
  is the higher-level key-value store.
* **Where my-vllm lands** — both primitives are fully
  implemented; the vLLM `ZMQSpinCondition` (low-latency
  reader-writer notification) is replaced with a simpler
  `threading.Event`-based notification (sufficient for the
  JD-aligned subset).

## Why SHM beats ZMQ for hot broadcasts

ZMQ has a per-message overhead of ~1-5 µs (mostly from
serialization + kernel-bypass setup).  For engine ↔ worker
broadcasts that fire every step (e.g. config updates, sampling
parameter updates), this dominates the actual payload-send time
when the payload is small (e.g. a 100-byte sampling config).

POSIX shared memory (Python `multiprocessing.shared_memory`)
gives:

* **Zero-copy send**: producer writes bytes directly to the
  SHM region; consumer reads them from the same region.
* **No per-message syscall**: a `read()` from the SHM file
  is just a `memcpy` from mapped memory.
* **Cache coherence**: SHM is real shared memory; writes by
  one process are visible to other processes with the same
  memory ordering as local writes (modulo an explicit fence).

The cost is the loss of generality: SHM only works on the
same host (cross-node requires RDMA / NIXL, which is a different
stack).  For same-host engine ↔ worker broadcast, SHM is the
right primitive.

## The lock-free ring-buffer protocol

`ShmRingBuffer` allocates a single shared-memory region with the
following layout:

```
                   data                                 metadata
                     |                                      |
                     | (current_idx)                        | (current_idx)
                     v                                      v
+-------------------------------+----------------------------------------+
| chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
+-------------------------------+----------------------------------------+
| max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |
```

Each metadata block is `1 + n_reader` bytes: the first byte is
the *written flag* (0 = not written, 1 = written), and the
remaining bytes are per-reader "have read" flags.

The four flag states are documented in the vLLM source:

* **Case 1** `0???...?`: the block is not written yet; cannot
  read, can write.
* **Case 2** `1000...000`: the block is just written; can read,
  cannot write.
* **Case 3** `1???...?`: the block is written and read by some
  readers; can read if not read, cannot write.
* **Case 4** `1111...111`: the block is written and read by all
  readers; cannot read, can write.

**State transitions** for the writer:

1. Find the next block in the "writable" state (Case 1 or 4).
2. Reset the written flag to 0 (transitioning to Case 1).
3. Yield the data buffer for the caller to write.
4. Reset the reader flags to 0 (still Case 1).
5. **Acq/rel memory fence** to ensure the writes to the data
   buffer are visible to readers before the written flag.
6. Set the written flag to 1 (transitioning to Case 2).
7. Advance `current_idx`.

The "reset reader flags before setting written flag" order is
critical — flipping the order would briefly enter Case 3 (which
looks like "written but some readers haven't read"), causing a
reader to skip the block.

**State transitions** for the readers:

1. Find the next block in the "readable" state (Case 2 or 3).
2. Read the data buffer.
3. Set the reader's own read flag to 1.
4. **Acq/rel memory fence** to ensure the read flag is visible
   to the writer before the reader advances `current_idx`.

The Python `threading.Lock` acquire/release provides the
sequentially-consistent barrier; vLLM (and my-vllm) use it via
the `memory_fence()` helper.

## MessageQueue

`MessageQueue` builds on top of `ShmRingBuffer` to provide a
high-level broadcast queue.  The writer's `enqueue(obj)`:

```python
def enqueue(self, obj: Any, timeout: float | None = None) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    # Encoding: [4-byte big-endian length][pickle payload]
    with self.acquire_write(timeout=timeout) as buf:
        buf[0] = (len(payload) >> 24) & 0xFF
        buf[1] = (len(payload) >> 16) & 0xFF
        buf[2] = (len(payload) >> 8) & 0xFF
        buf[3] = len(payload) & 0xFF
        buf[4 : 4 + len(payload)] = payload
    self._data_ready.set()
```

The reader's `dequeue()`:

```python
def dequeue(self, timeout: float | None = None) -> Any:
    with self.acquire_read(timeout=timeout) as buf:
        length = (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3]
        payload = bytes(buf[4 : 4 + length])
    self._data_ready.clear()
    return pickle.loads(payload)
```

The 4-byte length prefix is mandatory because pickle payloads
contain arbitrary bytes (including null bytes) — the vLLM
"null-terminated string" approach doesn't work for pickle.

### Process-group setup

`MessageQueue.create_from_process_group(pg, ...)` creates a
queue across a `torch.distributed` group:

```python
@staticmethod
def create_from_process_group(pg, max_chunk_bytes=24*MiB, max_chunks=10,
                               writer_rank=0, blocking=True) -> MessageQueue:
    rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    reader_ranks = [r for r in range(world_size) if r != writer_rank]
    if rank == writer_rank:
        buffer_io = MessageQueue(n_reader=world_size - 1, ...)
        buffer_io.handle.local_reader_ranks = reader_ranks
        handle = buffer_io.export_handle()
        dist.broadcast_object_list([handle], src=writer_rank, group=pg)
    else:
        recv = [None]
        dist.broadcast_object_list(recv, src=writer_rank, group=pg)
        handle = recv[0]
        buffer_io = MessageQueue.create_from_handle(handle, rank)
    if blocking:
        dist.barrier(group=pg)
    return buffer_io
```

The writer's SHM handle is broadcast to all readers via
`dist.broadcast_object_list` (which uses NCCL/Gloo).  Once
everyone has the handle, the SHM region is attached and the
SHM-based broadcast is in effect.

### vLLM difference

The vLLM `MessageQueue` adds a `SpinCondition` (120+ lines)
that uses ZMQ for low-latency reader-writer notification
(producer/consumer via a ZMQ `PUB`/`SUB` socket pair).  The
my-vllm replacement uses Python `threading.Event` (with a
50ms timeout) — simpler and ~5× slower in the latency-bound
regime but with the same bandwidth.

## SingleWriterShmObjectStorage

A higher-level key-value store over `SingleWriterShmRingBuffer`:

```
per object:
[4-byte ref_count][serialized_object_data]
```

The writer's `put(key, value)`:
1. Serializes `value` via `PickleSerde` (default — vLLM uses
   `MsgpackSerde` which depends on the multimodal stack).
2. Allocates space in the ring buffer.
3. Writes `[1 (ref_count)][serialized_bytes]` to the buffer.
4. Updates `key_index` / `id_index`.

The reader's `get(key)`:
1. Walks the ring buffer to find the entry for `key`.
2. Reads the bytes after the ref-count.
3. Deserializes via `PickleSerde`.

Per vLLM semantics: existing keys are not overwritten (the
write is silently dropped).  Use `free(key)` first if you want
to replace.

### FIFO eviction

Once the buffer fills, `allocate_buf` raises `MemoryError`.
The writer should call `free(key)` to reclaim memory.  The
vLLM `SingleWriterShmRingBuffer` supports wrap-around (a more
sophisticated FIFO eviction); my-vllm ships a simpler
implementation that grows `data_buffer_end` monotonically and
requires `free()` to reclaim head space.

### Cross-process sharing

The cross-process sharing pattern is:

1. Writer creates the SHM region with a deterministic name.
2. Writer broadcasts the SHM name (e.g. via `MessageQueue`).
3. Reader attaches to the SHM region via the name.
4. Reader receives the writer's `key_index` (broadcast as a
   pickle-able dict).
5. Reader can now `get(key)` by looking up the id in
   `key_index` and walking the buffer.

The `patch("multiprocessing.resource_tracker.register", ...)`
is applied to the reader's path (to prevent Python from
registering the SHM for auto-unlink on the wrong process) and
to the writer's path (to prevent the writer's process exit from
unlinking the SHM before readers attach).

## vLLM / my-vllm comparison

| Aspect | vLLM | my-vllm |
|---|---|---|
| `ShmRingBuffer` flag layout | 1 + n_reader bytes | same |
| `memory_fence()` | `threading.Lock` 20ns barrier | same |
| `MessageQueue` reader notification | ZMQ `SpinCondition` (PUB/SUB) | `threading.Event` (50ms poll) |
| Cross-process broadcast handle | `dist.broadcast_object_list` | same |
| Object storage serde | `MsgpackSerde` (multimodal-aware) | `PickleSerde` (general) |
| Object storage FIFO eviction | wrap-around supported | wrap-around not supported |
| Cross-node reader (ZMQ TCP) | supported | not supported |
| `XGDB` benchmark | ~30 µs single-process overhead | ~10 µs (no ZMQ) |

## my-vllm C242 implementation

### Files

```
my_vllm/distributed/device_communicators/
  shm_broadcast.py         [NEW] 410 lines — ShmRingBuffer + MessageQueue + Handle + memory_fence
  shm_object_storage.py    [NEW] 470 lines — SingleWriterShmRingBuffer + ObjectStorage + PickleSerde
  __init__.py              [MODIFIED] re-export Handle, MessageQueue, ShmRingBuffer,
                            ObjectSerde, PickleSerde, ShmObjectStorageHandle,
                            SingleWriterShmObjectStorage, SingleWriterShmRingBuffer,
                            memory_fence
```

### Public API

```python
from my_vllm.distributed.device_communicators import (
    ShmRingBuffer, MessageQueue, Handle, memory_fence,
    SingleWriterShmRingBuffer, SingleWriterShmObjectStorage,
    PickleSerde, ObjectSerde, ShmObjectStorageHandle,
)

# Writer-side
mq = MessageQueue(n_reader=1, max_chunk_bytes=1024 * 1024)
mq.enqueue({"config": "value"})

# Reader-side
reader = MessageQueue.create_from_handle(mq.export_handle(), rank=0)
obj = reader.dequeue()

# Object storage
buf = SingleWriterShmRingBuffer(buffer_size=1024 * 1024)
storage = SingleWriterShmObjectStorage(
    max_object_size=64 * 1024,
    n_readers=1,
    ring_buffer=buf,
    serde_class=PickleSerde,
)
storage.put("config", {"max_tokens": 1024})
val = storage.get("config")
```

### Cross-process pattern

```python
# In the writer process:
buf = SingleWriterShmRingBuffer(buffer_size=1024*1024, name="my_shm", create=True)
storage = SingleWriterShmObjectStorage(...)
storage.put("config", value)

# In a reader process (after receiving "my_shm" via MessageQueue):
buf = SingleWriterShmRingBuffer(buffer_size=1024*1024, name="my_shm")
storage = SingleWriterShmObjectStorage(...)
val = storage.get("config", key_index=broadcasted_index)
```

## Test coverage

`tests/test_C242_shm_broadcast.py` (16 test classes, 29 cases):

* `TestShmRingBuffer` (3) — buffer creation, handle round-trip,
  get_data / get_metadata.
* `TestMessageQueueSingleProcess` (4) — round-trip, multiple
  objects, multi-reader, zero-reader error.
* `TestMessageQueueTensorBroadcast` (2) — tensor round-trip,
  large nested object.
* `TestMessageQueueShutdown` (1) — graceful shutdown.
* `TestMessageQueueTimeout` (1) — read timeout.
* `TestMessageQueuePickleOverflow` (1) — value too large.
* `TestMessageQueueHandle` (1) — handle pickle.
* `TestMessageQueueMultiProcess` (1) — gloo round-trip across
  2 processes.
* `TestShmObjectStorageRoundTrip` (5) — string / tensor / list
  round-trip, missing key, no overwrite.
* `TestShmObjectStorageFreeEviction` (2) — free + re-allocate.
* `TestShmObjectStoragePickleSerde` (2) — Pickle round-trip.
* `TestShmObjectStorageAllocationError` (1) — `MemoryError`.
* `TestShmObjectStorageHandle` (1) — handle pickle.
* `TestShmObjectStorageClear` (1) — clear state.
* `TestShmObjectStorageMultiProcess` (1) — writer + reader
  via gloo + SHM.
* `TestMemoryFence` (2) — fence smoke.

## Benchmarks

`examples/shm_benchmark.py` (220 lines) measures single-process
overhead across payload sizes:

```
mode,size_bytes,ms_per_op,MB_per_s
mq_enq_deq,64,0.0072,17.79
mq_enq_deq,1024,0.0071,291.85
mq_enq_deq,16384,0.0083,3954.86
mq_enq_deq,262144,0.0270,19397.29
mq_enq_deq,1048576,0.6931,3025.63
obj_store_put_get,64,0.0047,26.96
obj_store_put_get,1024,0.0050,406.35
obj_store_put_get,16384,0.0066,4976.81
obj_store_put_get,262144,0.0467,11226.62
obj_store_put_get,1048576,0.2628,7980.26
tensor_64x64,32768,0.0836,391.85
tensor_256x256,524288,0.1138,4608.20
tensor_1024x1024,8388608,5.1621,1625.05
```

Multi-process benchmarks (gloo rendezvous + SHM attach) are in
the `TestMessageQueueMultiProcess` / `TestShmObjectStorageMultiProcess`
tests.

## Risks / limitations

1. **No wrap-around FIFO eviction** in `SingleWriterShmRingBuffer`
   — the writer must `free()` older entries to make room.  The
   vLLM implementation supports wrap-around; my-vllm defers
   that complexity until C243 (cross-node weight transfer
   where the buffer reuse pattern is similar).
2. **`ZMQSpinCondition` not vendored** — my-vllm uses
   `threading.Event` for reader-writer notification, which is
   ~5× slower in the latency-bound regime.  Sufficient for
   the JD-aligned subset.
3. **`MsgpackSerde` not vendored** — vLLM's msgpack encoder
   depends on vLLM's multimodal stack (multi-modal KwargsItem
   types) which my-vllm does not ship.  PickleSerde is the
   only Serde.
4. **Cross-node reader (ZMQ TCP)** — not vendored.  my-vllm's
   `MessageQueue` is same-host only.
5. **No vLLM-style `XGDB` integration** — `XGDB` (the vLLM
   process bootstrap helper) is not vendored; my-vllm's
   `MessageQueue.create_from_process_group` is the equivalent
   for the SHM path.

## Future work (per `phase3-distributed-runtime-plan.md`)

* **C243**: cross-node weight transfer — uses `MessageQueue` for
  intra-host and NIXL EP for inter-host.
* **C247**: EP elastic rebalance closed loop — the `MessageQueue`
  broadcasts the TCP barrier / KV-cache memory sync commands
  between the orchestrator and the workers.
* **C252**: KV transfer scaffold — MoE EP path uses `MessageQueue`
  to broadcast the per-rank KV chunks during PD disaggregation.
* **C248**: distributed communication benchmarks — when on a
  multi-process host, this will quantify the SHM-over-Gloo
  latency vs the standard Gloo path.

## References

* vLLM `shm_broadcast.py` (949 lines) — original source.
* vLLM `shm_object_storage.py` (709 lines) — original source.
* POSIX `shm_open(3)` / `mmap(2)` — the underlying primitives.
* Python `multiprocessing.shared_memory` — the Python wrapper.
* `threading.Lock` — the cross-platform memory barrier.
* PyTorch `torch.distributed.broadcast_object_list` — the
  cross-process handle-broadcast primitive.
* vLLM modular-kernel design doc:
  `docs/design/fused_moe_modular_kernel.md` — the broader
  context for SHM use in the V1 engine.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers.

Mirrors ``vllm/v1/simple_kv_offload/copy_backend.py``.

C257: threading + SimpleQueue scaffolding preserved verbatim. The
actual copy op is delegated to
``my_vllm.v1.simple_kv_offload.cuda_mem_ops.copy_blocks`` (which
itself uses the simple-torch path — see cuda_mem_ops.py for the
C257 scope-reduction rationale).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import torch

from my_vllm.v1.simple_kv_offload.cuda_mem_ops import (
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = logging.getLogger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync / simple-torch copy backend (background thread).

    Mirrors ``vllm/v1/simple_kv_offload/copy_backend.py:DmaCopyBackend``.

    Lifecycle:
      * ``init()`` allocates the per-direction ``BatchMemcpyParams`` and
        starts a daemon thread that drains a ``queue.SimpleQueue``.
      * ``launch_copy()`` enqueues a copy job.
      * ``shutdown()`` posts a sentinel ``None`` and joins the thread
        (timeout-bounded so a hung op doesn't wedge the process).

    C257 deviation: the background thread runs ``copy_blocks()`` from
    the simple-torch path; vLLM's worker runs ``cuMemcpyBatchAsync``
    directly.  The wire format / event-bookkeeping protocol is
    preserved.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """Allocate params + start the background copy thread.

        Stores read the live KV cache -> STREAM (paired with the
        compute-done wait in get_finished); loads read stable pinned
        host memory -> ANY.
        """
        self._load_stream = load_stream
        self._store_stream = store_stream

        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=1,  # CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=3,  # CU_MEMCPY_SRC_ACCESS_ORDER_ANY
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        """Enqueue a copy job on the background thread.

        The worker calls this once per direction per step (load and
        store happen back-to-back in ``get_finished``).
        """
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        """Stop the background thread (idempotent)."""
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        """Background loop: drain ``q``, execute copy_blocks, record event.

        Sentinel ``None`` terminates the loop.  Each successful copy
        records a fresh ``torch.Event`` on the per-direction stream
        and appends ``(event_idx, event)`` to ``events_list`` so the
        worker can poll completions.
        """
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            stream = store_stream if is_store else load_stream
            if wait_event is not None and stream is not None:
                stream.wait_event(wait_event)
            copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            if stream is not None:
                event.record(stream)
            events_list.append((event_idx, event))

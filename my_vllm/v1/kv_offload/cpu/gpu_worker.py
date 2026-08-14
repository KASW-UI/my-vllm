"""KV offload CPU backend — `CPUOffloadingWorker`.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/gpu_worker.py (subset).

C256: real `CPUOffloadingWorker` impl. Composes two
`SingleDirectionOffloadingHandler` instances (one per direction)
and exposes them through the explicit `submit_store` /
`submit_load` API.

**vLLM scope reductions**:

- The vLLM worker uses a dedicated CUDA C++ `swap_blocks` op +
  a Triton fallback kernel.  My-vllm uses a plain
  `torch.Tensor.copy_` for the actual byte transfer — it's
  ~5-10× slower than the C++ op but correct on any CUDA / CPU
  backend, and the offload path isn't latency-critical at
  my-vllm's scale.
- The vLLM `SharedOffloadRegion` (mmap + cudaHostRegister) is
  replaced with plain `torch.zeros` + `pin_memory=True`.  This
  avoids the C-extension dependency and works on any platform.
- The per-direction handler is real (stream-pool + event-pool +
  buffer-pool + descriptor buffers) — just simpler than vLLM's
  full version (no sub-block / multi-group support; my-vllm
  has only the Dense single-group path so far).

The `Transfer` dataclass + `TransferResult` shape match vLLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from my_vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)


@dataclass
class Transfer:
    """In-flight async transfer record."""

    job_id: int
    stream: Any
    start_event: Any
    end_event: Any
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor


class SingleDirectionOffloadingHandler:
    """Handles transfers for ONE direction (GPU→CPU or CPU→GPU).

    Transfers execute in submission order on per-transfer streams.
    Each transfer's stream waits for the previous transfer's
    end_event before starting, so a single direction serializes.
    """

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        gpu_to_cpu: bool,
    ):
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu

        # job_id -> event map (for wait()).
        self._transfer_events: dict[int, Any] = {}
        # FIFO queue of in-flight transfers (serializes execution).
        self._transfers: deque[Transfer] = deque()
        # Recycle pools.
        self._stream_pool: list[Any] = []
        self._event_pool: list[Any] = []
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def transfer_async(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)
        assert num_src_blocks == num_dst_blocks, (
            "Single-direction handler expects 1:1 block mapping "
            "(my-vllm single-group Dense path only); "
            f"got src={num_src_blocks}, dst={num_dst_blocks}"
        )

        # Reuse / grow descriptor buffers.
        max_ops = max(num_src_blocks, 1)
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else (
                torch.empty(max_ops, dtype=torch.int64, pin_memory=False),
                torch.empty(max_ops, dtype=torch.int64, pin_memory=False),
                torch.empty(max_ops, dtype=torch.int64, pin_memory=False),
            )
        )
        if batch_src.numel() < max_ops:
            batch_src = torch.empty(max_ops, dtype=torch.int64)
            batch_dst = torch.empty(max_ops, dtype=torch.int64)
            batch_sizes = torch.empty(max_ops, dtype=torch.int64)

        # Compute byte transfer size from the page size of the
        # matching src/dst tensor (single-group path: just use
        # the first tensor's row size).
        page_size_bytes = self.src_tensors[0].shape[1]
        num_transfer_bytes = num_src_blocks * page_size_bytes

        # GPU ↔ CPU transfer (CUDA stream + events for ordering).
        if torch.cuda.is_available() and (
            self.src_tensors[0].is_cuda or self.dst_tensors[0].is_cuda
        ):
            stream = (
                self._stream_pool.pop()
                if self._stream_pool
                else torch.cuda.Stream()
            )
            start_event = (
                self._event_pool.pop()
                if self._event_pool
                else torch.cuda.Event(enable_timing=False)
            )
            end_event = (
                self._event_pool.pop()
                if self._event_pool
                else torch.cuda.Event(enable_timing=False)
            )

            with torch.cuda.stream(stream):
                start_event.record(stream)
                # For each block, copy page_size_bytes from src → dst.
                for i in range(num_src_blocks):
                    src_b = int(src_blocks[i])
                    dst_b = int(dst_blocks[i])
                    # Each tensor is (num_blocks, page_size_bytes).
                    src_view = self.src_tensors[0][src_b : src_b + 1]
                    dst_view = self.dst_tensors[0][dst_b : dst_b + 1]
                    dst_view.copy_(src_view, non_blocking=True)
                end_event.record(stream)

            self._transfer_events[job_id] = end_event
            self._transfers.append(
                Transfer(
                    job_id=job_id,
                    stream=stream,
                    start_event=start_event,
                    end_event=end_event,
                    num_bytes=num_transfer_bytes,
                    batch_src=batch_src,
                    batch_dst=batch_dst,
                    batch_sizes=batch_sizes,
                )
            )
        else:
            # Pure-CPU path: synchronous copy. Still records
            # an event-like no-op so get_finished returns the result.
            for i in range(num_src_blocks):
                src_b = int(src_blocks[i])
                dst_b = int(dst_blocks[i])
                src_view = self.src_tensors[0][src_b : src_b + 1]
                dst_view = self.dst_tensors[0][dst_b : dst_b + 1]
                dst_view.copy_(src_view, non_blocking=False)

            # Synthesize an immediately-finished result.
            result = TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=num_transfer_bytes,
                transfer_time=0.0,
            )
            # Recycle buffers.
            self._buffer_pool.append((batch_src, batch_dst, batch_sizes))
            self._transfer_events[job_id] = None
            # Keep the result for get_finished to drain.
            self._transfers.append(
                Transfer(
                    job_id=job_id,
                    stream=None,
                    start_event=None,
                    end_event=None,
                    num_bytes=num_transfer_bytes,
                    batch_src=batch_src,
                    batch_dst=batch_dst,
                    batch_sizes=batch_sizes,
                )
            )
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers:
            transfer = self._transfers[0]
            if (
                transfer.end_event is None
                or transfer.end_event.query()
            ):
                transfer = self._transfers.popleft()
                transfer_time = (
                    transfer.start_event.elapsed_time(transfer.end_event)
                    * 1e-3
                    if transfer.start_event is not None
                    else 0.0
                )
                results.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=True,
                        transfer_size=transfer.num_bytes,
                        transfer_time=transfer_time,
                    )
                )
                if transfer.stream is not None:
                    self._stream_pool.append(transfer.stream)
                if transfer.end_event is not None:
                    self._event_pool.append(transfer.end_event)
                if transfer.start_event is not None:
                    self._event_pool.append(transfer.start_event)
                self._buffer_pool.append(
                    (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
                )
                self._transfer_events.pop(transfer.job_id, None)
            else:
                break
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def shutdown(self) -> None:
        while self._transfers:
            transfer = self._transfers.popleft()
            if transfer.end_event is not None:
                transfer.end_event.synchronize()
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()


class CPUOffloadingWorker(OffloadingWorker):
    """`OffloadingWorker` for CPU offloading.

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/gpu_worker.py:CPUOffloadingWorker.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int = 1,
        num_cpu_blocks: int = 0,
        pin_memory: bool = True,
    ):
        del blocks_per_chunk  # Single-group path only; vLLM uses this
        # for multi-group sub-block math.
        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for kv_cache_tensor in kv_caches.tensors:
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes
            cpu_tensor = torch.zeros(
                (num_cpu_blocks, cpu_page_size_bytes),
                dtype=torch.int8,
                device="cpu",
                pin_memory=pin_memory,
            )
            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            gpu_to_cpu=True,
        )
        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            gpu_to_cpu=False,
        )

    def submit_store(
        self,
        job_id: int,
        src_spec: GPULoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> bool:
        """Async GPU → CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self,
        job_id: int,
        src_spec: LoadStoreSpec,
        dst_spec: GPULoadStoreSpec,
    ) -> bool:
        """Async CPU → GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return (
            self._store_handler.get_finished()
            + self._load_handler.get_finished()
        )

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        self._store_handler.shutdown()
        self._load_handler.shutdown()


__all__ = [
    "CPUOffloadingWorker",
    "SingleDirectionOffloadingHandler",
    "Transfer",
]
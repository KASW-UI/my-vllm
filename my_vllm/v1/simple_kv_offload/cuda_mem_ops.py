# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-level CUDA/HIP memory helpers: pinning and batch DMA transfers.

Mirrors ``vllm/v1/simple_kv_offload/cuda_mem_ops.py`` (subset).

C257 scope reductions (vs upstream):

* **No ``cuMemcpyBatchAsync`` / ``hipMemcpyBatchAsync``**. The CUDA
  driver ``cuMemcpyBatchAsync`` API is CUDA 12.8+ and not available
  in every PyTorch install; the HIP equivalent requires ROCm 7.1+.
  Instead we use a per-block ``tensor.copy_()`` loop with the same
  ``BatchMemcpyParams`` shape so the worker-side code can stay
  unchanged.  Performance is lower than vLLM's batched DMA path but
  the wire format / lifecycle is preserved.

* **No ROCm path**: ``is_rocm()`` branch is preserved as a no-op
  ``_resolve_batch_memcpy`` stub so my-vllm's CUDA-only platform
  does not branch on ROCm.

* **Ctypes structures preserved** (``_CUmemcpyAttributes``,
  ``_CUmemLocation``) so any future migration to the real batch
  memcpy can be a drop-in replacement.
"""

from __future__ import annotations

import ctypes
from typing import Any, NamedTuple

import numpy as np
import torch

from my_vllm.platforms import current_platform

# CUmemcpySrcAccessOrder values (CUDA driver API). STREAM(1): source read in
# stream order, safe when the source may still be written. ANY(3): source may
# be read early, only safe for a stable source (e.g. pinned host memory).
CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1
CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3


def pin_tensor(tensor: torch.Tensor) -> None:
    """Pin a CPU tensor via cudaHostRegister.

    Mirrors ``vllm/v1/simple_kv_offload/cuda_mem_ops.py:pin_tensor``.
    This bypasses PyTorch's ``CUDACachingHostAllocator`` which rounds
    every ``pin_memory=True`` allocation up to the next power of 2
    (e.g. 100 GB becomes 128 GB).

    On non-CUDA platforms or when ``torch.cuda`` is unavailable, this
    is a no-op (the tensor stays pageable).
    """
    if not torch.cuda.is_available():
        return
    err = torch.cuda.cudart().cudaHostRegister(
        tensor.data_ptr(), tensor.nbytes, 0
    )
    if err.value != 0:
        raise RuntimeError(f"cudaHostRegister failed: {err}")


class _CUmemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("id", ctypes.c_int)]


class _CUmemcpyAttributes(ctypes.Structure):
    _fields_ = [
        ("srcAccessOrder", ctypes.c_uint),
        ("srcLocHint", _CUmemLocation),
        ("dstLocHint", _CUmemLocation),
        ("flags", ctypes.c_uint),
    ]


def _simple_torch_copy(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    src_block_ids: list[int],
    dst_block_ids: list[int],
) -> int:
    """Per-block ``tensor.copy_()`` for the simple-torch DMA path.

    Walks ``src_caches`` / ``dst_caches`` (one tensor per layer) and
    performs ``dst_caches[name][dst_id].copy_(src_caches[name][src_id])``
    for each (src_id, dst_id) pair.  Mirrors the wire-format expected
    by ``cuMemcpyBatchAsync`` but uses torch's tensor copy directly.

    Returns 0 on success; raises ``RuntimeError`` if the copy fails.
    """
    if len(src_block_ids) != len(dst_block_ids):
        raise RuntimeError(
            f"src/dst block-id count mismatch: "
            f"{len(src_block_ids)} vs {len(dst_block_ids)}"
        )
    for s_id, d_id in zip(src_block_ids, dst_block_ids):
        for name, src_tensor in src_caches.items():
            dst_tensor = dst_caches[name]
            if s_id < 0 or d_id < 0:
                continue
            src_view = src_tensor[s_id]
            dst_view = dst_tensor[d_id]
            dst_view.copy_(src_view)
    return 0


class BatchMemcpyParams(NamedTuple):
    """Pre-resolved DMA parameters for a single direction.

    Mirrors ``vllm/v1/simple_kv_offload/cuda_mem_ops.py:BatchMemcpyParams``.

    C257 deviation: holds a reference to the underlying ``src_caches``
    / ``dst_caches`` dict so ``_simple_torch_copy`` can perform the
    per-layer ``tensor.copy_()``.  The ``attrs`` / ``attrs_idx``
    fields are preserved as a driver-API shim but unused.
    """

    src_caches: dict[str, torch.Tensor]
    dst_caches: dict[str, torch.Tensor]
    src_bases: np.ndarray  # [num_layers] uint64 — data_ptr per layer
    dst_bases: np.ndarray  # [num_layers] uint64
    bpb: np.ndarray  # [num_layers] uint64 — bytes per block
    num_layers: int
    attrs: _CUmemcpyAttributes
    attrs_idx: ctypes.c_size_t
    fail_idx: ctypes.c_size_t
    stream_handle: int  # raw cudaStream_t / CUstream


def build_params(
    src_caches: dict[str, torch.Tensor],
    dst_caches: dict[str, torch.Tensor],
    stream: torch.cuda.Stream | Any,
    src_access_order: int = CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
) -> BatchMemcpyParams:
    """Pre-compute the per-layer DMA bookkeeping.

    Mirrors ``vllm/v1/simple_kv_offload/cuda_mem_ops.py:build_params``.
    Returns a ``BatchMemcpyParams`` whose ``copy_blocks()`` performs
    per-block ``tensor.copy_()`` (C257 scope reduction).
    """
    assert list(src_caches.keys()) == list(dst_caches.keys()), (
        "src/dst cache key order mismatch"
    )
    src_tensors = list(src_caches.values())
    dst_tensors = list(dst_caches.values())

    src_bases, dst_bases, bpb = [], [], []
    for s, d in zip(src_tensors, dst_tensors):
        s_bpb = s.stride(0) * s.element_size()
        assert s_bpb == d.stride(0) * d.element_size(), (
            "src/dst byte-per-block mismatch"
        )
        src_bases.append(s.data_ptr())
        dst_bases.append(d.data_ptr())
        bpb.append(s_bpb)

    attrs = _CUmemcpyAttributes(srcAccessOrder=src_access_order)

    return BatchMemcpyParams(
        src_caches=src_caches,
        dst_caches=dst_caches,
        src_bases=np.array(src_bases, dtype=np.uint64),
        dst_bases=np.array(dst_bases, dtype=np.uint64),
        bpb=np.array(bpb, dtype=np.uint64),
        num_layers=len(src_tensors),
        attrs=attrs,
        attrs_idx=ctypes.c_size_t(0),
        fail_idx=ctypes.c_size_t(0),
        stream_handle=getattr(stream, "cuda_stream", 0),
    )


def copy_blocks(
    src_block_ids: list[int],
    dst_block_ids: list[int],
    params: BatchMemcpyParams,
) -> None:
    """Copy blocks via the simple-torch DMA path.

    Mirrors ``vllm/v1/simple_kv_offload/cuda_mem_ops.py:copy_blocks``.
    C257 deviation: delegates to ``_simple_torch_copy`` (per-block
    ``tensor.copy_()``) instead of ``cuMemcpyBatchAsync``.
    """
    err = _simple_torch_copy(
        params.src_caches,
        params.dst_caches,
        src_block_ids,
        dst_block_ids,
    )
    if err != 0:
        raise RuntimeError(
            f"batch memcpy failed: err={err} failIdx={params.fail_idx.value}"
        )

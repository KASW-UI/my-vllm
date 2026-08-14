"""my_vllm/distributed/custom_all_reduce.py — Python wrapper for the
CUDA custom_all_reduce kernels (csrc/custom_all_reduce.cu).

Implements the vLLM-canonical API surface (init_custom_ar / all_reduce /
dispose / register_buffer / get_handle / open_mem_handle) for the
CUDA IPC-based ring all-reduce.  In production vLLM this is used for
same-host multi-GPU all-reduce to avoid NCCL's small-message latency.

Usage (multi-process, same host):

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Each rank allocates its own scratch buffer + signal buffer.
    scratch = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    handle_bytes = _C.get_handle()  # local IPC handle (cudaIpcMemHandle_t)
    # ... broadcast handle_bytes to all peers ...
    peer_handles = [broadcast(...) for _ in range(world_size)]

    # Open peer IPC handles (returns local pointers into peer GPU memory).
    peer_ptrs = [_C.open_mem_handle(h) for h in peer_handles]

    # Init the CustomAllreduce instance.
    ar = _C.init_custom_ar(peer_ptrs, scratch, rank, fully_connected=True)

    # Run an all-reduce.
    inp = torch.randn(1024, dtype=torch.float32, device="cuda")
    out = torch.empty_like(inp)
    _C.all_reduce(ar, inp, out)

    # Cleanup.
    _C.dispose(ar)

The my-vllm implementation is a simplified single-pass ring (vs vLLM's
two-shot optimization); sufficient for the JD "分布式通信算子原理"
requirement and for same-host multi-GPU benchmarks.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import torch

from my_vllm._custom_ops import _C_CACHE_OPS

logger = logging.getLogger(__name__)


class CustomAllreduce:
    """Python wrapper around the CUDA custom_all_reduce extension.

    Attributes:
        handle (int): The opaque `fptr_t` returned by `init_custom_ar`.
        rank (int): This worker's rank (0..world_size-1).
        world_size (int): Total number of participating ranks.
        scratch (torch.Tensor): Pre-allocated GPU scratch buffer for the
            allreduce kernel.  Owned by the caller (we don't free).
    """

    def __init__(
        self,
        peer_ipc_handles: Sequence[bytes],
        scratch: torch.Tensor,
        rank: int,
        fully_connected: bool = True,
    ):
        if _C_CACHE_OPS is None:
            raise RuntimeError(
                "custom_all_reduce requires the compiled my_vllm._C "
                "extension; rebuild with `MAX_JOBS=3 python setup.py "
                "build_ext --inplace`."
            )
        if scratch.dtype != torch.uint8:
            raise ValueError(
                f"scratch must be uint8 (signal buffer + ring buffer), "
                f"got {scratch.dtype}"
            )
        if not scratch.is_cuda:
            raise ValueError("scratch must be a CUDA tensor")
        if rank < 0 or rank >= len(peer_ipc_handles):
            raise ValueError(
                f"rank={rank} out of range [0, {len(peer_ipc_handles)})"
            )

        # Open peer IPC handles → local GPU pointers (int64).
        peer_ptrs: list[int] = []
        for i, h in enumerate(peer_ipc_handles):
            if i == rank:
                # The local rank's own scratch doesn't need an open
                # handle (it's already a local pointer).  We still
                # need to pass a valid pointer for the C++ API; using
                # the scratch's data_ptr works.
                peer_ptrs.append(scratch.data_ptr())
            else:
                # h is a torch.Tensor (the bytes from get_handle()).
                ptr = _C_CACHE_OPS.open_mem_handle(h)
                peer_ptrs.append(ptr)

        self._ar = _C_CACHE_OPS.init_custom_ar(
            peer_ptrs, scratch, rank, fully_connected,
        )
        self.rank = rank
        self.world_size = len(peer_ipc_handles)
        self.scratch = scratch

    def all_reduce(self, inp: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Out-of-place all-reduce.

        Args:
            inp: [numel] tensor on CUDA.  Must match the scratch dtype.
            out: [numel] tensor on CUDA; if None, a new tensor is allocated.

        Returns:
            ``out`` (or a newly allocated tensor) with the reduced result.
        """
        if out is None:
            out = torch.empty_like(inp)
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(
                f"custom_all_reduce supports fp32/fp16/bf16, got {inp.dtype}"
            )
        _C_CACHE_OPS.all_reduce(self._ar, inp, out, 0, 0)
        return out

    def dispose(self) -> None:
        """Free the underlying CustomAllreduce instance."""
        if self._ar is not None:
            _C_CACHE_OPS.dispose(self._ar)
            self._ar = None

    def __del__(self):
        # Best-effort cleanup; if interpreter is shutting down the
        # _C extension may already be unloaded.
        try:
            self.dispose()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level helpers (mirroring vLLM's communication_op.py surface)
# ---------------------------------------------------------------------------


def get_local_ipc_handle() -> torch.Tensor:
    """Allocate a Signal buffer on the current device and return its
    IPC memhandle as a uint8 tensor.

    The returned tensor can be broadcast to peer ranks (e.g. via
    torch.distributed.broadcast) and then opened via open_ipc_handle.
    """
    if _C_CACHE_OPS is None:
        raise RuntimeError("my_vllm._C not compiled")
    return _C_CACHE_OPS.get_handle()


def open_ipc_handle(mem_handle: torch.Tensor) -> int:
    """Open a remote IPC memhandle, returning a local GPU pointer.

    The returned pointer must be passed to ``CustomAllreduce.__init__``
    as part of ``peer_ipc_handles``.
    """
    if _C_CACHE_OPS is None:
        raise RuntimeError("my_vllm._C not compiled")
    return _C_CACHE_OPS.open_mem_handle(mem_handle)


def meta_size() -> int:
    """Size of the Signal struct (bytes).  Exposed for tooling that
    wants to allocate Signal buffers in user-space."""
    if _C_CACHE_OPS is None:
        raise RuntimeError("my_vllm._C not compiled")
    return _C_CACHE_OPS.meta_size()


__all__ = [
    "CustomAllreduce",
    "get_local_ipc_handle",
    "open_ipc_handle",
    "meta_size",
]

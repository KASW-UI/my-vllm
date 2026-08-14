"""my_vllm/distributed/device_communicators/cuda_communicator.py

CUDA-backed ``DeviceCommunicatorBase`` implementation.

C240: extracted from ``vllm/distributed/device_communicators/cuda_communicator.py``
(745 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``CudaCommunicator`` keeps the AG/RS family of collective primitives
    (``all_gather`` / ``all_gatherv`` / ``reduce_scatter`` /
    ``reduce_scatterv``) and the ``dispatch_router_logits`` /
    ``dispatch`` / ``combine`` shims that delegate to
    ``all2all_manager``.
  * NVLS symmetric memory (``_all_gather_symm_mem`` /
    ``_reduce_scatter_symm_mem`` / ``_get_symm_scratch`` /
    ``pynccl_allocator``) is omitted — the C-layer PyNCCL allocator
    is not vendored; the standard PyTorch ``torch.distributed``
    path is sufficient for my-vllm's correctness verification.
  * FlashInfer ``fi_ar_comm``, AITER ``aiter_ar_comm``, custom
    ``ca_comm`` (CUDA custom all-reduce) — omitted.
  * Custom all-reduce integration with C214's ``csrc/custom_all_reduce
    .cu`` extension is deferred (the vLLM CudaCommunicator has a
    full pynccl/custom-ar interop that my-vllm doesn't need for
    the All-to-All path).

References:
  * vLLM cuda_communicator.py:
    ``vllm/distributed/device_communicators/cuda_communicator.py:574-738``
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from my_vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class CudaCommunicator(DeviceCommunicatorBase):
    """CUDA-backed communicator that routes collectives through a ``GroupCoordinator``.

    The constructor takes a ``GroupCoordinator`` (instead of raw
    ``ProcessGroup`` objects) so that the actual NCCL/Gloo
    primitives (already implemented in ``parallel_state.GroupCoordinator``)
    are reused.  This keeps the CudaCommunicator surface thin:
    it only adds the ``DeviceCommunicatorBase`` contract on top.
    """

    def __init__(
        self,
        cpu_group: Any,
        device: torch.device | None = None,
        device_group: Any | None = None,
        unique_name: str = "",
        group_coordinator: Any | None = None,
    ) -> None:
        super().__init__(
            cpu_group=cpu_group,
            device=device,
            device_group=device_group,
            unique_name=unique_name,
        )
        # The ``group_coordinator`` is the GroupCoordinator that
        # implements the actual collective primitives.  If ``None``,
        # we fall back to ``torch.distributed`` directly (sufficient
        # for ``world_size == 1`` callers).
        self.group_coordinator = group_coordinator
        if group_coordinator is not None:
            self.rank = group_coordinator.rank_in_group
            self.world_size = group_coordinator.world_size

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Variable-size all-gather via the bound GroupCoordinator.

        ``dim != 0`` is rejected because the AG/RS path is
        column-major along the first axis only (matches vLLM).
        """
        if dim != 0:
            raise NotImplementedError(
                "CudaCommunicator.all_gatherv only supports dim=0 "
                "(the AgRsAll2AllManager path)."
            )
        if self.world_size == 1:
            return input_
        if self.group_coordinator is None:
            raise RuntimeError(
                "CudaCommunicator requires a GroupCoordinator for "
                "world_size > 1; got None."
            )
        return self.group_coordinator.all_gatherv(input_, dim=dim, sizes=sizes)

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        if dim != 0:
            raise NotImplementedError(
                "CudaCommunicator.reduce_scatterv only supports dim=0 "
                "(the AgRsAll2AllManager path)."
            )
        if self.group_coordinator is None:
            raise RuntimeError(
                "CudaCommunicator requires a GroupCoordinator for "
                "world_size > 1; got None."
            )
        return self.group_coordinator.reduce_scatterv(
            input_, dim=0, sizes=sizes
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if self.group_coordinator is not None:
            return self.group_coordinator.all_reduce(input_)
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def reduce_scatter(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if self.group_coordinator is not None:
            return self.group_coordinator.reduce_scatter(input_, dim=dim)
        return super().reduce_scatter(input_, dim=dim)

    def all_gather(
        self, input_: torch.Tensor, dim: int = -1
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if self.group_coordinator is not None:
            return self.group_coordinator.all_gather(input_, dim=dim)
        return super().all_gather(input_, dim=dim)

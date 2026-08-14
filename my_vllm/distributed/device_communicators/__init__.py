"""my_vllm/distributed/device_communicators — Per-device communication primitives.

C240: introduced for the All-to-All (MoE EP) path.

Mirrors ``vllm/distributed/device_communicators/`` but ships only the
CUDA pieces used by ``AgRsAll2AllManager`` (NCCL/Gloo all-gather +
reduce-scatter).  CPU/ROCm/XPU/TPU backends are not vendored; the
single-platform scope is documented in ``AGENTS.md`` JD-driven scope.

Public surface
==============

  * ``base_device_communicator.All2AllManagerBase`` — abstract base for
    EP/DP all-to-all managers (dispatch + dispatch_router_logits +
    combine).
  * ``base_device_communicator.DeviceCommunicatorBase`` — per-device
    communicator façade (``all_gather`` / ``all_gatherv`` /
    ``reduce_scatter`` / ``reduce_scatterv`` / ``dispatch`` /
    ``combine``).
  * ``all2all.AgRsAll2AllManager`` — concrete AllGather + ReduceScatter
    all-to-all (the reference naive backend, mapped to NCCL ``all_gatherv``
    / ``reduce_scatterv``).
  * ``cuda_communicator.CudaCommunicator`` — CUDA-backed
    ``DeviceCommunicatorBase`` that delegates to ``GroupCoordinator``
    (NCCL/Gloo) for the actual collectives.
"""

from __future__ import annotations

from my_vllm.distributed.device_communicators.all2all import (
    AgRsAll2AllManager,
    DeepEPAll2AllManagerBase,
    DeepEPHTAll2AllManager,
    DeepEPLLAll2AllManager,
    DeepEPV2All2AllManager,
    FlashInferNVLinkOneSidedAll2AllManager,
    FlashInferNVLinkTwoSidedAll2AllManager,
    MoriAll2AllManager,
    NixlEPAll2AllManager,
)
from my_vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    Cache,
    DeviceCommunicatorBase,
)
from my_vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)


__all__ = [
    "All2AllManagerBase",
    "Cache",
    "DeviceCommunicatorBase",
    "AgRsAll2AllManager",
    "DeepEPAll2AllManagerBase",
    "DeepEPHTAll2AllManager",
    "DeepEPLLAll2AllManager",
    "DeepEPV2All2AllManager",
    "FlashInferNVLinkOneSidedAll2AllManager",
    "FlashInferNVLinkTwoSidedAll2AllManager",
    "MoriAll2AllManager",
    "NixlEPAll2AllManager",
    "CudaCommunicator",
]

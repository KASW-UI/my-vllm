"""my_vllm/distributed/device_communicators — Per-device communication primitives.

C240: introduced for the All-to-All (MoE EP) path.
C241: adds ``SymmMemCommunicator`` — NVLS-accelerated collectives
via ``torch.distributed._symmetric_memory``.
C242: adds ``ShmRingBuffer`` + ``MessageQueue`` (tensor /
object broadcast over shared memory) and
``SingleWriterShmObjectStorage`` (key-value store over shared
memory).

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
    (NCCL/Gloo) for the actual collectives.  C241: optionally uses
    ``SymmMemCommunicator`` for the all-reduce fast path when NVLS
    is available.
  * ``symm_mem.SymmMemCommunicator`` — NVLS-accelerated all-reduce
    via ``torch.distributed._symmetric_memory``.
  * ``all_reduce_utils`` — sizing tables + ``should_nccl_symm_mem_*``
    selection heuristics.
  * ``shm_broadcast.ShmRingBuffer`` + ``MessageQueue`` — same-host
    broadcast over shared memory.
  * ``shm_object_storage.SingleWriterShmObjectStorage`` — same-host
    key-value store over a shared-memory ring buffer.
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
from my_vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
    should_nccl_symm_mem_ag_rs,
    should_nccl_symm_mem_allreduce,
)
from my_vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
    Cache,
    DeviceCommunicatorBase,
)
from my_vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)
from my_vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
    ShmRingBuffer,
    memory_fence,
)
from my_vllm.distributed.device_communicators.shm_object_storage import (
    ObjectSerde,
    PickleSerde,
    ShmObjectStorageHandle,
    SingleWriterShmObjectStorage,
    SingleWriterShmRingBuffer,
)
from my_vllm.distributed.device_communicators.symm_mem import (
    SymmMemCommunicator,
    is_symmetric_memory_available,
)


__all__ = [
    "All2AllManagerBase",
    "AgRsAll2AllManager",
    "Cache",
    "CudaCommunicator",
    "CUSTOM_ALL_REDUCE_MAX_SIZES",
    "DeepEPAll2AllManagerBase",
    "DeepEPHTAll2AllManager",
    "DeepEPLLAll2AllManager",
    "DeepEPV2All2AllManager",
    "DeviceCommunicatorBase",
    "FlashInferNVLinkOneSidedAll2AllManager",
    "FlashInferNVLinkTwoSidedAll2AllManager",
    "Handle",
    "MessageQueue",
    "MoriAll2AllManager",
    "NCCL_SYMM_MEM_ALL_REDUCE_CONFIG",
    "NixlEPAll2AllManager",
    "ObjectSerde",
    "PickleSerde",
    "SYMM_MEM_ALL_REDUCE_MAX_SIZES",
    "ShmObjectStorageHandle",
    "ShmRingBuffer",
    "SingleWriterShmObjectStorage",
    "SingleWriterShmRingBuffer",
    "SymmMemCommunicator",
    "is_symmetric_memory_available",
    "memory_fence",
    "should_nccl_symm_mem_ag_rs",
    "should_nccl_symm_mem_allreduce",
]

"""Weight transfer engines for syncing model weights from trainers
to inference workers.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Subpackages:

* :mod:`my_vllm.distributed.weight_transfer.base` — abstract base
  classes, dataclasses, and the :class:`VLLMWeightSyncClient` protocol.
* :mod:`my_vllm.distributed.weight_transfer.nccl_common` — shared NCCL
  process-group initialization helpers (dense + sparse engines).
* :mod:`my_vllm.distributed.weight_transfer.nccl_engine` — dense NCCL
  broadcast engine (the canonical cross-process backend).
* :mod:`my_vllm.distributed.weight_transfer.sparse_nccl_engine` —
  in-place flat-index patch engine (sparse RLHF updates).
* :mod:`my_vllm.distributed.weight_transfer.ipc_engine` — CUDA IPC
  engine (same-node multi-process).
* :mod:`my_vllm.distributed.weight_transfer.packed_tensor` — pack /
  unpack + double-buffered NCCL / IPC producer / consumer.
* :mod:`my_vllm.distributed.weight_transfer.factory` — lazy-loaded
  registry mapping ``WeightTransferConfig.backend`` strings to engine
  classes.
* :mod:`my_vllm.distributed.weight_transfer.clients` — built-in
  :class:`VLLMWeightSyncClient` implementations
  (only :class:`InProcessVLLMWeightSyncClient` in my-vllm).
"""

from my_vllm.distributed.weight_transfer.base import (
    ModuleSource,
    ParamMeta,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
    materialize_full_tensor,
)
from my_vllm.distributed.weight_transfer.clients import InProcessVLLMWeightSyncClient
from my_vllm.distributed.weight_transfer.factory import (
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from my_vllm.distributed.weight_transfer.ipc_engine import (
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
    IPCTrainerSendWeightsArgs,
)
from my_vllm.distributed.weight_transfer.nccl_common import (
    NCCLWeightTransferInitInfo,
    stateless_init_process_group,
    trainer_init,
    worker_init_process_group,
)
from my_vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferUpdateInfo,
)
from my_vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)

__all__ = [
    # base
    "ParamMeta",
    "WeightSource",
    "ModuleSource",
    "WeightTransferEngine",
    "TrainerWeightTransferEngine",
    "WeightTransferInitInfo",
    "WeightTransferUpdateInfo",
    "VLLMWeightSyncClient",
    "materialize_full_tensor",
    # clients
    "InProcessVLLMWeightSyncClient",
    # factory
    "WeightTransferEngineFactory",
    "WeightTransferTrainerFactory",
    # NCCL (dense)
    "NCCLWeightTransferEngine",
    "NCCLWeightTransferInitInfo",
    "NCCLWeightTransferUpdateInfo",
    "NCCLTrainerSendWeightsArgs",
    "stateless_init_process_group",
    "worker_init_process_group",
    "trainer_init",
    # NCCL (sparse)
    "SparseNCCLWeightTransferEngine",
    "SparseNCCLWeightTransferUpdateInfo",
    "SparseWeightPatch",
    # IPC
    "IPCWeightTransferEngine",
    "IPCWeightTransferInitInfo",
    "IPCWeightTransferUpdateInfo",
    "IPCTrainerSendWeightsArgs",
]

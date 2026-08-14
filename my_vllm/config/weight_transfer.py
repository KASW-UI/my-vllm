"""WeightTransferConfig — backend selection for trainer→worker weight sync.

C243: closes the A-level "跨节点 Weight Transfer" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors ``vllm/config/weight_transfer.py:WeightTransferConfig`` 1:1 —
a single ``backend`` field that names the registered engine. The actual
engine wiring (NCCL / IPC / sparse-NCCL) lives in
``my_vllm/distributed/weight_transfer/factory.py``.

Scope reductions vs vLLM:

* No FSDP-specific extensions (vLLM has DTensor-aware materialization in
  ``materialize_full_tensor``; my-vllm keeps the helper but it's a plain
  identity for non-DTensor inputs).
* No layerwise-reload option (my-vllm doesn't vendor
  ``vllm/model_executor/model_loader/reload`` — checkpoint streaming is
  not in scope for OPT-125m-sized models).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class WeightTransferConfig:
    """Configuration for weight transfer during RL training.

    Mirrors vLLM ``vllm/config/weight_transfer.py:WeightTransferConfig``.
    """

    backend: Literal["nccl", "ipc", "sparse_nccl"] | str = "nccl"
    """The backend to use for weight transfer. Validated against the
    :class:`my_vllm.distributed.weight_transfer.factory.WeightTransferEngineFactory`
    registry at engine creation time."""

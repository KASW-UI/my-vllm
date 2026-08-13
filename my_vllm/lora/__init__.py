"""LoRA package — adapters + layers + forward ops.

Mirrors vLLM V1 lora/.

Public API:
  - LoRARequest — request-side identifier for an adapter.
  - BaseLayer / BaseLinearLayer — LoRA-augmented layer base classes.
  - LoRAWeights — one adapter's A/B parameter pair.
  - lora_forward_dense / lora_forward_dense_inplace — pure-torch
    forward helpers.

C144: re-export the foundation introduced in this stage. The
ColumnParallel / RowParallel / MoE LoRA variants + the
LoRAModel + ModelManager + WorkerManager pipeline land in
后续 commits.
"""

from my_vllm.lora.layers.base import BaseLayer, LoRAWeights
from my_vllm.lora.layers.base_linear import BaseLinearLayer
from my_vllm.lora.ops.torch_ops import (
    lora_forward_dense,
    lora_forward_dense_inplace,
)
from my_vllm.lora.request import LoRARequest

__all__ = [
    "BaseLayer",
    "BaseLinearLayer",
    "LoRARequest",
    "LoRAWeights",
    "lora_forward_dense",
    "lora_forward_dense_inplace",
]
"""LoRA package — adapters + layers + forward ops.

Mirrors vLLM V1 lora/.

Public API:
  - LoRARequest — request-side identifier for an adapter.
  - BaseLayer / BaseLinearLayer — LoRA-augmented layer base classes.
  - ColumnParallelLinearLoRA / RowParallelLinearLoRA — TP-aware
    LoRA Linear variants.
  - FusedMoEWithLoRA — MoE LoRA skeleton.
  - LoRAWeights — one adapter's A/B parameter pair.
  - lora_forward_dense / lora_forward_dense_inplace — pure-torch
    forward helpers.
  - LoRAModel / LoRAModelManager / LoRAWorkerManager — model +
    manager pipeline (C146).
"""

from my_vllm.lora.layers.base import BaseLayer, LoRAWeights
from my_vllm.lora.layers.base_linear import BaseLinearLayer
from my_vllm.lora.layers.column_parallel_linear import ColumnParallelLinearLoRA
from my_vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from my_vllm.lora.layers.row_parallel_linear import RowParallelLinearLoRA
from my_vllm.lora.lora_model import LoRAModel
from my_vllm.lora.model_manager import LoRAModelManager
from my_vllm.lora.ops.torch_ops import (
    lora_forward_dense,
    lora_forward_dense_inplace,
)
from my_vllm.lora.request import LoRARequest
from my_vllm.lora.worker_manager import LoRAWorkerManager

__all__ = [
    "BaseLayer",
    "BaseLinearLayer",
    "ColumnParallelLinearLoRA",
    "FusedMoEWithLoRA",
    "LoRAModel",
    "LoRAModelManager",
    "LoRARequest",
    "LoRAWeights",
    "LoRAWorkerManager",
    "RowParallelLinearLoRA",
    "lora_forward_dense",
    "lora_forward_dense_inplace",
]
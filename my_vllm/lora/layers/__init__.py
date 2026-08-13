"""LoRA layers — base + per-host-type wrappers.

Mirrors vllm/vllm/lora/layers/__init__.py (subset).

Public API:
  - BaseLayer / BaseLinearLayer — common base classes.
  - ColumnParallelLinearLoRA — wraps ColumnParallelLinear
    (also covers QKVParallelLinear / MergedColumnParallelLinear
    since they extend ColumnParallelLinear).
  - RowParallelLinearLoRA — wraps RowParallelLinear (adds the
    TP all_reduce to the LoRA delta).
  - FusedMoEWithLoRA — MoE LoRA skeleton.

The wrap-from-base-class lookup is centralised in
`my_vllm.lora.model_manager.LoRAModelManager.create_lora_modules`,
which uses `LORA_LAYER_TYPE_MAP` below to pick the right wrapper
for each `nn.Linear`-style module it encounters.
"""

from my_vllm.lora.layers.base import BaseLayer, LoRAWeights
from my_vllm.lora.layers.base_linear import BaseLinearLayer
from my_vllm.lora.layers.column_parallel_linear import ColumnParallelLinearLoRA
from my_vllm.lora.layers.fused_moe import FusedMoEWithLoRA
from my_vllm.lora.layers.row_parallel_linear import RowParallelLinearLoRA

__all__ = [
    "BaseLayer",
    "BaseLinearLayer",
    "ColumnParallelLinearLoRA",
    "FusedMoEWithLoRA",
    "LoRAWeights",
    "RowParallelLinearLoRA",
]
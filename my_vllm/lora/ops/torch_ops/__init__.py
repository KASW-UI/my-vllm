"""LoRA torch ops package — pure-torch forward (Dense Linear path).

Mirrors vLLM V1 lora/ops/torch_ops/.

C144: contains `lora_forward_dense(...)` + `lora_forward_dense_inplace(...)`
for the Dense Linear path. Re-exports them at the package level
so callers can write `from my_vllm.lora.ops.torch_ops import
lora_forward_dense`.
"""

from my_vllm.lora.ops.torch_ops.lora_ops import (
    lora_forward_dense,
    lora_forward_dense_inplace,
)

__all__ = [
    "lora_forward_dense",
    "lora_forward_dense_inplace",
]
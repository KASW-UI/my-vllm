"""LoRA-augmented Dense Linear layer.

Mirrors vLLM V1 lora/layers/base_linear.py:BaseLinearLayerWithLoRA.

vLLM's `BaseLinearLayerWithLoRA` is the abstract base for
LoRA-augmented `Linear` layers. Subclasses
(`ColumnParallelLinearLoRA`, `RowParallelLinearLoRA`, the
non-parallel `LinearLoRA`) implement the actual forward.

The forward pass:
  y = base_linear(x) + scale * (x @ A^T) @ B^T
where:
  base_linear is the host layer's `.forward(x)`
  A is `[rank, in_features]`, B is `[out_features, rank]`
  scale = lora_alpha / lora_rank

C144: minimal subset — `BaseLinearLayer(BaseLayer)` with the
`apply_lora(out, x)` hook (called after the base forward to
add the LoRA delta) + a `forward(x)` that does both halves.
Uses `lora_ops.lora_forward_dense_inplace(...)` for the delta.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from my_vllm.lora.layers.base import BaseLayer
from my_vllm.lora.ops.torch_ops.lora_ops import lora_forward_dense_inplace

logger = logging.getLogger(__name__)


class BaseLinearLayer(BaseLayer):
    """Abstract base for LoRA-augmented Linear layers.

    Mirrors vLLM V1 lora/layers/base_linear.py:BaseLinearLayerWithLoRA.
    """

    def __init__(
        self,
        base_layer: Any,
        max_loras: int = 1,
        max_lora_rank: int = 16,
    ):
        super().__init__(
            base_layer=base_layer,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
        )

    def apply_lora(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Add the LoRA delta to `out` (in-place).

        Mirrors vLLM V1 lora/layers/base_linear.py:
        BaseLinearLayerWithLoRA.apply_lora.

        C144: minimal — uses `lora_forward_dense_inplace` for the
        delta. Active adapter is read from `self.active_lora_int_id`;
        if no adapter is set, this is a no-op.
        """
        lora_int_id = self.active_lora_int_id
        if lora_int_id is None:
            return out
        weights = self._adapters.get(lora_int_id)
        if weights is None:
            return out
        return lora_forward_dense_inplace(
            out=out,
            x=x,
            base_weight=self._base_weight_for_lora(),
            lora_a=weights.lora_a,
            lora_b=weights.lora_b,
            scaling=weights.scaling,
            base_bias=None,
        )

    def _base_weight_for_lora(self) -> torch.Tensor:
        """Return the base layer's weight matrix.

        Subclasses override this to return the partitioned weight
        for TP-aware layers (ColumnParallel / RowParallel).
        """
        return self.base_layer.weight

    def forward(self, x: torch.Tensor):
        """Compute `y = base_linear(x) + scale * (x @ A^T) @ B^T`.

        C147: handle the `(output, output_bias)` tuple return from
        vLLM-shaped ColumnParallel / RowParallel layers (used
        by my-vllm's OPTForCausalLM's `qkv_proj, fc1, fc2,
        out_proj`). The LoRA delta is added in-place to the
        output tensor; the bias is returned unchanged.

        Returns:
          - `Tensor` if the host layer's `return_bias=False`.
          - `(output, output_bias)` tuple if `return_bias=True`.
            The OPT decoder code uses `out, _ = layer(x)` to
            discard the bias; LoRA wraps the returned output
            while preserving the tuple shape.
        """
        out = self.base_layer(x)
        if isinstance(out, tuple):
            output, output_bias = out
            output = self.apply_lora(output, x)
            return output, output_bias
        return self.apply_lora(out, x)


__all__ = ["BaseLinearLayer"]
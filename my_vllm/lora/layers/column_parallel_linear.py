"""LoRA-augmented ColumnParallelLinear — TP-aware LoRA variant.

Mirrors vLLM V1 lora/layers/column_parallel_linear.py:
ColumnParallelLinearLoRA / MergedColumnParallelLinearLoRA.

vLLM's ColumnParallelLinearLoRA partitions the LoRA A/B matrices
along the same axis as the host's ColumnParallelLinear: A is
full-size `[rank, in_features]` (input is replicated across TP
ranks), B is partitioned `[out_features // tp, rank]` (output
is split across TP ranks). The LoRA delta is computed in the
partitioned space; no extra all_reduce / all_gather needed.

C145: minimal subset — `ColumnParallelLinearLoRA(BaseLinearLayer)`
overrides `_base_weight_for_lora()` to return the host layer's
already-partitioned weight. No additional TP logic is needed
because the host's weight is already sharded; the LoRA delta
naturally lives in the same partitioned space.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from my_vllm.lora.layers.base_linear import BaseLinearLayer

logger = logging.getLogger(__name__)


class ColumnParallelLinearLoRA(BaseLinearLayer):
    """LoRA-augmented ColumnParallelLinear.

    Mirrors vLLM V1 lora/layers/column_parallel_linear.py:
    ColumnParallelLinearLoRA (subset).

    C145: the LoRA delta is computed in the same partitioned space
    as the host's output (B is sharded along the output axis,
    matching the host's `output_size // tp_size`). No
    all_reduce needed — the LoRA delta is per-rank.
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

    def _base_weight_for_lora(self) -> torch.Tensor:
        """Return the host's already-partitioned weight."""
        return self.base_layer.weight

    def apply_lora(
        self,
        out: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Add the LoRA delta to the partitioned output.

        Mirrors vLLM V1 lora/layers/column_parallel_linear.py:
        ColumnParallelLinearWithLoRA.apply_lora (subset).

        C145: minimal — no additional TP logic. The host's
        forward already shards the output, so the LoRA delta
        (computed from partitioned B) lives in the same
        partitioned output space.
        """
        return super().apply_lora(out, x)


__all__ = ["ColumnParallelLinearLoRA"]
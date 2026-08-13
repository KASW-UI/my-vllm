"""LoRA-augmented RowParallelLinear — TP-aware LoRA variant.

Mirrors vLLM V1 lora/layers/row_parallel_linear.py:
RowParallelLinearLoRA.

vLLM's RowParallelLinearLoRA partitions the LoRA A/B matrices
along the same axis as the host's RowParallelLinear: A is
partitioned `[rank, in_features // tp]` (input is split across
TP ranks), B is full-size `[out_features, rank]` (output is
replicated). The LoRA delta computation requires an
all_reduce to combine per-rank partial results — same as the
host's RowParallelLinear.

C145: minimal subset — `RowParallelLinearLoRA(BaseLinearLayer)`
overrides `_base_weight_for_lora()` and adds an `all_reduce`
on the LoRA delta. The all_reduce uses my-vllm's
`distributed.parallel_state._TP` singleton (the same
GroupCoordinator the host's `RowParallelLinear` uses).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from my_vllm.lora.layers.base_linear import BaseLinearLayer

logger = logging.getLogger(__name__)


class RowParallelLinearLoRA(BaseLinearLayer):
    """LoRA-augmented RowParallelLinear.

    Mirrors vLLM V1 lora/layers/row_parallel_linear.py:
    RowParallelLinearLoRA (subset).

    C145: minimal — A is partitioned along the input axis,
    matching the host's `input_size // tp_size`. The LoRA delta
    computed per-rank requires an all_reduce across the TP
    group; we delegate to the host's `_TP` GroupCoordinator.
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
        """Add the LoRA delta + all_reduce across TP group.

        C145: minimal — call super().apply_lora(...) to add the
        per-rank delta, then all_reduce on the result (matches
        the host RowParallelLinear's all_reduce pattern).
        """
        out = super().apply_lora(out, x)
        # all_reduce across TP group (no-op when tp_size == 1).
        from my_vllm.distributed.parallel_state import _TP
        if _TP is not None and _TP.world_size > 1:
            _TP.all_reduce(out)
        return out


__all__ = ["RowParallelLinearLoRA"]
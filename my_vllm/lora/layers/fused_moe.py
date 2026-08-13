"""LoRA-augmented FusedMoE — MoE LoRA variant.

Mirrors vLLM V1 lora/layers/fused_moe.py:
FusedMoEWithLoRA / ReplicatedFusedMoEWithLoRA / SharedFusedMoEWithLoRA.

vLLM's MoE LoRA path handles routed experts + shared experts
with per-expert LoRA A/B matrices (one set of A/B per `(adapter,
expert)` pair). The forward is more involved than the dense
path: each token's LoRA delta is computed from the expert it
was routed to, weighted by the routing probability.

C146: minimal subset — `FusedMoEWithLoRA` class skeleton + the
public API surface (`create_lora_weights`, `set_lora`,
`reset_lora`). The actual per-expert forward is out of scope
per docs/alignment-plan.md §8 ("Dense Linear 的一条 LoRA 前向
路径即可证明理解，不追求全 backend").

A follow-up wires the per-expert dispatch via the existing
`fused_moe` module's `RoutedExperts` / `SharedExperts` types.
"""

from __future__ import annotations

import logging
from typing import Any

from my_vllm.lora.layers.base import BaseLayer

logger = logging.getLogger(__name__)


class FusedMoEWithLoRA(BaseLayer):
    """LoRA-augmented FusedMoE.

    Mirrors vLLM V1 lora/layers/fused_moe.py:FusedMoEWithLoRA.

    C146: skeleton — constructor stores the host FusedMoE;
    `create_lora_weights` / `set_lora` / `reset_lora` mirror
    the BaseLayer API. The actual per-expert forward is not
    implemented (routing-aware dispatch deferred).
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

    def create_lora_weights(
        self,
        lora_int_id: int,
        rank: int,
        in_features: int,
        out_features: int,
        alpha: float = 1.0,
    ) -> None:
        """Stub — per-expert A/B dispatch deferred."""
        # In vLLM this iterates over `num_experts` and allocates
        # one A/B pair per expert. my-vllm's subset uses a single
        # (rank, in, out) shape shared across experts; wiring
        # through the expert routing is a follow-up.
        super().create_lora_weights(
            lora_int_id=lora_int_id,
            rank=rank,
            in_features=in_features,
            out_features=out_features,
            alpha=alpha,
        )


__all__ = ["FusedMoEWithLoRA"]
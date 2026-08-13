"""Router factory — aligns with vLLM create_fused_moe_router()."""

import torch

from .fused_moe_router import FusedMoERouter
from .fused_topk_router import FusedTopKRouter


def create_fused_moe_router(
    top_k: int,
    global_num_experts: int,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    scoring_func: str = "softmax",
    num_fused_shared_experts: int = 0,
    shared_expert_weight: float = 1.0,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    custom_routing_function=None,
    zero_expert_type: str | None = None,
    num_logical_experts: int | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> FusedMoERouter:
    """Create a MoE router — aligns with vLLM create_fused_moe_router()."""
    return FusedTopKRouter(
        top_k=top_k,
        global_num_experts=global_num_experts,
        scoring_func=scoring_func,
        renormalize=renormalize,
    )

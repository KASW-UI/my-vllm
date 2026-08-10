"""FusedMoE layer — aligns with vLLM FusedMoE (factory → MoERunner)."""

import torch
import torch.nn as nn

from my_vllm.model_executor.layers.fused_moe.fused_moe import ExpertMLP, fused_moe
from my_vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)


class FusedMoE(nn.Module):
    """MoE layer replacing a standard MLP block.

    Aligns with vLLM FusedMoE — creates router + experts, runs TopK routing.
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        self.router = FusedMoERouter(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_dim, intermediate_dim)
            for _ in range(num_experts)
        ])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights, expert_indices = self.router(x)
        return fused_moe(x, self.experts, routing_weights, expert_indices, self.top_k)

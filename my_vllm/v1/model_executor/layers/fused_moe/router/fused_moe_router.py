"""FusedMoERouter — TopK gating with load balancing. Aligns with vLLM FusedMoERouter."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedMoERouter(nn.Module):
    """Top-K gating router for MoE.

    Computes router logits, selects top_k experts per token,
    returns softmax-normalized routing weights and expert indices.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_logits = self.gate(x)
        routing_weights, expert_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        return routing_weights, expert_indices

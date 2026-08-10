"""Fused MoE kernel — Python fallback (Triton stub for C081).

Aligns with vLLM fused_moe_kernel.  Each expert computes independently;
outputs are combined by routing_weights and expert_indices.
"""

import torch
import torch.nn as nn


def fused_moe(
    x: torch.Tensor,
    experts: nn.ModuleList,
    routing_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    """Route tokens to experts and combine weighted outputs.

    Args:
        x: [num_tokens, hidden_dim]
        experts: ModuleList of ExpertMLP
        routing_weights: [num_tokens, top_k]
        expert_indices: [num_tokens, top_k]
        top_k: number of experts per token

    Returns:
        output: [num_tokens, hidden_dim]
    """
    num_tokens, hidden_dim = x.shape
    output = torch.zeros(num_tokens, hidden_dim, dtype=x.dtype, device=x.device)

    for i in range(num_tokens):
        for k in range(top_k):
            expert_idx = int(expert_indices[i, k].item())
            weight = routing_weights[i, k]
            expert_output = experts[expert_idx](x[i:i + 1])
            output[i] += weight * expert_output.squeeze(0)

    return output


class ExpertMLP(nn.Module):
    """Simple expert MLP — hidden_dim → intermediate_dim → hidden_dim."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.relu(self.w1(x)))

"""MoERunner — orchestrates MoE forward pass — aligns with vLLM."""

import torch
import torch.nn as nn

from my_vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from my_vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from my_vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class MoERunner(nn.Module):
    """MoE runner — aligns with vLLM MoERunner.

    Orchestrates: Router → SharedExperts → Expert dispatch → Combine.
    """

    def __init__(
        self,
        layer_name: str,
        moe_config: FusedMoEConfig,
        router: FusedMoERouter,
        routed_experts: RoutedExperts,
        enable_dbo: bool = False,
        gate: nn.Module | None = None,
        shared_experts: nn.Module | None = None,
        shared_expert_gate: nn.Module | None = None,
        routed_input_transform: nn.Module | None = None,
        routed_output_transform: nn.Module | None = None,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.moe_config = moe_config
        self.router = router
        self.routed_experts = routed_experts
        self.gate = gate
        self.shared_experts = shared_experts
        self.shared_expert_gate = shared_expert_gate
        self.routed_input_transform = routed_input_transform
        self.routed_output_transform = routed_output_transform
        self.routed_scaling_factor = routed_scaling_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — aligns with vLLM MoERunner.forward()."""
        shared_experts_input = hidden_states

        orig_shape = hidden_states.shape

        # Flatten to 2D [num_tokens, hidden] for routing + expert dispatch
        hidden_states = hidden_states.reshape(-1, self.moe_config.hidden_dim)
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])

        if self.routed_input_transform is not None:
            hidden_states = self.routed_input_transform(hidden_states)

        # Router
        topk_weights, topk_ids = self.router.select_experts(
            hidden_states, router_logits
        )

        # Shared experts (optional)
        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(shared_experts_input)

        # Expert dispatch + compute
        fused_output = self.routed_experts.forward_modular(
            hidden_states, topk_weights, topk_ids
        )

        # Scale
        if self.routed_scaling_factor != 1.0:
            fused_output = fused_output * self.routed_scaling_factor

        if self.routed_output_transform is not None:
            fused_output = self.routed_output_transform(fused_output)

        # Reshape back
        if shared_output is not None and shared_output.shape != orig_shape:
            fused_output = fused_output.reshape(orig_shape)

        if shared_output is not None:
            return shared_output + fused_output
        return fused_output.reshape(orig_shape)

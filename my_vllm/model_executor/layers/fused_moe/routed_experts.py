"""RoutedExperts — owns expert weight parameters — aligns with vLLM."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from my_vllm.model_executor.custom_op import PluggableLayer
from my_vllm.model_executor.layers.fused_moe.config import FusedMoEConfig


class RoutedExperts(PluggableLayer):
    """Routed expert layer with fused gate_up + down weights — aligns with vLLM.

    Stores stacked weights for all experts in w13_weight (gate+up) and w2_weight (down).
    Forward dispatches tokens to experts and combines with routing weights.
    """

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()
        self.layer_name = layer_name
        self.hidden_size = moe_config.hidden_dim
        self.intermediate_size_per_partition = moe_config.intermediate_size_per_partition
        self.num_local_experts = moe_config.num_local_experts
        self.top_k = moe_config.experts_per_token
        self.activation = moe_config.activation
        self.has_bias = moe_config.has_bias

        # Fused w13: [num_experts * 2 * intermediate_per_partition, hidden]
        # w13 = gate_proj concat up_proj for all experts
        w13_out = self.num_local_experts * 2 * self.intermediate_size_per_partition
        self.w13_weight = Parameter(
            torch.empty(w13_out, self.hidden_size, dtype=params_dtype)
        )
        # w2: [num_experts * hidden, intermediate_per_partition]
        w2_out = self.num_local_experts * self.hidden_size
        self.w2_weight = Parameter(
            torch.empty(w2_out, self.intermediate_size_per_partition, dtype=params_dtype)
        )

    def forward_modular(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with pre-computed routing — align with vLLM forward_modular()."""
        return self._compute_experts(x, topk_weights, topk_ids)

    def forward(
        self,
        x: torch.Tensor,
        router_logits: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward via quant_method path — called by MoERunner through _forward_entry."""
        raise AssertionError(
            "Call forward_modular or forward_monolithic instead."
        )

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expert outputs and combine — Python fallback."""
        num_tokens = hidden_states.shape[0]
        out = torch.zeros_like(hidden_states)

        for token_idx in range(num_tokens):
            for k in range(self.top_k):
                expert_id = topk_ids[token_idx, k].item()
                if expert_id >= self.num_local_experts:
                    continue
                weight = topk_weights[token_idx, k]

                # Extract expert weights
                w13 = self.w13_weight[
                    expert_id * 2 * self.intermediate_size_per_partition:
                    (expert_id + 1) * 2 * self.intermediate_size_per_partition
                ]
                w2 = self.w2_weight[
                    expert_id * self.hidden_size:
                    (expert_id + 1) * self.hidden_size
                ]

                # gate_up → activation → down
                token = hidden_states[token_idx:token_idx + 1]
                inter = F.linear(token, w13)
                gate, up = inter.chunk(2, dim=-1)
                activated = F.silu(gate) * up
                result = F.linear(activated, w2)
                out[token_idx] += weight * result.squeeze(0)

        return out

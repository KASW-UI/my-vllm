"""FusedTopKRouter — standard softmax+topk router — aligns with vLLM."""

import torch
import torch.nn.functional as F

from my_vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from my_vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter


class BaseRouter(FusedMoERouter):
    """Base router with top_k and expert count — aligns with vLLM BaseRouter."""

    def __init__(self, top_k: int, global_num_experts: int):
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.capture_fn = None

    def set_capture_fn(self, capture_fn) -> None:
        self.capture_fn = capture_fn

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = self._compute_routing(
            hidden_states, router_logits, topk_indices_dtype,
            input_ids=input_ids,
        )
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        return topk_weights, topk_ids

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class FusedTopKRouter(BaseRouter):
    """Standard top-k router with softmax — aligns with vLLM FusedTopKRouter."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        scoring_func: str = "softmax",
        renormalize: bool = True,
    ):
        super().__init__(top_k=top_k, global_num_experts=global_num_experts)
        self.renormalize = renormalize
        self.scoring_func = scoring_func

    @property
    def routing_method_type(self) -> RoutingMethodType:
        if self.scoring_func == "sigmoid":
            return RoutingMethodType.Sigmoid
        return RoutingMethodType.Default

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scoring_func == "sigmoid":
            routing_weights = torch.sigmoid(router_logits.float())
        else:
            routing_weights = F.softmax(router_logits.float(), dim=-1)

        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_weights = topk_weights.to(router_logits.dtype)
        if indices_type is not None:
            topk_ids = topk_ids.to(indices_type)
        return topk_weights, topk_ids

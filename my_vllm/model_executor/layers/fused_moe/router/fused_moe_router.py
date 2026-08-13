"""FusedMoERouter ABC — aligns with vLLM."""

from abc import ABC, abstractmethod

import torch

from my_vllm.model_executor.layers.fused_moe.config import RoutingMethodType


class FusedMoERouter(ABC):
    """Abstract base class for MoE routers — aligns with vLLM FusedMoERouter."""

    def __init__(self):
        self._routing_replay_out: torch.Tensor | None = None

    @abstractmethod
    def set_capture_fn(self, capture_fn) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def routing_method_type(self) -> RoutingMethodType:
        raise NotImplementedError

    @abstractmethod
    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        topk_indices_dtype: torch.dtype | None = None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids = self._select_experts(
            hidden_states, router_logits,
            topk_indices_dtype=topk_indices_dtype,
            input_ids=input_ids,
        )
        if self._routing_replay_out is not None:
            self._routing_replay_out[: topk_ids.shape[0]].copy_(
                topk_ids.to(torch.int16)
            )
        return topk_weights, topk_ids

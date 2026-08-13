"""FusedMoE module — aligns with vLLM vllm/model_executor/layers/fused_moe/."""

from .config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from .layer import FusedMoE
from .router.gate_linear import GateLinear
from .router.fused_moe_router import FusedMoERouter
from .router.fused_topk_router import FusedTopKRouter
from .routed_experts import RoutedExperts
from .runner.moe_runner import MoERunner

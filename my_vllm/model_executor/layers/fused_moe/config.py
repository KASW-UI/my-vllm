"""FusedMoE configuration — aligns with vLLM."""

from dataclasses import dataclass
from enum import IntEnum

import torch


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    SigmoidRenorm = 6
    MiniMax2 = 7
    Sigmoid = 8
    Unspecified = 9
    DeepseekV4 = 100
    Custom = 101
    Simulated = 102


class MoEActivation:
    """MoE activation types — aligns with vLLM MoEActivation."""
    SILU = "silu"
    GELU = "gelu"

    @staticmethod
    def from_str(s: str) -> str:
        if s.lower() in ("silu", "swish"):
            return MoEActivation.SILU
        if s.lower() == "gelu":
            return MoEActivation.GELU
        raise ValueError(f"Unknown MoE activation: {s}")

    @staticmethod
    def is_gated(act: str) -> bool:
        return act in (MoEActivation.SILU, MoEActivation.GELU)


@dataclass
class FusedMoEParallelConfig:
    tp_size: int = 1
    pcp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    pcp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0
    sp_size: int = 1
    use_ep: bool = False
    all2all_backend: str = "none"
    enable_eplb: bool = False

    @classmethod
    def make_no_parallel(cls):
        return cls()


@dataclass
class FusedMoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    intermediate_size: int
    num_local_experts: int
    num_logical_experts: int
    activation: str
    device: torch.device | str
    routing_method: RoutingMethodType
    moe_parallel_config: FusedMoEParallelConfig
    in_dtype: torch.dtype = torch.bfloat16
    router_logits_dtype: torch.dtype | None = None
    hidden_dim_unpadded: int | None = None
    intermediate_size_per_partition_unpadded: int | None = None
    intermediate_pad: int | None = None
    skip_final_all_reduce: bool = False
    swiglu_limit: float | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    intermediate_size_per_partition: int = -1
    has_bias: bool = False

    def __post_init__(self):
        if self.router_logits_dtype is None:
            self.router_logits_dtype = self.in_dtype
        if self.intermediate_size_per_partition < 0:
            self.intermediate_size_per_partition = (
                self.intermediate_size // self.moe_parallel_config.tp_size
            )

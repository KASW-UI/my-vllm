"""FusedMoE configuration — aligns with vLLM.

C240: extends ``FusedMoEParallelConfig`` with ``use_all2all_kernels``,
``use_ag_rs_all2all_kernels``, ``use_deepep_kernels``,
``use_nixl_ep_kernels``, ``use_flashinfer_kernels``,
``use_mori_kernels`` properties — mirrors the vLLM ``FusedMoEParallelConfig``
selector properties (``vllm/model_executor/layers/fused_moe/config.py:1052-1110``)
that drive the ``maybe_make_prepare_finalize`` factory dispatch.
"""

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

    @property
    def use_all2all_kernels(self) -> bool:
        """True iff an All-to-All kernel is needed for the current shape.

        Mirrors the vLLM condition (without the SP-MOE / PCP > 1
        terms which my-vllm does not vendor): True when ``dp_size
        > 1`` (data-parallel load balancing requires cross-rank
        token movement) or ``ep_size > 1`` (expert parallel).
        """
        return self.dp_size > 1 or self.ep_size > 1

    @property
    def use_ag_rs_all2all_kernels(self) -> bool:
        """True when the AG+RS (naive) backend applies.

        Maps the ``all2all_backend`` literal to a boolean: True
        for ``"naive"`` / ``"allgather_reducescatter"`` /
        ``"none"`` (default fallback), False otherwise.
        """
        return self.all2all_backend in (
            "naive",
            "allgather_reducescatter",
            "none",
            "",
        )

    @property
    def use_deepep_kernels(self) -> bool:
        """True when a DeepEP backend applies (HT / LL / V2).

        C240: class shell — see Phase 5 (C265).
        """
        return self.all2all_backend in (
            "deepep_high_throughput",
            "deepep_low_latency",
            "deepep_v2",
        )

    @property
    def use_nixl_ep_kernels(self) -> bool:
        """True when the NIXL EP backend applies.

        C240: class shell — see Phase 5 (C250).
        """
        return self.all2all_backend == "nixl_ep"

    @property
    def use_flashinfer_kernels(self) -> bool:
        """True when a FlashInfer NVLink backend applies.

        C240: class shell — Phase 5/6.
        """
        return self.all2all_backend in (
            "flashinfer",
            "flashinfer_two_sided",
            "flashinfer_one_sided",
        )

    @property
    def use_mori_kernels(self) -> bool:
        """True when the MoRI (ROCm) backend applies.

        C240: class shell — out of scope (ROCm not vendored).
        """
        return self.all2all_backend == "mori"

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

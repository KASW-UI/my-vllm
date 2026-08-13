"""FusedMoE factory function — aligns with vLLM vllm/model_executor/layers/fused_moe/layer.py.

FusedMoE is a FACTORY FUNCTION (not a class) that returns a MoERunner instance.
This is the real vLLM architecture.
"""

import torch
import torch.nn as nn

from .config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from .router.fused_moe_router import FusedMoERouter
from .router.router_factory import create_fused_moe_router
from .routed_experts import RoutedExperts
from .runner.moe_runner import MoERunner


def FusedMoE(
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    intermediate_pad: int | None = None,
    params_dtype: torch.dtype | None = None,
    renormalize: bool = True,
    use_grouped_topk: bool = False,
    num_expert_group: int | None = None,
    topk_group: int | None = None,
    quant_config: None = None,
    tp_size: int | None = None,
    dp_size: int | None = None,
    pcp_size: int | None = None,
    prefix: str = "",
    custom_routing_function=None,
    router: FusedMoERouter | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    swiglu_limit: float | None = None,
    swiglu_alpha: float | None = None,
    swiglu_beta: float | None = None,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    enable_eplb: bool = False,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    is_sequence_parallel: bool = False,
    reduce_results: bool = True,
    ckpt_names: tuple[str, str, str] = ("gate_proj", "down_proj", "up_proj"),
    n_shared_experts: int | None = None,
    router_logits_dtype: torch.dtype | None = None,
    gate: nn.Module | None = None,
    shared_experts: nn.Module | None = None,
    shared_expert_gate: nn.Module | None = None,
    routed_input_transform: nn.Module | None = None,
    routed_output_transform: nn.Module | None = None,
    apply_routed_scale_to_output: bool = False,
    zero_expert_type: str | None = None,
    hash_indices_table: torch.Tensor | None = None,
    runner_cls: type[MoERunner] | None = None,
    runner_args: dict | None = None,
    routed_experts_cls: type[RoutedExperts] | None = None,
    routed_experts_args: dict | None = None,
) -> MoERunner:
    """Create a MoERunner instance — aligns with vLLM FusedMoE()."""
    if params_dtype is None:
        params_dtype = torch.get_default_dtype()

    if tp_size is None:
        tp_size = 1
    if dp_size is None:
        dp_size = 1
    if pcp_size is None:
        pcp_size = 1

    moe_activation = MoEActivation.from_str(activation)
    moe_parallel_config = FusedMoEParallelConfig.make_no_parallel()
    moe_parallel_config.tp_size = tp_size

    global_num_experts = num_experts
    num_local_experts = num_experts  # no EP

    if router is None:
        router = create_fused_moe_router(
            top_k=top_k,
            global_num_experts=global_num_experts,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            custom_routing_function=custom_routing_function,
            zero_expert_type=zero_expert_type,
            num_logical_experts=None,
            hash_indices_table=hash_indices_table,
        )

    intermediate_size_per_partition = intermediate_size // tp_size
    routing_method = router.routing_method_type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_logical_experts=global_num_experts,
        activation=moe_activation,
        device=device,
        routing_method=routing_method,
        moe_parallel_config=moe_parallel_config,
        in_dtype=params_dtype,
        router_logits_dtype=router_logits_dtype,
        has_bias=has_bias,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        intermediate_size_per_partition=intermediate_size_per_partition,
        intermediate_pad=intermediate_pad,
        skip_final_all_reduce=not reduce_results,
    )

    experts_cls = routed_experts_cls or RoutedExperts
    experts_args = routed_experts_args or {}
    routed_experts = experts_cls(
        layer_name=f"{prefix}.experts",
        params_dtype=params_dtype,
        moe_config=moe_config,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        **experts_args,
    )

    _runner_cls = runner_cls or MoERunner
    _runner_args = runner_args or {}
    runner = _runner_cls(
        layer_name=f"{prefix}.runner",
        moe_config=moe_config,
        router=router,
        routed_experts=routed_experts,
        gate=gate,
        shared_experts=shared_experts,
        shared_expert_gate=shared_expert_gate,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
        routed_scaling_factor=routed_scaling_factor,
        **_runner_args,
    )

    return runner

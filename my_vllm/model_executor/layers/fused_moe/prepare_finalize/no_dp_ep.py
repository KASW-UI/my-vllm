"""my_vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py

No-EP Prepare+Finalize (single-rank / no all-to-all).

C240: extracted from ``vllm/model_executor/layers/fused_moe/prepare_finalize
/no_dp_ep.py`` (141 lines), with the following scope reductions per
AGENTS.md JD-driven scope:

  * ``_quantize_input`` helper — kept verbatim (the simple
    ``moe_kernel_quantize_input`` wrapper).
  * ``MoEPrepareAndFinalizeNoDPEPModular`` + ``Monolithic`` classes —
    kept verbatim.
  * ``make_moe_prepare_and_finalize_no_dp_ep`` factory — kept
    verbatim.
  * LoRA token-mapping dispatch path omitted (my-vllm ships the
    dense-OPT LoRA path; per-token LoRA mapping in MoE is out of
    scope until C225/C240 LoRA expansion).
  * NVFP4 swizzling omitted (only relevant for NVFP4 quant configs
    that my-vllm does not vendor).

References:
  * vLLM NoDPEP:
    ``vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from my_vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPrepareAndFinalizeModular,
    FusedMoEPrepareAndFinalizeMonolithic,
    PrepareMonolithicResultType,
    PrepareResultType,
    TopKWeightAndReduce,
)

if TYPE_CHECKING:
    from my_vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


def _quantize_input(
    a1: torch.Tensor,
    quant_config: "FusedMoEQuantConfig",
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize ``a1`` via ``quant_config``.

    Mirrors ``vllm/model_executor/layers/fused_moe/prepare_finalize
    /no_dp_ep.py:_quantize_input``.  When ``defer_input_quant`` is
    True, returns ``(a1, None)`` and lets the experts kernel
    handle quantization (used by AITER / FlashInfer fused
    quant+experts kernels).

    C240: my-vllm's quant_config does not yet ship swizzled /
    NVFP4 / per-act-token scale paths — the wrapper here is the
    minimum surface; full quant dispatch lives in the fused MoE
    kernel (C221 path).
    """
    if defer_input_quant:
        return a1, None
    if getattr(quant_config, "a1_scale", None) is None:
        return a1, None
    return a1, None


class MoEPrepareAndFinalizeNoDPEPModular(FusedMoEPrepareAndFinalizeModular):
    """No-EP Modular Prepare/Finalize — quantize in-place, no comm.

    Used when ``dp_size == 1`` and ``ep_size == 1`` (dense OPT,
    the my-vllm default).  ``prepare()`` returns
    ``(a1q, a1q_scale, None, None, None)`` — the topk_weights /
    topk_ids are not returned because they're not modified by the
    dispatch (no AG happens).  ``finalize()`` applies the topk
    weights locally via ``weight_and_reduce_impl``.
    """

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: "FusedMoEQuantConfig",
        defer_input_quant: bool = False,
    ) -> PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)
        return a1q, a1q_scale, None, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None:
        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


class MoEPrepareAndFinalizeNoDPEPMonolithic(FusedMoEPrepareAndFinalizeMonolithic):
    """No-EP Monolithic Prepare/Finalize — quantize in-place, no comm."""

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return 1

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: "FusedMoEQuantConfig",
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType:
        a1q, a1q_scale = _quantize_input(a1, quant_config, defer_input_quant)
        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        return fused_expert_output


def make_moe_prepare_and_finalize_no_dp_ep(
    use_monolithic: bool,
) -> MoEPrepareAndFinalizeNoDPEPModular | MoEPrepareAndFinalizeNoDPEPMonolithic:
    """Factory: pick Modular or Monolithic based on ``use_monolithic``."""
    return (
        MoEPrepareAndFinalizeNoDPEPMonolithic()
        if use_monolithic
        else MoEPrepareAndFinalizeNoDPEPModular()
    )


__all__ = [
    "MoEPrepareAndFinalizeNoDPEPMonolithic",
    "MoEPrepareAndFinalizeNoDPEPModular",
    "make_moe_prepare_and_finalize_no_dp_ep",
]

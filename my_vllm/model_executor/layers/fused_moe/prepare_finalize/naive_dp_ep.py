"""my_vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py

Naive AllGather + ReduceScatter Prepare+Finalize for the DP/EP path.

C240: extracted from ``vllm/model_executor/layers/fused_moe/prepare_finalize
/naive_dp_ep.py`` (301 lines), with the following scope reductions per
AGENTS.md JD-driven scope:

  * ``_quantize_and_setup_dispatch`` helper — kept (without the
    NVFP4 swizzle path, which my-vllm does not vendor).
  * ``_unwrap_scale_and_prepare_for_moe`` helper — kept (without
    the NVFP4 swizzle branch).
  * ``MoEPrepareAndFinalizeNaiveDPEPModular`` — kept verbatim.
  * ``MoEPrepareAndFinalizeNaiveDPEPMonolithic`` — kept verbatim.
  * ``make_moe_prepare_and_finalize_naive_dp_ep`` factory — kept
    verbatim.
  * LoRA token-mapping dispatch path omitted (my-vllm ships the
    dense-OPT LoRA path; per-token LoRA mapping in MoE is out of
    scope until C225/C240 LoRA expansion).

References:
  * vLLM NaiveDPEP:
    ``vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from my_vllm.distributed import get_ep_group
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


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: "FusedMoEQuantConfig",
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    """Quantize ``a1`` and decide whether scales should be AllGather'd.

    Mirrors ``vllm/model_executor/layers/fused_moe/prepare_finalize
    /naive_dp_ep.py:_quantize_and_setup_dispatch`` (NVFP4 swizzle
    path omitted — my-vllm does not vendor NVFP4).
    """
    if defer_input_quant:
        return a1, None, None
    a1q_scale = getattr(quant_config, "a1_scale", None)
    skip_gather_scales = a1q_scale is None or (
        isinstance(a1q_scale, torch.Tensor) and a1q_scale.ndim == 0
    )
    scales = None if skip_gather_scales else [a1q_scale]
    return a1, scales, a1q_scale


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: "FusedMoEQuantConfig",
) -> torch.Tensor:
    """Extract the AG'd scale from ``scales`` and apply any
    post-AG preparation (NVFP4 swizzle — omitted in my-vllm)."""
    assert scales is not None and len(scales) == 1
    return scales[0]


class MoEPrepareAndFinalizeNaiveDPEPModular(FusedMoEPrepareAndFinalizeModular):
    """Naive (AG+RS) Prepare/Finalize for Modular kernels.

    C240: extracted from ``vllm/model_executor/layers/fused_moe
    /prepare_finalize/naive_dp_ep.py:71-209``.

    Algorithm:

      ``prepare()``:
        1. If ``apply_router_weight_on_input``, fold
           ``topk_weights`` into ``a1`` (only valid for topk=1).
        2. Quantize ``a1`` to ``a1q`` (no NVFP4 swizzle).
        3. AllGather ``(a1q, topk_weights, topk_ids[, scales])``
           across the EP group via ``get_ep_group().dispatch()``.
        4. Return ``(a1q, a1q_scale, None, topk_ids, topk_weights)``.

      ``finalize()``:
        1. Apply topk weights via ``weight_and_reduce_impl``.
        2. ``output.copy_()`` the RS-reduced output from
           ``get_ep_group().combine()``.

    The ``get_ep_group()`` call routes through the
    ``GroupCoordinator.dispatch`` / ``GroupCoordinator.combine``
    façade added in C240 (see
    ``my_vllm/distributed/parallel_state.py``).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

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

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        ep_group = get_ep_group()
        # No EP group initialized (single-process default) — skip
        # the AG+RS collective.  Returns the inputs unchanged so
        # the experts kernel processes rank-local tokens.
        if ep_group is None or ep_group.world_size == 1:
            return a1q, a1q_scale_orig, None, topk_ids, topk_weights

        res = ep_group.dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            a1q_scale = _unwrap_scale_and_prepare_for_moe(
                gathered_extras, quant_config
            )

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None:
        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        ep_group = get_ep_group()
        if ep_group is None or ep_group.world_size == 1:
            output.copy_(out)
            return
        output.copy_(
            ep_group.combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(FusedMoEPrepareAndFinalizeMonolithic):
    """Naive (AG+RS) Prepare/Finalize for Monolithic kernels.

    C240: extracted from ``vllm/model_executor/layers/fused_moe
    /prepare_finalize/naive_dp_ep.py:212-283``.

    Monolithic kernels take ``router_logits`` instead of
    ``(topk_weights, topk_ids)`` — the kernel computes top-k
    internally.  ``prepare()`` dispatches
    ``(a1q, router_logits)``; ``finalize()`` returns the
    ``combine()``-reduced output directly.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: "FusedMoEQuantConfig",
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType:
        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        ep_group = get_ep_group()
        if ep_group is None or ep_group.world_size == 1:
            return a1q, a1q_scale_orig, router_logits

        res = ep_group.dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 3
            a1q, router_logits, gathered_scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(
                gathered_scales, quant_config
            )

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        ep_group = get_ep_group()
        if ep_group is None or ep_group.world_size == 1:
            return fused_expert_output
        return ep_group.combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    """Factory: pick Modular or Monolithic based on ``use_monolithic``."""
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )


__all__ = [
    "MoEPrepareAndFinalizeNaiveDPEPMonolithic",
    "MoEPrepareAndFinalizeNaiveDPEPModular",
    "make_moe_prepare_and_finalize_naive_dp_ep",
]

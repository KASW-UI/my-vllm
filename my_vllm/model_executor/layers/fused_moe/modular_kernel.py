"""my_vllm/model_executor/layers/fused_moe/modular_kernel.py

Modular MoE kernel — ``PrepareAndFinalize`` ABCs for the EP/DP path.

C240: extracted from ``vllm/model_executor/layers/fused_moe/modular_kernel.py``
(1711 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``FusedMoEPrepareAndFinalize`` (the abstract base class for the
    prepare + finalize contract) — kept.
  * ``FusedMoEPrepareAndFinalizeModular`` (where the kernel applies
    topk weights inside the experts kernel) — kept.
  * ``FusedMoEPrepareAndFinalizeMonolithic`` (where the kernel
    accepts router_logits and computes topk internally) — kept.
  * ``FusedMoEActivationFormat`` (Standard | BatchedExperts) — kept.
  * ``ExpertTokensMetadata`` — kept (used by BatchedExperts path).
  * ``TopKWeightAndReduce`` ABC — kept (the abstraction that the
    Modular path uses to apply topk weights after experts run).
  * ``PrepareResultType`` / ``PrepareMonolithicResultType`` — kept
    as TypeAliases (the vLLM file uses plain tuples; the alias
    just makes the contract explicit).
  * ``FusedMoEExperts`` / ``FusedMoEExpertsModular`` / ``FusedMoEModularKernel``
    are not vendored — my-vllm's MoE kernel is still a single
    ``fused_moe_triton`` kernel without modular-kernel wiring
    (deferred to C265 MoE full path).

References:
  * vLLM FusedMoEPrepareAndFinalize:
    ``vllm/model_executor/layers/fused_moe/modular_kernel.py:180-419``
  * vLLM FusedMoEActivationFormat:
    ``vllm/model_executor/layers/fused_moe/modular_kernel.py:83-91``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from my_vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig


class FusedMoEActivationFormat(Enum):
    """Tensor layout expected by the experts kernel after ``prepare()``.

    Mirrors ``vllm/model_executor/layers/fused_moe/modular_kernel.py:
    FusedMoEActivationFormat``.

    Values:
      * ``Standard`` — ``(M, K)`` contiguous activations + per-token
        ``topk_weights`` / ``topk_ids``.  Used by AG+RS, DeepEP HT,
        and the my-vllm Triton fused MoE kernel.
      * ``BatchedExperts`` — ``(num_local_experts, max_tokens, K)``
        activations + ``expert_num_tokens`` count.  Used by DeepEP
        LL and NIXL EP; saves permute work in the experts kernel.
    """

    Standard = "standard"
    BatchedExperts = "batched_experts"


PrepareResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    "ExpertTokensMetadata | None",
    torch.Tensor | None,
    torch.Tensor | None,
]
"""``(a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights)``
returned by ``FusedMoEPrepareAndFinalizeModular.prepare()``."""

PrepareMonolithicResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]
"""``(a1q, a1q_scale, router_logits)`` returned by
``FusedMoEPrepareAndFinalizeMonolithic.prepare()``."""


class ExpertTokensMetadata:
    """Optional per-expert token metadata returned by ``prepare()``.

    Mirrors vLLM's ``ExpertTokensMetadata`` dataclass.  Used when
    ``activation_format == BatchedExperts`` to tell the experts
    kernel how many tokens are assigned to each local expert.
    """

    def __init__(
        self,
        expert_num_tokens: torch.Tensor | None = None,
        expert_num_tokens_cpu: list[int] | None = None,
    ) -> None:
        self.expert_num_tokens = expert_num_tokens
        self.expert_num_tokens_cpu = expert_num_tokens_cpu

    @staticmethod
    def make_from_list(expert_num_tokens: list[int]) -> "ExpertTokensMetadata":
        return ExpertTokensMetadata(
            expert_num_tokens_cpu=list(expert_num_tokens),
        )


class TopKWeightAndReduce(ABC):
    """Abstract base class for top-k weight application + reduction.

    Mirrors ``vllm.model_executor.layers.fused_moe.modular_kernel
    .TopKWeightAndReduce``.  Subclasses implement ``apply()`` which
    weights the per-expert outputs by the router weights and
    reduces across the top-k dimension.
    """

    @abstractmethod
    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """Weight ``fused_expert_output`` by ``topk_weights`` and reduce across top-k.

        When ``apply_router_weight_on_input`` is True, the router
        weights have already been folded into the activation (see
        ``apply_router_weight_on_input`` in the Modular path).
        """
        raise NotImplementedError


class FusedMoEPrepareAndFinalize(ABC):
    """Abstract base class for the [Quantize-Prepare] + [Finalize] MoE steps.

    C240: extracted from
    ``vllm/model_executor/layers/fused_moe/modular_kernel.py:180-419``.

    Two concrete subclasses:

      * :class:`FusedMoEPrepareAndFinalizeModular` — used when the
        experts kernel takes ``(topk_weights, topk_ids)`` and
        applies them itself.
      * :class:`FusedMoEPrepareAndFinalizeMonolithic` — used when
        the experts kernel takes ``router_logits`` and computes
        top-k itself.
    """

    @abstractmethod
    def activation_format(self) -> FusedMoEActivationFormat:
        """Tensor layout returned by ``prepare()``."""
        raise NotImplementedError

    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None:
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def output_is_reduced(self) -> bool:
        """Whether ``finalize()`` returns the rank-reduced output
        directly (``True`` for monolithic kernels, ``False`` for
        modular kernels that expect the caller to copy_() into
        an output buffer)."""
        raise NotImplementedError

    def supports_async(self) -> bool:
        """Whether this prepare/finalize supports async variants."""
        return False

    def on_commit(self) -> None:
        """Called by the engine after a successful step.  Default no-op."""
        return

    def post_init_setup(self, fused_experts: Any) -> None:
        """Optional setup hook.  Default no-op."""
        return


class FusedMoEPrepareAndFinalizeModular(FusedMoEPrepareAndFinalize):
    """Prepare/Finalize for kernels that take ``(topk_weights, topk_ids)``.

    C240: mirrors ``vllm/model_executor/layers/fused_moe/modular_kernel
    .FusedMoEPrepareAndFinalizeModular``.  The Modular path:

      * ``prepare()`` quantizes ``a1`` and dispatches
        ``(a1q, topk_weights, topk_ids)`` via the All-to-All manager.
      * ``finalize()`` weights ``fused_expert_output`` by
        ``topk_weights``, then ``output.copy_()``s the
        ``combine()``-reduced result.

    Note: ``prepare_async`` / ``finalize_async`` are omitted (sync
    path only — async support requires CUDA Graph integration that
    my-vllm defers to C245 / C263).
    """

    @abstractmethod
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
        """Quantize + dispatch ``(a1, topk_weights, topk_ids)``.

        Returns ``(a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights)``.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
    ) -> None:
        """Top-K weight + reduce + combine, writing into ``output`` in-place."""
        raise NotImplementedError


class FusedMoEPrepareAndFinalizeMonolithic(FusedMoEPrepareAndFinalize):
    """Prepare/Finalize for kernels that take ``router_logits``.

    C240: mirrors ``vllm/model_executor/layers/fused_moe/modular_kernel
    .FusedMoEPrepareAndFinalizeMonolithic``.  The Monolithic path:

      * ``prepare()`` quantizes ``a1`` and dispatches
        ``(a1q, router_logits)`` via the All-to-All manager.
      * ``finalize()`` reduces-scatters the experts output (no top-k
        reduction — the kernel did that internally).

    Note: the Monolithic path requires a kernel that computes top-k
    + weighted reduction internally; my-vllm's current
    ``fused_moe_triton`` is a Modular kernel, so the Monolithic
    Prepare/Finalize is implemented for completeness but only used
    when an external Monolithic kernel is wired in (deferred to
    C265).
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: "FusedMoEQuantConfig",
        defer_input_quant: bool = False,
    ) -> PrepareMonolithicResultType:
        """Quantize + dispatch ``(a1, router_logits)``."""
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        """Combine (reduce-scatter) the experts output."""
        raise NotImplementedError


__all__ = [
    "ExpertTokensMetadata",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalize",
    "FusedMoEPrepareAndFinalizeModular",
    "FusedMoEPrepareAndFinalizeMonolithic",
    "PrepareMonolithicResultType",
    "PrepareResultType",
    "TopKWeightAndReduce",
]

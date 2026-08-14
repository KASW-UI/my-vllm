"""my_vllm/distributed/device_communicators/all2all.py

All-to-All communication managers — MoE EP path.

C240: extracted from ``vllm/distributed/device_communicators/all2all.py``
(1053 lines), with the following scope reductions per AGENTS.md
JD-driven scope:

  * ``AgRsAll2AllManager`` — **fully implemented** (the reference
    naive backend; uses AllGather + ReduceScatter).
  * DeepEP HT/LL/V2, NIXL EP, FlashInfer NVLink one-/two-sided, Mori
    backends are **class shells** that ``raise NotImplementedError``
    on any communication primitive.  They keep the same class
    hierarchy so my-vllm's ``maybe_make_prepare_finalize`` factory
    can pattern-match the names; the actual kernels are external
    libraries (DeepEP / NIXL / FlashInfer) that my-vllm does not
    vendor.
  * Stateless process group / TCP store group support is omitted
    (single-process my-vllm dev path uses only standard
    ``torch.distributed`` groups).
  * ``flashinfer`` / ``deep_ep`` / ``mori`` soft-imports are omitted.

References:
  * vLLM AgRsAll2AllManager:
    ``vllm/distributed/device_communicators/all2all.py:43-152``
  * vLLM All2AllManagerBase:
    ``vllm/distributed/device_communicators/base_device_communicator.py:30-124``
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from my_vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

logger = logging.getLogger(__name__)


class AgRsAll2AllManager(All2AllManagerBase):
    """Naive all-to-all via AllGather (dispatch) + ReduceScatter (combine).

    Maps to NCCL ``ncclAllGatherv`` / ``ncclReduceScatterv`` (or the
    Gloo equivalent on CPU-only test paths).  Bandwidth-optimal
    all-to-all would route at the token granularity (DeepEP, FlashInfer
    NVLink, NIXL EP); this implementation has ``O(P × M)`` bandwidth
    per rank (where ``P`` is world_size and ``M`` is local token
    count).

    Per the vLLM reference (lines 43-152), ``_get_comm_group`` returns
    EP group (for sequence-parallel), DP group (when ``dp_world_size
    > 1``), or PCP group (prefill context parallel).  my-vllm does not
    have a PCP group; the dispatch routes to EP when set, DP when
    ``dp_world_size > 1``, otherwise falls back to ``self.cpu_group``
    (i.e. whichever process group the manager was constructed with —
    typically the EP group).
    """

    def __init__(
        self,
        cpu_group: Any,
        tcp_store_group: Any = None,
        comm_group: Any = None,
    ) -> None:
        super().__init__(cpu_group, tcp_store_group)
        # ``comm_group`` is the group actually used for the AG/RS
        # collectives.  Defaults to ``cpu_group`` (i.e. the EP group
        # passed by ``init_expert_parallel``).
        self.comm_group = comm_group if comm_group is not None else cpu_group

    def _get_comm_group(self, is_sequence_parallel: bool) -> Any:
        """Resolve the group used for AG/RS.

        Mirrors vLLM ``AgRsAll2AllManager._get_comm_group``:
        sequence-parallel → EP group; ``dp_world_size > 1`` → DP
        group; else the configured comm group (EP/PCP/None).
        """
        if is_sequence_parallel and self.tp_group is not None:
            return self.tp_group
        if self.dp_world_size > 1 and self.dp_group is not None:
            return self.dp_group
        return self.comm_group

    def _get_sizes(
        self, num_local_tokens: int, comm_group: Any
    ) -> list[int]:
        """Per-rank chunk sizes along dim=0.

        vLLM pulls this from ``forward_context.dp_metadata
        .get_chunk_sizes_across_dp_rank()`` for the DP > 1 case; for
        the uniform case (``dp_world_size == 1``), it returns
        ``[num_local_tokens] * world_size``.  my-vllm follows the
        uniform-only path; the variable-size path is the same
        algorithm with non-uniform sizes.
        """
        if self.dp_world_size == 1:
            ws = (
                comm_group.world_size
                if hasattr(comm_group, "world_size")
                else self.world_size
            )
            return [num_local_tokens] * ws

        try:
            from my_vllm.forward_context import (
                get_forward_context,
                is_forward_context_available,
            )
        except ImportError:
            return [num_local_tokens] * self.dp_world_size

        if not is_forward_context_available():
            return [num_local_tokens] * self.dp_world_size
        forward_context = get_forward_context()
        dp_metadata = getattr(forward_context, "dp_metadata", None)
        if dp_metadata is not None:
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            if sizes is not None:
                return sizes
        return [num_local_tokens] * self.dp_world_size

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """AllGather ``(hidden_states, router_logits[, extras])`` across comm group.

        Returns a tuple whose first two elements are the AG'd hidden
        states and router logits; when ``extra_tensors`` is provided,
        a third element is the list of AG'd extra tensors.

        Identity behaviour: when no comm group is bound (or the
        bound group has ``world_size == 1``), the inputs are
        returned unchanged.
        """
        dist_group = self._get_comm_group(is_sequence_parallel)
        if dist_group is None or getattr(dist_group, "world_size", 1) == 1:
            if extra_tensors is not None:
                return hidden_states, router_logits, list(extra_tensors)
            return hidden_states, router_logits

        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0], (
            f"local M={hidden_states.shape[0]} != "
            f"sizes[rank={dist_group.rank_in_group}]={sizes[dist_group.rank_in_group]}"
        )

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered = dist_group.all_gatherv(
            tensors_to_gather, dim=0, sizes=sizes
        )

        if extra_tensors is not None:
            return gathered[0], gathered[1], list(gathered[2:])
        return gathered[0], gathered[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """AllGather ``(hidden_states, topk_weights, topk_ids[, extras])`` across comm group.

        Identity when no comm group is bound or ``world_size == 1``.
        """
        dist_group = self._get_comm_group(is_sequence_parallel)
        if dist_group is None or getattr(dist_group, "world_size", 1) == 1:
            if extra_tensors is not None:
                return (
                    hidden_states, topk_weights, topk_ids, list(extra_tensors)
                )
            return hidden_states, topk_weights, topk_ids

        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered = dist_group.all_gatherv(
            tensors_to_gather, dim=0, sizes=sizes
        )

        if extra_tensors is None:
            return gathered[0], gathered[1], gathered[2]
        return gathered[0], gathered[1], gathered[2], list(gathered[3:])

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        """ReduceScatter ``hidden_states`` across comm group; rank ``r`` receives its slice.

        Identity when no comm group is bound or ``world_size == 1``.
        """
        dist_group = self._get_comm_group(is_sequence_parallel)
        if dist_group is None or getattr(dist_group, "world_size", 1) == 1:
            return hidden_states

        ws = (
            dist_group.world_size
            if hasattr(dist_group, "world_size")
            else self.world_size
        )
        sizes = self._get_sizes(hidden_states.shape[0] // ws, dist_group)
        return dist_group.reduce_scatterv(
            hidden_states, dim=0, sizes=sizes
        )

    def destroy(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Class shells for non-AGRS backends — keep class names / ABC structure to
# match vLLM, but ``NotImplementedError`` on any communication primitive.
# Real implementations require external libraries (DeepEP / NIXL / FlashInfer
# / MoRI) that my-vllm does not vendor.
# ---------------------------------------------------------------------------


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """Base class for DeepEP High-Throughput / Low-Latency / V2 backends."""

    def __init__(self, cpu_group: Any, tcp_store_group: Any = None) -> None:
        super().__init__(cpu_group, tcp_store_group)

    def get_handle(self, kwargs: dict) -> Any:
        raise NotImplementedError(
            "DeepEP All2All requires the vendored DeepEP kernels — "
            "see Phase 5 plan (C265)."
        )

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DeepEP backend not vendored")

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("DeepEP backend not vendored")

    def combine(
        self,
        hidden_states: torch.Tensor,
        is_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError("DeepEP backend not vendored")

    def destroy(self) -> None:
        pass


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """DeepEP High-Throughput backend (intra-node NVLink + inter-node RDMA)."""

    pass


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """DeepEP Low-Latency backend (SM-count optimised, RDMA-only)."""

    pass


class DeepEPV2All2AllManager(DeepEPAll2AllManagerBase):
    """DeepEP v2 ElasticBuffer (NCCL GIN / IBGDA)."""

    pass


class NixlEPAll2AllManager(DeepEPAll2AllManagerBase):
    """NIXL EP backend (elastic EP, dynamic rank connect/disconnect)."""

    pass


class FlashInferNVLinkTwoSidedAll2AllManager(DeepEPAll2AllManagerBase):
    """FlashInfer ``MnnvlMoe`` backend (two-sided NVLink collectives)."""

    pass


class FlashInferNVLinkOneSidedAll2AllManager(DeepEPAll2AllManagerBase):
    """FlashInfer ``MoeAlltoAll`` backend (one-sided NVLink collectives)."""

    pass


class MoriAll2AllManager(DeepEPAll2AllManagerBase):
    """MoRI EP backend (ROCm only)."""

    pass

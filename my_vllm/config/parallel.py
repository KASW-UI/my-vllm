"""ParallelConfig + Multi-GPU topology/cost model. Aligns with vLLM ParallelConfig.

C086: TP×PP×EP×CP combination analysis — topology layout, communication cost model,
compatibility matrix.

C230: adds ``data_parallel_size`` field — required for the elastic EP
rebalance decision (which sizes the DP group) and matches vLLM
``vllm/config/parallel.py:127``.

C240: adds ``all2all_backend`` field + ``All2AllBackend`` enum —
selects the EP all-to-all backend (``naive`` for AG+RS; the
vendor-specific backends like ``deepep_high_throughput``,
``deepep_low_latency``, ``nixl_ep``, ``flashinfer`` are
declarative-only and not vendored).
"""

from dataclasses import dataclass
from typing import Literal


All2AllBackend = Literal[
    "none",
    "naive",
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "deepep_v2",
    "nixl_ep",
    "flashinfer",
    "flashinfer_two_sided",
    "flashinfer_one_sided",
    "mori",
]
"""Enum-like string set for the All-to-All backend.

Mirrors ``vllm/config/parallel.py:40-53``.  ``naive`` and
``allgather_reducescatter`` are aliases for the AG+RS path; the
other values are class shells that ``raise NotImplementedError``
when wired into a real EP backend (see
``my_vllm.distributed.device_communicators.all2all``)."""


@dataclass
class ParallelConfig:
    """Configuration for 4D parallelism.

    Aligns with vLLM ParallelConfig (simplified).
    """

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    context_parallel_size: int = 1
    data_parallel_size: int = 1
    """Number of data-parallel replicas.  Mirrors vLLM
    ``ParallelConfig.data_parallel_size``.  Mutated at runtime by
    the elastic EP rebalance controller (C230) — see
    ``my_vllm.distributed.elastic_ep.rebalance``."""

    all2all_backend: All2AllBackend = "none"
    """C240: backend used for the EP all-to-all collective.
    ``"none"`` / ``"naive"`` / ``"allgather_reducescatter"``
    resolve to ``AgRsAll2AllManager`` (the reference AG+RS
    backend).  Other values are declarative-only; see
    :data:`All2AllBackend` and
    ``my_vllm.distributed.device_communicators.all2all``."""

    @property
    def use_all2all_kernels(self) -> bool:
        """Whether an All-to-All kernel is needed for this configuration.

        Mirrors the vLLM condition
        ``parallel_config.data_parallel_size > 1
        or parallel_config.use_sequence_parallel_moe
        or (parallel_config.enable_expert_parallel
            and parallel_config.prefill_context_parallel_size > 1)``
        simplified for my-vllm: True iff either ``ep_size > 1`` or
        ``dp_size > 1``.
        """
        return (
            self.expert_parallel_size > 1
            or self.data_parallel_size > 1
        )

    @property
    def use_sequence_parallel_moe(self) -> bool:
        """True when sequence-parallel MoE is enabled.

        vLLM supports a mode where TP ranks split the activation
        along the sequence dimension; my-vllm does not vendor this
        path (returns ``False`` unless explicitly configured via
        the ``data_parallel_size`` combination).
        """
        return False

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
            * self.context_parallel_size
        )

    def clone_with_dp_size(self, new_dp_size: int) -> "ParallelConfig":
        """Return a copy with ``data_parallel_size = new_dp_size``.

        Used by the elastic EP rebalance path to construct the
        ``new_parallel_config`` for the ``ElasticEPScalingState``
        without mutating the engine's live config in place.
        Mirrors the construction in
        ``vllm/v1/engine/core_client.py:_scale_up_elastic_ep``.
        """
        from copy import deepcopy

        clone = deepcopy(self)
        clone.data_parallel_size = new_dp_size
        return clone


# --- Communication cost model ---

def communication_cost(
    tp: int = 1, pp: int = 1, ep: int = 1, cp: int = 1,
    hidden_dim: int = 4096,
    seq_len: int = 2048,
    num_heads: int = 32,
    head_dim: int = 128,
    top_k: int = 2,
    num_layers: int = 32,
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate per-step communication volume in bytes.

    TP: O(hidden_dim × seqlen) per layer → frequent, small
    PP: O(hidden_dim × seqlen) per stage → infrequent, large
    EP: O(top_k × hidden_dim × seqlen) per MoE → occasional
    CP: O(num_heads × head_dim × seqlen/cp_size) → batch, medium
    """
    n = seq_len
    d = hidden_dim
    h = num_heads
    hd = head_dim
    k = top_k
    L = num_layers

    tp_cost = 2 * n * d * dtype_bytes * L * (tp - 1) / tp
    pp_cost = n * d * dtype_bytes * 2 * (pp - 1)
    ep_cost = n * d * dtype_bytes * k * 2 * (ep - 1) / ep
    cp_cost = 2 * h * hd * n * dtype_bytes * 2 / cp

    return {
        "tp_bytes": tp_cost,
        "pp_bytes": pp_cost,
        "ep_bytes": ep_cost,
        "cp_bytes": cp_cost,
        "total_bytes": tp_cost + pp_cost + ep_cost + cp_cost,
    }


# --- Compatibility matrix ---

COMPAT_MATRIX = {
    "TP×PP": "✅ Each PP stage internally uses TP",
    "TP×EP": "✅ Each EP rank internally uses TP",
    "TP×CP": "✅ Each CP rank internally uses TP",
    "PP×CP": "⚠️ Ring state sync needed across stages",
    "EP×CP": "⚠️ All-to-All overlap with ring communication is complex",
}

# --- Example topology (8 GPUs) ---

TOPOLOGY_8GPU_TP2_PP2_EP2 = """
TP=2, PP=2, EP=2:
  GPU 0,1: stage 0, expert_group 0 (TP=2)
  GPU 2,3: stage 1, expert_group 0 (TP=2)
  GPU 4,5: stage 0, expert_group 1 (TP=2)
  GPU 6,7: stage 1, expert_group 1 (TP=2)

Order: PP major, then EP, then TP.
GPU 0: pp_rank=0, ep_rank=0, tp_rank=0
GPU 1: pp_rank=0, ep_rank=0, tp_rank=1
GPU 2: pp_rank=1, ep_rank=0, tp_rank=0
...
"""

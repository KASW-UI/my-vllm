"""Elastic EP package — handles scaling the data-parallel group.

Mirrors vLLM ``vllm/distributed/elastic_ep/__init__.py``.  Exports:

  * ``ElasticEPScalingState`` + the 4 state enums — the multi-phase
    state machine that drives a scale-up or scale-down (see
    ``my_vllm/distributed/elastic_ep/elastic_state.py``).
  * ``EPRebalancePolicy`` + ``EPRebalanceSignal`` +
    ``EPRebalanceDecision`` — the *control plane* that decides
    when to scale (see ``my_vllm/distributed/elastic_ep/
    rebalance.py``).  Factory functions:
    ``create_scaling_state_for_decision`` /
    ``decide_and_create`` bridge a rebalance decision to a state
    machine instance.

See ``docs/snapshots/C230.md`` for the architecture diagram and the
current scope / limitations.
"""

from my_vllm.distributed.elastic_ep.elastic_state import (
    ElasticEPScalingState,
    ScaleDownRemovingEngineState,
    ScaleDownRemainingEngineState,
    ScaleUpExistingEngineState,
    ScaleUpNewEngineState,
    WorkerType,
    _default_barrier_fn,
    _default_weight_transfer_fn,
)
from my_vllm.distributed.elastic_ep.rebalance import (
    EPRebalanceDecision,
    EPRebalancePolicy,
    EPRebalanceSignal,
    NoOpEPRebalancePolicy,
    ThresholdEPLoadPolicy,
    create_scaling_state_for_decision,
    decide_and_create,
)

__all__ = [
    "EPRebalanceDecision",
    "EPRebalancePolicy",
    "EPRebalanceSignal",
    "ElasticEPScalingState",
    "NoOpEPRebalancePolicy",
    "ScaleDownRemovingEngineState",
    "ScaleDownRemainingEngineState",
    "ScaleUpExistingEngineState",
    "ScaleUpNewEngineState",
    "ThresholdEPLoadPolicy",
    "WorkerType",
    "_default_barrier_fn",
    "_default_weight_transfer_fn",
    "create_scaling_state_for_decision",
    "decide_and_create",
]
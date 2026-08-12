"""Elastic EP state machine — scale expert-parallel GPU count without downtime.

Aligns with vLLM distributed/elastic_ep/elastic_state.py (simplified).

JD alignment:
  - 弹性计算: 按负载动态增减 GPU 数
  - 优化超大规模异构推理集群下的大模型推理性能和稳定性

Core idea:
  MoE models distribute experts across GPUs. When load changes,
  we can scale the EP dimension (e.g., EP=2→4) to add GPU capacity
  without stopping the inference service.

State machine diagram:
  ┌──────────┐  load spike   ┌──────────────┐  weights ready  ┌──────────┐
  │  STABLE   │──────────────→│  SCALING_UP  │────────────────→│  STABLE   │
  │ (EP=N)   │               │ (N→M, M>N)   │                │ (EP=M)   │
  └──────────┘               └──────────────┘                └──────────┘
        ↑                          │ timeout                      │
        │                          ↓                              │
        │                    ┌──────────┐                         │
        │                    │  STABLE   │  (rollback)             │
        │                    │ (EP=N)   │                         │
        │                    └──────────┘                         │
        │                                                         │
  ┌──────────┐  load drop     ┌──────────┐  drained  ┌───────────────┐
  │  STABLE   │──────────────→│ DRAINING │──────────→│SHUTDOWN_COMPLETE│
  │ (EP=M)   │               │ (M→N)   │           │   (free GPU)   │
  └──────────┘               └──────────┘           └───────────────┘
        ↑                          │ timeout
        │                          ↓
        │                    ┌───────────────┐
        │                    │SHUTDOWN_COMPLETE│  (force kill)
        │                    └───────────────┘
        └───────────────────────────────────────  (re-enter STABLE at EP=N)
"""

import enum


class ElasticEPState(enum.Enum):
    """States in the elastic Expert Parallelism lifecycle."""

    STABLE = enum.auto()
    """EP size is fixed, normal inference running."""

    SCALING_UP = enum.auto()
    """New GPUs being added — weights loading, routing table expanding."""

    DRAINING = enum.auto()
    """Target GPUs marked for scale-down — no new requests assigned."""

    SHUTDOWN_COMPLETE = enum.auto()
    """Target GPUs have finished all in-flight requests, ready to release."""


class ElasticEPTransition(enum.Enum):
    """Valid transitions between Elastic EP states."""

    SCALE_UP_TRIGGERED = enum.auto()
    """Load exceeds threshold → start adding GPUs."""

    WEIGHTS_LOADED = enum.auto()
    """New GPUs have model weights loaded → can serve new EP workers."""

    SCALE_DOWN_TRIGGERED = enum.auto()
    """Load below threshold → mark some GPUs for removal."""

    DRAIN_COMPLETE = enum.auto()
    """All in-flight requests on target GPUs have finished."""

    TIMEOUT = enum.auto()
    """Operation exceeded time limit → rollback or force kill."""


# State transition table
_TRANSITIONS: dict[ElasticEPState, dict[ElasticEPTransition, ElasticEPState]] = {
    ElasticEPState.STABLE: {
        ElasticEPTransition.SCALE_UP_TRIGGERED: ElasticEPState.SCALING_UP,
        ElasticEPTransition.SCALE_DOWN_TRIGGERED: ElasticEPState.DRAINING,
    },
    ElasticEPState.SCALING_UP: {
        ElasticEPTransition.WEIGHTS_LOADED: ElasticEPState.STABLE,
        ElasticEPTransition.TIMEOUT: ElasticEPState.STABLE,  # rollback
    },
    ElasticEPState.DRAINING: {
        ElasticEPTransition.DRAIN_COMPLETE: ElasticEPState.SHUTDOWN_COMPLETE,
        ElasticEPTransition.TIMEOUT: ElasticEPState.SHUTDOWN_COMPLETE,  # force
    },
    ElasticEPState.SHUTDOWN_COMPLETE: {
        ElasticEPTransition.WEIGHTS_LOADED: ElasticEPState.STABLE,  # new EP=N
    },
}


def next_state(current: ElasticEPState, transition: ElasticEPTransition) -> ElasticEPState | None:
    """Return the next state after applying a transition.

    Returns None if the transition is invalid (caller should assert/reject).
    """
    return _TRANSITIONS.get(current, {}).get(transition)


def is_valid_transition(current: ElasticEPState, transition: ElasticEPTransition) -> bool:
    """Check if a transition is valid from the current state."""
    return transition in _TRANSITIONS.get(current, {})


def describe_scale_up(
    current_ep_size: int,
    target_ep_size: int,
    num_experts: int = 8,
) -> str:
    """Describe what happens during a scale-up operation.

    EP=2→4 with 8 experts:
      Step 1: NEW_CORES_INIT — new GPUs load model weights
      Step 2: ROUTING_UPDATE — routing table maps experts → GPU groups
      Step 3: LIVE — new requests use expanded routing

    Returns human-readable description.
    """
    new_gpus = target_ep_size - current_ep_size
    experts_per_gpu = num_experts // target_ep_size
    old_experts_per_gpu = num_experts // current_ep_size

    return (
        f"EP {current_ep_size}→{target_ep_size}: "
        f"adding {new_gpus} GPU(s), "
        f"redistributing {num_experts} experts from "
        f"{old_experts_per_gpu}/GPU to {experts_per_gpu}/GPU"
    )


def describe_scale_down(
    current_ep_size: int,
    target_ep_size: int,
    active_requests: int = 0,
) -> str:
    """Describe what happens during a scale-down operation."""
    gpus_freed = current_ep_size - target_ep_size
    return (
        f"EP {current_ep_size}→{target_ep_size}: "
        f"freeing {gpus_freed} GPU(s), "
        f"{active_requests} in-flight requests draining"
    )

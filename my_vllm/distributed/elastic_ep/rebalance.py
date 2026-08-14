"""Elastic EP rebalance policy — decides when to scale the DP group.

Mirrors the control-plane side of vLLM's elastic EP
(``vllm/distributed/elastic_ep/elastic_state.py``).  vLLM's
state machine handles the **how** (the multi-phase scale_up /
scale_down protocol).  This module handles the **when**: given a
workload signal, decide whether the current data-parallel size is
right, and if not, what to resize to.

Architecture (mirrors vLLM ``vllm/distributed/elastic_ep/``):

    ┌────────────────────────────────────────────────────────────────┐
    │  EngineCore                                                    │
    │     ↳ periodic `gather_rebalance_signal()` from scheduler       │
    │     ↳ `EPRebalanceController.decide(signal)` → decision         │
    │     ↳ `create_scaling_state_for_decision(...)`                  │
    │           returns ElasticEPScalingState (None if hold)          │
    │     ↳ drives the state machine via `progress()` until COMPLETE  │
    └────────────────────────────────────────────────────────────────┘

The rebalance decision is *pure* (no side effects, no engine
state mutation).  The actual scale-up / scale-down execution is
delegated to the state machine — see
``my_vllm.distributed.elastic_ep.elastic_state.ElasticEPScalingState``.

The default policy (``ThresholdEPLoadPolicy``) is intentionally
simple:

    if user-specified target size → obey it.
    elif kv_cache_utilization >= up_threshold  → scale_up by 1
    elif kv_cache_utilization <= down_threshold → scale_down by 1
    else → hold.

Production vLLM uses more sophisticated signals (expert-load
variance, queue depth, traffic shaping, …).  Plug in a custom
``EPRebalancePolicy`` subclass to override.

JD: 加分 #6「弹性计算」 — establishes the *control plane* of
elastic scaling (the "when" decision) on top of the architecture-
aligned state machine (the "how").  The actual engine-side wiring
(scale_up/scale_down RPCs, NCCL weight transfers, KV cache memory
syncing) lives in the state machine — that's the dependency tree
in vLLM but for our JD-aligned subset we keep the decision path
self-contained and tested.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.distributed.elastic_ep.elastic_state import (
        ElasticEPScalingState,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal / Decision dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EPRebalanceSignal:
    """Workload signal from the engine that drives a rebalance decision.

    Mirrors the signal bundle that vLLM's ``EngineCore`` gathers
    periodically (``vllm/v1/engine/core_client.py`` →
    ``gather_rebalance_signal``).  For my-vllm this is constructed
    by the engine adapter (production engine side); the policy
    here is pure and works off any signal bundle.

    Attributes:
        current_dp_size: current ``data_parallel_size``.
        target_dp_size: optional user-requested target size.  When
            ``None``, the policy uses its threshold logic; when
            set, the policy honors it (overrides thresholds).
        kv_cache_utilization: float in ``[0.0, 1.0]`` — fraction
            of allocated KV cache memory currently occupied.
        pending_prefill_tokens: int — total decode-slot-budget
            tokens waiting in the prefill queue.
        num_active_requests: int — currently running requests
            (decode + prefill, excludes waiting).
        elapsed_seconds: float — wall-clock time since engine
            start; used for hysteresis / debouncing.
    """

    current_dp_size: int
    target_dp_size: int | None = None
    kv_cache_utilization: float = 0.0
    pending_prefill_tokens: int = 0
    num_active_requests: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class EPRebalanceDecision:
    """Output of the rebalance controller.

    Attributes:
        scale_type: ``"scale_up"``, ``"scale_down"``, or ``"hold"``.
        new_dp_size: target ``data_parallel_size``.  Equal to
            ``current_dp_size`` when ``scale_type == "hold"``.
        reason: short human-readable string explaining the decision
            (logged by the engine controller).
        metadata: free-form dict for downstream telemetry
            (debounce state, hysteresis position, etc.).
    """

    scale_type: Literal["scale_up", "scale_down", "hold"]
    new_dp_size: int
    reason: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def is_hold(self) -> bool:
        return self.scale_type == "hold"


# ---------------------------------------------------------------------------
# Policy interface
# ---------------------------------------------------------------------------


class EPRebalancePolicy(ABC):
    """Abstract rebalance policy.

    Subclasses implement ``decide(signal)`` and return an
    ``EPRebalanceDecision``.  Policies are stateless (decisions are
    idempotent given a signal).  All side effects (engine mutation,
    state machine invocation, NCCL collectives) live in the
    controller, not the policy.
    """

    @abstractmethod
    def decide(self, signal: EPRebalanceSignal) -> EPRebalanceDecision:
        """Compute a rebalance decision from the current signal."""


# ---------------------------------------------------------------------------
# Default policies
# ---------------------------------------------------------------------------


class NoOpEPRebalancePolicy(EPRebalancePolicy):
    """Policy that never rebalances — the default for static deployments."""

    def decide(self, signal: EPRebalanceSignal) -> EPRebalanceDecision:
        return EPRebalanceDecision(
            scale_type="hold",
            new_dp_size=signal.current_dp_size,
            reason="no_op_policy",
        )


class ThresholdEPLoadPolicy(EPRebalancePolicy):
    """Threshold-based scale-up / scale-down on KV cache utilization.

    Decision rules (applied in order):

      1. **Explicit user target** (``signal.target_dp_size`` set):
         always honor it.
           * ``target > current`` → ``"scale_up"``;
           * ``target < current`` → ``"scale_down"``;
           * ``target == current`` → ``"hold"``.
      2. **Utilization-driven**:
           * If ``utilization >= up_threshold`` and we're not
             already at ``max_dp_size`` → ``"scale_up"`` by one
             step.
           * If ``utilization <= down_threshold`` and we're not
             already at ``min_dp_size`` → ``"scale_down"`` by one
             step.
           * Otherwise → ``"hold"``.

    Hysteresis (``min_downtime_seconds``) avoids flapping: the
    policy refuses to issue a scale decision within
    ``min_downtime_seconds`` of the engine start (lets load average
    stabilize before reacting to a cold-start burst).
    """

    def __init__(
        self,
        up_threshold: float = 0.85,
        down_threshold: float = 0.4,
        min_dp_size: int = 1,
        max_dp_size: int = 16,
        min_downtime_seconds: float = 30.0,
    ) -> None:
        if not 0.0 <= down_threshold <= up_threshold <= 1.0:
            raise ValueError(
                f"Require 0.0 <= down_threshold ({down_threshold}) <= "
                f"up_threshold ({up_threshold}) <= 1.0"
            )
        if not 1 <= min_dp_size <= max_dp_size:
            raise ValueError(
                f"Require 1 <= min_dp_size ({min_dp_size}) <= "
                f"max_dp_size ({max_dp_size})"
            )
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.min_dp_size = min_dp_size
        self.max_dp_size = max_dp_size
        self.min_downtime_seconds = min_downtime_seconds

    def decide(self, signal: EPRebalanceSignal) -> EPRebalanceDecision:
        # 1. Explicit user target.
        if (
            signal.target_dp_size is not None
            and signal.target_dp_size != signal.current_dp_size
        ):
            if signal.target_dp_size > signal.current_dp_size:
                target = min(signal.target_dp_size, self.max_dp_size)
                return EPRebalanceDecision(
                    "scale_up",
                    target,
                    reason=f"user_target={signal.target_dp_size}",
                )
            target = max(signal.target_dp_size, self.min_dp_size)
            return EPRebalanceDecision(
                "scale_down",
                target,
                reason=f"user_target={signal.target_dp_size}",
            )

        # Hysteresis — defer until we've been up long enough.
        if signal.elapsed_seconds < self.min_downtime_seconds:
            return EPRebalanceDecision(
                "hold",
                signal.current_dp_size,
                reason="hysteresis: cold start",
                metadata={
                    "elapsed_seconds": signal.elapsed_seconds,
                    "min_downtime_seconds": self.min_downtime_seconds,
                },
            )

        # 2. Utilization-driven.
        util = signal.kv_cache_utilization
        if (
            util >= self.up_threshold
            and signal.current_dp_size < self.max_dp_size
        ):
            return EPRebalanceDecision(
                "scale_up",
                signal.current_dp_size + 1,
                reason=f"util={util:.3f} >= up_threshold={self.up_threshold}",
            )
        if (
            util <= self.down_threshold
            and signal.current_dp_size > self.min_dp_size
        ):
            return EPRebalanceDecision(
                "scale_down",
                signal.current_dp_size - 1,
                reason=f"util={util:.3f} <= down_threshold={self.down_threshold}",
            )
        return EPRebalanceDecision(
            "hold",
            signal.current_dp_size,
            reason=f"util={util:.3f} within [{self.down_threshold}, {self.up_threshold}]",
        )


# ---------------------------------------------------------------------------
# Controller — bridges policy → state machine
# ---------------------------------------------------------------------------


def create_scaling_state_for_decision(
    vllm_config: "VllmConfig",
    decision: EPRebalanceDecision,
) -> "ElasticEPScalingState | None":
    """Build an ``ElasticEPScalingState`` for a rebalance decision.

    Returns ``None`` when ``decision.scale_type == "hold"`` (no
    state machine needed).

    For ``"scale_up"`` and ``"scale_down"``, returns a fresh
    state machine configured for the decision.  The controlling
    engine's worker_type semantics:
      * ``"scale_up"`` → ``"existing"`` (the surviving engines
        are the ones driving the state machine).
      * ``"scale_down"`` → ``"remaining"`` (similarly).
    The "new" / "removing" engines live on the *other side* of
    the same scaling event and are managed by separate engine
    cores — see vLLM ``vllm/v1/engine/core_client.py`` for the
    full SPAWN / SHUTDOWN side.  That orchestration lives at the
    engine-frontend layer; my-vllm's subset keeps the state
    machine self-contained on the controlling side.

    Returns ``None`` if the new ``ParallelConfig`` would be the
    same as the current one (the decision had no effect — log and
    return None instead of constructing a redundant state machine).
    """
    if decision.is_hold:
        return None
    cur = vllm_config.parallel_config
    if cur.data_parallel_size == decision.new_dp_size:
        logger.info(
            "rebalance decision requested dp_size=%d but we're already "
            "at that size; skipping state machine",
            decision.new_dp_size,
        )
        return None

    # Build the new parallel config with the target dp_size.
    new_parallel_config = cur.clone_with_dp_size(decision.new_dp_size)

    # Lazy import: the state machine module owns the ElasticEPScalingState
    # class.  Importing at top level would risk a circular dep with
    # config.vllm → distributed.parallel_state.
    from my_vllm.distributed.elastic_ep.elastic_state import (
        ElasticEPScalingState,
    )

    worker_type = "existing" if decision.scale_type == "scale_up" else "remaining"
    state = ElasticEPScalingState(
        vllm_config=vllm_config,
        new_parallel_config=new_parallel_config,
        worker_type=worker_type,
        scale_type=decision.scale_type,
    )
    logger.info(
        "ElasticEP scaling state created: scale_type=%s worker=%s "
        "dp_size %d → %d (%s)",
        decision.scale_type,
        worker_type,
        cur.data_parallel_size,
        decision.new_dp_size,
        decision.reason,
    )
    return state


# ---------------------------------------------------------------------------
# Convenience: full decide-and-build in one call
# ---------------------------------------------------------------------------


def decide_and_create(
    vllm_config: "VllmConfig",
    signal: EPRebalanceSignal,
    policy: EPRebalancePolicy | None = None,
) -> "ElasticEPScalingState | None":
    """Run the policy + factory in one call.  Returns the state
    machine (or None if hold).  The default policy is
    ``ThresholdEPLoadPolicy(up_threshold=0.85, down_threshold=0.4)``.
    """
    if policy is None:
        policy = ThresholdEPLoadPolicy()
    decision = policy.decide(signal)
    return create_scaling_state_for_decision(vllm_config, decision)


__all__ = [
    "EPRebalanceDecision",
    "EPRebalancePolicy",
    "EPRebalanceSignal",
    "NoOpEPRebalancePolicy",
    "ThresholdEPLoadPolicy",
    "create_scaling_state_for_decision",
    "decide_and_create",
]
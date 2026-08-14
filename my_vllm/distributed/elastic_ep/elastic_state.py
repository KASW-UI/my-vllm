"""Elastic EP scaling state machine — orchestrates scale-up/scale-down.

Real vLLM architecture (vllm/distributed/elastic_ep/elastic_state.py):

  ElasticEPScalingState coordinates the multi-phase state machine for
  elastic expert parallelism (EP) scaling across DP groups.

  Scale-up flow:
    New engines:  PRE_KV_INIT → PREPARE → EPLB_RESHUFFLE → COMPLETE
    Existing:  WAIT_NEW_INIT → CREATE_STANDBY → TRANSFER_MAPPING
            → WAIT_WEIGHTS → TRANSFER_WEIGHTS → SYNC_KV
            → SWITCH_PREPARE → EPLB_RESHUFFLE → COMPLETE

  Scale-down flow:
    Removing:  PREPARE → EPLB_RESHUFFLE → COMPLETE
    Remaining: PREPARE → EPLB_RESHUFFLE → SWITCH_PREPARE → COMPLETE

  Key coordination mechanisms:
    - Two-stage TCPStore barriers for cross-node synchronization
    - Weight transfer via NCCL P2P (sparse for MoE experts)
    - KV cache memory size sync after topology change
    - EPLB reshuffle for load-balanced expert placement
    - Standby group creation for fast failover

C247: wires these mechanisms via **callback hooks** on the state
machine.  Each state transition fires:

  1. ``_barrier_fn(name)`` — TCP barrier with all participants
     (default: no-op logger; vLLM wires a real
     ``TCPStore``-backed barrier).
  2. ``_weight_transfer_fn(src_worker_id, dst_worker_id, kind)`` —
     NCCL weight transfer from src to dst (default: no-op;
     production wires ``NCCLWeightTransferEngine`` from C243).
  3. ``transitions`` / ``barriers`` / ``weight_transfers`` event logs
     for inspection + testing.

The state machine progresses one state per ``progress()`` call.
Call ``run_to_completion()`` to drive the full sequence.
"""

import enum
import logging
import time
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from my_vllm.config import ParallelConfig, VllmConfig

WorkerType = Literal["existing", "new", "removing"]

logger = logging.getLogger(__name__)


class ScaleUpExistingEngineState(enum.IntEnum):
    """States for engines that survive a scale-up."""
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY_SIZE = 5
    SWITCH_AND_PREPARE = 6
    EPLB_RESHUFFLE = 7
    COMPLETE = 8


class ScaleUpNewEngineState(enum.IntEnum):
    """States for newly added engines during scale-up."""
    PRE_KV_INIT = 0
    PREPARE = 1
    EPLB_RESHUFFLE = 2
    COMPLETE = 3


class ScaleDownRemainingEngineState(enum.IntEnum):
    """States for engines that survive a scale-down."""
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    SWITCH_AND_PREPARE = 2
    COMPLETE = 3


class ScaleDownRemovingEngineState(enum.IntEnum):
    """States for engines being removed during scale-down."""
    PREPARE = 0
    EPLB_RESHUFFLE = 1
    COMPLETE = 2


# ---------------------------------------------------------------------------
# Default callbacks (no-ops; production wires real TCP barriers + NCCL
# weight transfer via NCCLWeightTransferEngine).
# ---------------------------------------------------------------------------


def _default_barrier_fn(barrier_name: str, participants: list[str]) -> None:
    """Default barrier — logs only.

    vLLM production wires ``torch.distributed.TCPStore`` with
    ``barrier_name`` as the key and ``participants`` as the
    participating ranks.  my-vllm uses a no-op logger to keep the
    state machine exercisable in unit tests without a real TCPStore.
    """
    logger.debug(
        "ElasticEP barrier: %s (participants=%s)",
        barrier_name,
        participants,
    )


def _default_weight_transfer_fn(
    src_worker_id: int,
    dst_worker_id: int,
    kind: str,
) -> None:
    """Default weight transfer — logs only.

    vLLM production wires ``NCCLWeightTransferEngine`` (C243)
    which broadcasts weights from ``src_worker_id`` to
    ``dst_worker_id`` via NCCL or sparse-NCCL.  my-vllm uses a
    no-op logger + records the call so tests can verify which
    transitions triggered weight transfers.
    """
    logger.debug(
        "ElasticEP weight transfer: %s: %d → %d",
        kind,
        src_worker_id,
        dst_worker_id,
    )


class ElasticEPScalingState:
    """Orchestrates the elastic EP scaling lifecycle.

    Created once per scaling event (scale_up or scale_down) and
    progresses through the state machine via ``progress()`` calls.
    Each step fires the callback hooks (``barrier_fn`` /
    ``weight_transfer_fn``) so the state machine is the *plumbing*
    that bridges pure policy decisions (see
    ``my_vllm.distributed.elastic_ep.rebalance``) to real side effects
    (TCPStore barriers, NCCLWeightTransferEngine broadcasts).

    C247: callback-based design so:

    * Production wires real ``torch.distributed.TCPStore`` + the
      ``NCCLWeightTransferEngine`` (C243) via the ``barrier_fn`` /
      ``weight_transfer_fn`` kwargs.
    * Tests use record-only callbacks to verify which transitions
      fire which barriers / transfers, without a real distributed
      runtime.

    Each ``progress()`` call records the transition (and any
    barrier / transfer fired) in ``self.transitions`` /
    ``self.barriers`` / ``self.weight_transfers`` — the test suite
    inspects these lists to verify behavior.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        new_parallel_config: "ParallelConfig",
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
        *,
        barrier_fn: Callable[[str, list[str]], None] | None = None,
        weight_transfer_fn: Callable[[int, int, str], None] | None = None,
        local_worker_id: int = 0,
        participants: list[str] | None = None,
    ):
        self.vllm_config = vllm_config
        self.new_parallel_config = new_parallel_config
        self.worker_type = worker_type
        self.scale_type = scale_type

        # C247: callback wiring.
        self._barrier_fn = barrier_fn or _default_barrier_fn
        self._weight_transfer_fn = (
            weight_transfer_fn or _default_weight_transfer_fn
        )
        # The local worker's rank in the DP group (used for self-targeted
        # barriers + transfer ids).
        self.local_worker_id = local_worker_id
        # Names / ids of all DP participants in this scaling event.
        # Defaults to a single placeholder for tests; vLLM wires real
        # DP ranks at construction.
        self.participants = participants or [f"worker-{local_worker_id}"]

        if scale_type == "scale_up":
            if worker_type == "new":
                self.state = ScaleUpNewEngineState.PRE_KV_INIT
            else:
                self.state = ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT
        else:
            if worker_type == "removing":
                self.state = ScaleDownRemovingEngineState.PREPARE
            else:
                self.state = ScaleDownRemainingEngineState.PREPARE

        self._start_time = time.monotonic()

        # Event logs (C247): tests inspect these to verify which
        # callbacks fired at each transition.  Format:
        #   self.transitions: list of (from_state, to_state, scale_type)
        #   self.barriers: list of (barrier_name, transition_index)
        #   self.weight_transfers: list of (src_id, dst_id, kind, transition_index)
        self.transitions: list[tuple[Any, Any, str]] = []
        self.barriers: list[tuple[str, str]] = []
        self.weight_transfers: list[tuple[int, int, str]] = []

    def progress(self) -> bool:
        """Advance the state machine by one step.

        Returns ``True`` if the state advanced (or we were already
        at ``COMPLETE``).

        The engine controller loop calls this until
        ``is_complete`` is ``True``.  Each call fires the callback
        hooks for the transition (barrier for state-specific
        synchronization, weight transfer for state-specific
        replication).
        """
        if self.is_complete:
            return True
        if self.scale_type == "scale_up":
            return self._progress_scale_up()
        else:
            return self._progress_scale_down()

    def run_to_completion(self, max_steps: int = 64) -> int:
        """Drive the state machine to ``COMPLETE`` and return steps taken.

        Args:
            max_steps: Upper bound on the number of successful
                transitions to allow.  Defaults to 64 (each path has
                at most ~10 states, so 64 is comfortable headroom).
                ``max_steps=0`` is legal — no transitions are taken
                and ``steps=0`` is returned.

        Returns:
            Number of transitions traversed (= number of
            ``progress()`` calls that returned ``True``).  Returns
            ``max_steps`` exactly when the loop hit the guard
            without completing — a misconfiguration bug.

        Loop semantics: ``while not is_complete and steps_completed
        < max_steps: progress(); steps_completed += 1``.  This makes
        ``max_steps=0`` a true no-op (zero transitions).
        """
        steps = 0
        while not self.is_complete and steps < max_steps:
            if not self.progress():
                break  # progress() blocked (real impl: waiting on barrier)
            steps += 1
        if not self.is_complete and steps >= max_steps:
            # Reached max_steps without completing — log + return
            # the count (caller can compare against max_steps to
            # detect overflow).
            logger.warning(
                "ElasticEP run_to_completion hit max_steps=%d without "
                "reaching COMPLETE",
                max_steps,
            )
        return steps

    def _progress_scale_up(self) -> bool:
        if self.worker_type == "new":
            return self._progress_scale_up_new()
        else:
            return self._progress_scale_up_existing()

    def _progress_scale_down(self) -> bool:
        if self.worker_type == "removing":
            return self._progress_scale_down_removing()
        else:
            return self._progress_scale_down_remaining()

    def _advance(
        self,
        next_state: Any,
        barrier_name: str | None = None,
        weight_transfer: tuple[int, str] | None = None,
    ) -> bool:
        """Transition bookkeeping helper.

        Fires the barrier (if any), fires the weight transfer (if
        any), records the transition, then updates ``self.state``.

        Args:
            next_state: The state to transition to.
            barrier_name: If set, fire ``self._barrier_fn(barrier_name,
                self.participants)`` before the transition and log the
                barrier in ``self.barriers``.
            weight_transfer: If set, fire
                ``self._weight_transfer_fn(src_worker_id, self.local_worker_id,
                kind)`` before the transition and log the transfer in
                ``self.weight_transfers``.  The tuple is
                ``(src_worker_id, kind)`` — ``dst_worker_id`` is the
                local worker (the receiver).

        Returns:
            Always ``True`` (the advance succeeded; vLLM's real
            implementations return ``False`` when blocked on a
            barrier, but my-vllm's hooks are synchronous).
        """
        transition_index = len(self.transitions)

        # Fire barrier BEFORE the state update (mirrors vLLM:
        # barrier-then-local-state-change).
        if barrier_name is not None:
            self._barrier_fn(barrier_name, self.participants)
            self.barriers.append((barrier_name, self.scale_type))

        # Fire weight transfer BEFORE the state update so receivers
        # have the new weights when they enter the next state.
        if weight_transfer is not None:
            src_worker_id, kind = weight_transfer
            self._weight_transfer_fn(
                src_worker_id, self.local_worker_id, kind,
            )
            self.weight_transfers.append(
                (src_worker_id, self.local_worker_id, kind),
            )

        # Record + apply the transition.
        from_state = self.state
        self.transitions.append(
            (from_state, next_state, self.scale_type),
        )
        self.state = next_state
        return True

    def _progress_scale_up_new(self) -> bool:
        """New-engine scale-up: PRE_KV_INIT → PREPARE → EPLB_RESHUFFLE → COMPLETE.

        The "new" engine path is light (the "existing" engines
        drive the heavy lifting); we only emit barriers + the
        ``TRANSFER_EXPERT_MAPPING`` action on PREPARE so the new
        engine receives its expert assignment via the ``existing``
        engines' broadcasts.
        """
        if self.state == ScaleUpNewEngineState.PRE_KV_INIT:
            return self._advance(
                ScaleUpNewEngineState.PREPARE,
                barrier_name="PRE_KV_INIT",
            )
        if self.state == ScaleUpNewEngineState.PREPARE:
            return self._advance(
                ScaleUpNewEngineState.EPLB_RESHUFFLE,
                barrier_name="PREPARE_NEW",
                weight_transfer=(self.local_worker_id, "expert_mapping"),
            )
        if self.state == ScaleUpNewEngineState.EPLB_RESHUFFLE:
            return self._advance(
                ScaleUpNewEngineState.COMPLETE,
                barrier_name="EPLB_RESHUFFLE",
            )
        return True  # already at COMPLETE

    def _progress_scale_up_existing(self) -> bool:
        """Existing-engine scale-up: full 9-state protocol.

        State map (transitions + emitted hook):

            0. WAIT_NEW_CORE_ENGINES_INIT
               → CREATE_STANDBY_GROUPS
                  barrier: WAIT_NEW_CORE_INIT
            1. CREATE_STANDBY_GROUPS
               → TRANSFER_EXPERT_MAPPING
                  barrier: CREATE_STANDBY
            2. TRANSFER_EXPERT_MAPPING
               → WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
                  barrier: TRANSFER_EXPERT_MAPPING
                  weight_transfer: (src=existing, kind="expert_mapping")
            3. WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
               → TRANSFER_WEIGHTS
                  barrier: WAIT_NEW_CORE_WEIGHTS_INIT
            4. TRANSFER_WEIGHTS
               → SYNC_KV_CACHE_MEMORY_SIZE
                  barrier: TRANSFER_WEIGHTS
                  weight_transfer: (src=existing, kind="weights")
            5. SYNC_KV_CACHE_MEMORY_SIZE
               → SWITCH_AND_PREPARE
                  barrier: SYNC_KV_CACHE_MEMORY_SIZE
            6. SWITCH_AND_PREPARE
               → EPLB_RESHUFFLE
                  barrier: SWITCH_AND_PREPARE
            7. EPLB_RESHUFFLE
               → COMPLETE
                  barrier: EPLB_RESHUFFLE
        """
        if self.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_INIT:
            return self._advance(
                ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS,
                barrier_name="WAIT_NEW_CORE_INIT",
            )
        if self.state == ScaleUpExistingEngineState.CREATE_STANDBY_GROUPS:
            return self._advance(
                ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING,
                barrier_name="CREATE_STANDBY",
            )
        if self.state == ScaleUpExistingEngineState.TRANSFER_EXPERT_MAPPING:
            return self._advance(
                ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT,
                barrier_name="TRANSFER_EXPERT_MAPPING",
                weight_transfer=(self.local_worker_id, "expert_mapping"),
            )
        if self.state == ScaleUpExistingEngineState.WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT:
            return self._advance(
                ScaleUpExistingEngineState.TRANSFER_WEIGHTS,
                barrier_name="WAIT_NEW_CORE_WEIGHTS_INIT",
            )
        if self.state == ScaleUpExistingEngineState.TRANSFER_WEIGHTS:
            return self._advance(
                ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE,
                barrier_name="TRANSFER_WEIGHTS",
                weight_transfer=(self.local_worker_id, "weights"),
            )
        if self.state == ScaleUpExistingEngineState.SYNC_KV_CACHE_MEMORY_SIZE:
            return self._advance(
                ScaleUpExistingEngineState.SWITCH_AND_PREPARE,
                barrier_name="SYNC_KV_CACHE_MEMORY_SIZE",
            )
        if self.state == ScaleUpExistingEngineState.SWITCH_AND_PREPARE:
            return self._advance(
                ScaleUpExistingEngineState.EPLB_RESHUFFLE,
                barrier_name="SWITCH_AND_PREPARE",
            )
        if self.state == ScaleUpExistingEngineState.EPLB_RESHUFFLE:
            return self._advance(
                ScaleUpExistingEngineState.COMPLETE,
                barrier_name="EPLB_RESHUFFLE",
            )
        return True  # already at COMPLETE

    def _progress_scale_down_removing(self) -> bool:
        """Removing-engine scale-down: PREPARE → EPLB_RESHUFFLE → COMPLETE."""
        if self.state == ScaleDownRemovingEngineState.PREPARE:
            return self._advance(
                ScaleDownRemovingEngineState.EPLB_RESHUFFLE,
                barrier_name="PREPARE_REMOVING",
            )
        if self.state == ScaleDownRemovingEngineState.EPLB_RESHUFFLE:
            return self._advance(
                ScaleDownRemovingEngineState.COMPLETE,
                barrier_name="EPLB_RESHUFFLE",
            )
        return True

    def _progress_scale_down_remaining(self) -> bool:
        """Remaining-engine scale-down: PREPARE → EPLB_RESHUFFLE → SWITCH_AND_PREPARE → COMPLETE."""
        if self.state == ScaleDownRemainingEngineState.PREPARE:
            return self._advance(
                ScaleDownRemainingEngineState.EPLB_RESHUFFLE,
                barrier_name="PREPARE_REMAINING",
            )
        if self.state == ScaleDownRemainingEngineState.EPLB_RESHUFFLE:
            return self._advance(
                ScaleDownRemainingEngineState.SWITCH_AND_PREPARE,
                barrier_name="EPLB_RESHUFFLE",
            )
        if self.state == ScaleDownRemainingEngineState.SWITCH_AND_PREPARE:
            return self._advance(
                ScaleDownRemainingEngineState.COMPLETE,
                barrier_name="SWITCH_AND_PREPARE",
            )
        return True

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def is_complete(self) -> bool:
        if self.scale_type == "scale_up":
            return (
                self.state == ScaleUpNewEngineState.COMPLETE
                if self.worker_type == "new"
                else self.state == ScaleUpExistingEngineState.COMPLETE
            )
        else:
            return (
                self.state == ScaleDownRemovingEngineState.COMPLETE
                if self.worker_type == "removing"
                else self.state == ScaleDownRemainingEngineState.COMPLETE
            )


__all__ = [
    "ScaleUpExistingEngineState",
    "ScaleUpNewEngineState",
    "ScaleDownRemainingEngineState",
    "ScaleDownRemovingEngineState",
    "ElasticEPScalingState",
    "WorkerType",
    "_default_barrier_fn",
    "_default_weight_transfer_fn",
]

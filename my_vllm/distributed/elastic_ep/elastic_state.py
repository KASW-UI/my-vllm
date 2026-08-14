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
"""

import enum
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, VllmConfig
    from vllm.v1.executor.abstract import Executor

WorkerType = Literal["existing", "new", "removing"]


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


class ElasticEPScalingState:
    """Orchestrates the elastic EP scaling lifecycle.

    Created once per scaling event (scale_up or scale_down) and
    progresses through the state machine via progress() calls.
    Each step typically involves:
      1. Local state transition (model reconfiguration, weight loading)
      2. Barrier synchronization across all DP ranks
      3. Communication operations (weight transfer, EPLB reshuffle)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        new_parallel_config: "ParallelConfig",
        worker_type: WorkerType,
        scale_type: Literal["scale_up", "scale_down"],
    ):
        self.vllm_config = vllm_config
        self.new_parallel_config = new_parallel_config
        self.worker_type = worker_type
        self.scale_type = scale_type

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

    def progress(self) -> bool:
        """Advance the state machine by one step.

        Returns ``True`` if the state advanced (or we were already
        at ``COMPLETE``).  Returns ``False`` if blocked (e.g.
        waiting on a barrier — the real implementation hooks here).

        In vLLM the engine controller loop calls this until it
        returns ``True`` AND ``is_complete`` is ``True``; the
        skeleton here always advances (no real barriers yet).
        """
        if self.is_complete:
            return True
        if self.scale_type == "scale_up":
            return self._progress_scale_up()
        else:
            return self._progress_scale_down()

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

    def _progress_scale_up_new(self) -> bool:
        states = list(ScaleUpNewEngineState)
        idx = states.index(self.state)
        if idx + 1 >= len(states):
            return True
        self.state = states[idx + 1]
        return True  # state advanced.

    def _progress_scale_up_existing(self) -> bool:
        states = list(ScaleUpExistingEngineState)
        idx = states.index(self.state)
        if idx + 1 >= len(states):
            return True
        self.state = states[idx + 1]
        return True  # state advanced.

    def _progress_scale_down_removing(self) -> bool:
        states = list(ScaleDownRemovingEngineState)
        idx = states.index(self.state)
        if idx + 1 >= len(states):
            return True
        self.state = states[idx + 1]
        return True  # state advanced.

    def _progress_scale_down_remaining(self) -> bool:
        states = list(ScaleDownRemainingEngineState)
        idx = states.index(self.state)
        if idx + 1 >= len(states):
            return True
        self.state = states[idx + 1]
        return True  # state advanced.

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

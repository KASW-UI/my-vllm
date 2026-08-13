"""DP Coordinator — data-parallel coordination process.

Mirrors vLLM V1 v1/engine/coordinator.py.

vLLM's `DPCoordinator` is a coordinator process that intermediates
between DP engine rank processes and front-end API server processes
when `data_parallel_size > 1`. Responsibilities:
  - Collect per-DP-engine stats (waiting / running queue lengths)
    and publish them to front-ends for load balancing.
  - Track the DP "request wave" number — a counter incremented
    every time the engines transition from running to paused.
  - Broadcast START_DP_WAVE messages to wake engines when new
    requests arrive.

C142: minimal subset — `DPCoordinator` class with the
public-API surface (`__init__`, `start`, `stop`,
`publish_stats`). The full vLLM impl uses ZMQ + multiprocessing
pipes + msgspec msgpack serialization; my-vllm's version is a
synchronous in-process stub. For DP=1 (my-vllm's default) the
coordinator is never instantiated.

The class exists for 1:1 file-tree alignment; a follow-up wires
real ZMQ sockets if/when DP>1 production rollout is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.config.parallel import ParallelConfig


logger = logging.getLogger(__name__)


@dataclass
class DPStats:
    """Per-DP-engine statistics published to front-ends.

    Mirrors vLLM V1 v1/engine/coordinator.py (DPStats shape).
    """

    dp_rank: int
    waiting: int = 0
    running: int = 0
    wave_number: int = 0
    is_running: bool = True


@dataclass
class DPCoordinator:
    """Coordinator for DP deployments.

    Mirrors vLLM V1 v1/engine/coordinator.py:DPCoordinator (L23).

    C142: minimal — a synchronous in-process stub. No ZMQ / pipe
    plumbing. The `stats_history` is exposed for front-end
    inspection; the `wave_number` is a single counter incremented
    via `tick_wave()`.
    """

    parallel_config: "ParallelConfig"
    enable_wave_coordination: bool = True
    dp_size: int = 1
    stats_history: list[DPStats] = field(default_factory=list)
    wave_number: int = 0
    is_running: bool = True
    _started: bool = False

    def __post_init__(self) -> None:
        if hasattr(self.parallel_config, "data_parallel_size"):
            self.dp_size = self.parallel_config.data_parallel_size or 1

    def start(self) -> None:
        """Start the coordinator (no-op for DP=1)."""
        if self._started:
            return
        self._started = True
        logger.info(
            "DPCoordinator started (dp_size=%d, wave_coordination=%s)",
            self.dp_size, self.enable_wave_coordination,
        )

    def stop(self) -> None:
        """Stop the coordinator."""
        self._started = False
        logger.info("DPCoordinator stopped")

    def publish_stats(self, stats: DPStats) -> None:
        """Record a stats snapshot from one DP engine."""
        if self.dp_size == 1:
            return
        self.stats_history.append(stats)

    def tick_wave(self) -> None:
        """Increment the request-wave counter (paused → running).

        C142: stub — increments `self.wave_number`. The full vLLM
        impl broadcasts START_DP_WAVE to all engines via ZMQ.
        """
        if self.enable_wave_coordination:
            self.wave_number += 1
            self.is_running = True

    def pause(self) -> None:
        """Mark the coordinator paused (waiting for new request)."""
        self.is_running = False


__all__ = ["DPCoordinator", "DPStats"]
"""Stat loggers — aligns with vLLM V1 v1/metrics/loggers.py.

C048: StatLoggerBase ABC + LoggingStatLogger (prints to stdout).
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.v1.metrics.stats import SchedulerStats, IterationStats

logger = logging.getLogger(__name__)


class StatLoggerBase(ABC):
    """Abstract stat logger."""

    @abstractmethod
    def record(
        self,
        scheduler_stats: "SchedulerStats",
        iteration_stats: "IterationStats | None" = None,
    ) -> None:
        """Record scheduler and optional iteration stat."""


class LoggingStatLogger(StatLoggerBase):
    """Logs stats to Python logging (stdout).

    Emits SchedulerStats every N steps and IterationStats per step.
    """

    def __init__(self, interval_steps: int = 10) -> None:
        self._step_count: int = 0
        self._interval_steps = interval_steps

    def record(
        self,
        scheduler_stats: "SchedulerStats",
        iteration_stats: "IterationStats | None" = None,
    ) -> None:
        self._step_count += 1
        if self._step_count % self._interval_steps == 0:
            logger.info("%s", scheduler_stats)
        if iteration_stats is not None:
            logger.debug("%s", iteration_stats)


class NoopStatLogger(StatLoggerBase):
    """No-op logger for when metrics are disabled."""

    def record(
        self,
        scheduler_stats: "SchedulerStats",
        iteration_stats: "IterationStats | None" = None,
    ) -> None:
        pass

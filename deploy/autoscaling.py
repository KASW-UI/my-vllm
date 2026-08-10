"""AutoscalingController — queue_depth based scaling for LLM inference.

C093: monitoring-based decision loop for replica up/down scaling.
LLM-specific: cold start (30-300s model load), queue_depth > GPU util.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.v1.metrics.stats import SchedulerStats


@dataclass
class ScalingDecision:
    action: str  # "scale_up" | "scale_down" | "noop"
    target_replicas: int
    reason: str


class AutoscalingController:
    """Queue-based autoscaling for LLM inference.

    Monitors queue_depth and avg_ttft to decide when to scale.
    Key insight: GPU util is near 100% in continuous batching,
    so queue_depth is the primary signal for scaling.
    """

    def __init__(
        self,
        min_replicas: int = 2,
        max_replicas: int = 8,
        scale_up_threshold: int = 10,
        scale_down_threshold: int = 3,
        scale_up_window_s: float = 30.0,
        scale_down_window_s: float = 300.0,
    ) -> None:
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.scale_up_window_s = scale_up_window_s
        self.scale_down_window_s = scale_down_window_s

        self.current_replicas = min_replicas
        self._window_history: list[tuple[float, int]] = []

    def update(self, scheduler_stats: "SchedulerStats") -> ScalingDecision | None:
        """Record current metrics and make scaling decision."""
        queue_depth = scheduler_stats.num_waiting_reqs
        now = time.time()

        self._window_history.append((now, queue_depth))

        cutoff_up = now - self.scale_up_window_s
        cutoff_down = now - self.scale_down_window_s
        self._window_history = [
            (t, d) for t, d in self._window_history if t >= cutoff_down
        ]

        recent = [d for t, d in self._window_history if t >= cutoff_up]
        if recent and all(d > self.scale_up_threshold for d in recent):
            if self.current_replicas < self.max_replicas:
                self.current_replicas += 1
                return ScalingDecision(
                    "scale_up", self.current_replicas,
                    f"queue_depth > {self.scale_up_threshold} for {self.scale_up_window_s}s"
                )

        older = [d for t, d in self._window_history]
        if older and all(d < self.scale_down_threshold for d in older):
            if self.current_replicas > self.min_replicas:
                self.current_replicas -= 1
                return ScalingDecision(
                    "scale_down", self.current_replicas,
                    f"queue_depth < {self.scale_down_threshold} for {self.scale_down_window_s}s"
                )

        return None

    def cold_start_time_estimate(self, model_params: int) -> float:
        """Rough estimate: ~30s for small models, ~300s for large models."""
        if model_params < 1e9:
            return 30.0
        elif model_params < 10e9:
            return 60.0
        elif model_params < 100e9:
            return 180.0
        return 300.0

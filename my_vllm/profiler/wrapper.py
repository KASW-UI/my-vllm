"""
Worker profiler — controls when profiling starts/stops during model execution.

Architecture:
  WorkerProfiler ABC — base class with warmup iters and max iters
  TorchProfiler     — wraps torch.profiler.profile for general profiling
  CUDAGraphProfiler — specialized for CUDA graph capture profiling

The profiler is integrated into GPUModelRunner.execute_model(), gated by
env vars or config flags (VLLM_PROFILER_ENABLED).
"""

from abc import ABC, abstractmethod


class WorkerProfiler(ABC):
    """Abstract base for worker-level profilers.

    Controls profiling lifecycle: warmup iterations, profiling iterations,
    start/stop callbacks.  Subclasses implement the actual profiling backend
    (torch.profiler, nsys, CUPTI, etc.).
    """

    def __init__(self, delay_iters: int = 5, max_iters: int = 20):
        self._delay_iters = delay_iters
        self._max_iters = max_iters
        self._current_iter = 0
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def step(self) -> bool:
        """Call once per engine step.  Returns True while profiling is active."""
        self._current_iter += 1
        if self._current_iter == self._delay_iters:
            self.start()
        if self._current_iter >= self._delay_iters + self._max_iters:
            self.stop()
            self._active = False
        return self._active

    @abstractmethod
    def start(self) -> None:
        """Start the profiler."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the profiler and collect results."""
        raise NotImplementedError


class TorchProfiler(WorkerProfiler):
    """Wraps torch.profiler.profile with activities=[CPU, CUDA].

    Collects forward/backward CUDA events across warmup + profiling steps,
    and produces LayerwiseProfileResults for per-layer/per-op analysis.
    """

    def __init__(
        self,
        delay_iters: int = 5,
        max_iters: int = 20,
        record_shapes: bool = True,
        profile_memory: bool = False,
        with_stack: bool = True,
    ):
        super().__init__(delay_iters=delay_iters, max_iters=max_iters)
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self._results: "LayerwiseProfileResults | None" = None

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    @property
    def results(self):
        return self._results


class CUDAGraphProfiler(WorkerProfiler):
    """Profiler for CUDA graph capture.

    CUDA graphs bundle many kernel launches into a single replay.
    This profiler captures the individual kernel events within
    a graph's capture period for latency breakdown.
    """

    def __init__(self, delay_iters: int = 3, max_iters: int = 5):
        super().__init__(delay_iters=delay_iters, max_iters=max_iters)

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

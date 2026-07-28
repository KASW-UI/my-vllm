"""Executor ABC — mirrors vLLM V1 v1/executor/abstract.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C028: enhances with factory method get_class(), initialize_cache(),
and full docstrings for the executor extension point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.v1.core.sched.output import SchedulerOutput
    from my_vllm.v1.outputs import ModelRunnerOutput


class Executor(ABC):
    """Abstract executor managing model execution on one or more devices.

    An executor is responsible for:
      - Owning one or more Worker instances
      - Dispatching SchedulerOutput to workers via execute_model()
      - Providing worker access for EngineCore state queries

    Subclasses:
      UniProcExecutor  — single worker, same-process (default)
      (future) MultiProcExecutor — multiple workers, multiprocess
    """

    uses_ray: bool = False
    supports_pp: bool = False

    @staticmethod
    def get_class(backend: str = "uni") -> type[Executor]:
        """Factory: select executor implementation by backend name.

        Args:
            backend: "uni" | "mp" (multiproc, stub at C028)

        Returns:
            Executor subclass to instantiate.
        """
        if backend == "mp":
            from my_vllm.v1.executor.uniproc_executor import UniProcExecutor

            return UniProcExecutor

        from my_vllm.v1.executor.uniproc_executor import UniProcExecutor

        return UniProcExecutor

    @abstractmethod
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
        """Execute the model on the given scheduler output.

        Args:
            scheduler_output: structured scheduling metadata.

        Returns:
            ModelRunnerOutput with per-request sampled token IDs,
            or None if no work was scheduled.
        """
        ...

    @abstractmethod
    def get_worker(self):
        """Return the primary worker for state queries (get_step, cleanup)."""

    def initialize_cache(
        self, num_kv_blocks: int, block_size: int
    ) -> None:
        """Initialize KV cache parameters on the executor.

        Called once at startup. Subclasses may override to configure
        per-worker cache sizes.
        """
        pass

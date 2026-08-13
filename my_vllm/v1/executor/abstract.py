"""Executor ABC — mirrors vLLM V1 v1/executor/abstract.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C028: enhances with factory method get_class(), initialize_cache(),
and full docstrings for the executor extension point.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
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
      MultiprocExecutor — multiple workers, multiprocess
    """

    supports_pp: bool = False

    @staticmethod
    def get_class(vllm_config: VllmConfig) -> type[Executor]:
        """Factory: select executor implementation from VllmConfig.

        Mirrors vLLM V1 Executor.get_class(vllm_config) (subset —
        only `mp` and `uni` backends. Ray is excluded per
        `docs/alignment-plan.md` §0.)
        """
        backend = getattr(
            vllm_config.parallel_config,
            "distributed_executor_backend",
            None,
        )
        if backend is None:
            if vllm_config.parallel_config.world_size > 1:
                backend = "mp"
            else:
                backend = "uni"

        if backend == "mp":
            from my_vllm.v1.executor.multiproc_executor import MultiprocExecutor
            return MultiprocExecutor

        from my_vllm.v1.executor.uniproc_executor import UniProcExecutor
        return UniProcExecutor

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config

    @abstractmethod
    def _init_executor(self) -> None:
        """Initialize the executor — create workers and load models."""

    @abstractmethod
    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput:
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
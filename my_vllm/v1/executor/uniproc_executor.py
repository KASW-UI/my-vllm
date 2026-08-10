"""UniProcExecutor — single-process executor.

Mirrors vLLM V1 v1/executor/uniproc_executor.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C028: enhanced with initialize_cache(), implements Executor ABC.
C032: creates Worker internally — matches vLLM's executor-driven
model loading chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu_worker import Worker
from my_vllm.v1.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


class UniProcExecutor(Executor):
    """Single-process executor — mirrors vLLM V1 UniProcExecutor.

    Receives VllmConfig via Executor.__init__, then _init_executor()
    creates Worker and loads the model.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        self._worker = Worker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method="env://",
            is_driver_worker=True,
        )

    def _init_executor(self):
        """Mirrors vLLM Executor._init_executor(): trigger model loading."""
        self._worker.init_device()

    @property
    def worker(self) -> WorkerBase:
        return self._worker

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        return self.worker.execute_model(scheduler_output)

    def get_worker(self) -> WorkerBase:
        return self.worker

    def initialize_cache(
        self, num_kv_blocks: int, block_size: int
    ) -> None:
        pass
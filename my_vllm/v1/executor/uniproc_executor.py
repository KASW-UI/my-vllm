"""UniProcExecutor — single-process executor.

Mirrors vLLM V1 v1/executor/uniproc_executor.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C028: enhanced with initialize_cache(), proper docstrings,
and implements the full Executor ABC interface.
"""

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase


class UniProcExecutor(Executor):
    """Single-process executor with one Worker.

    All model execution and state queries are performed
    synchronously in the current process via the worker.
    """

    def __init__(self, worker: WorkerBase):
        self._worker = worker

    @property
    def worker(self) -> WorkerBase:
        return self._worker

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """Delegates model execution directly to the worker."""
        return self.worker.execute_model(scheduler_output)

    def get_worker(self) -> WorkerBase:
        """Return the owned worker for state queries."""
        return self.worker

    def initialize_cache(
        self, num_kv_blocks: int, block_size: int
    ) -> None:
        """No-op: single-process executor shares cache with EngineCore."""
        pass

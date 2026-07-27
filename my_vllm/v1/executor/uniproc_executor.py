"""UniProcExecutor — single-process executor.

Mirrors vLLM V1 v1/executor/uniproc_executor.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
"""

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase


class UniProcExecutor(Executor):
    """Single-process executor with one Worker."""

    def __init__(self, worker: WorkerBase):
        self.worker = worker

    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        return self.worker.execute_model(scheduler_output)

    def get_worker(self) -> WorkerBase:
        return self.worker

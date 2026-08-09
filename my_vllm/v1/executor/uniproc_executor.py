"""UniProcExecutor — single-process executor.

Mirrors vLLM V1 v1/executor/uniproc_executor.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C028: enhanced with initialize_cache(), implements Executor ABC.
C032: creates Worker internally — matches vLLM's executor-driven
model loading chain.
"""

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu_worker import Worker
from my_vllm.v1.worker.worker_base import WorkerBase


class UniProcExecutor(Executor):
    """Single-process executor — mirrors vLLM V1 UniProcExecutor.

    C032: creates Worker and calls init_device() internally,
    matching vLLM's _init_executor() → init_worker() → init_device()
    → load_model() chain.
    """

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        dtype_str: str = "bfloat16",
        eos_token_id: int = 2,
        block_size: int = 16,
        max_num_blocks: int = 256,
        max_num_reqs: int = 64,
        rank: int = 0,
        world_size: int = 1,
    ):
        self._worker = Worker(
            model_name=model_name,
            dtype_str=dtype_str,
            eos_token_id=eos_token_id,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            max_num_reqs=max_num_reqs,
            rank=rank,
            world_size=world_size,
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

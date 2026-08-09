"""GPU Worker — mirrors vLLM V1 v1/worker/gpu_worker.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C021: accepts ModelState for model lifecycle.
C032: Worker loads model internally via GPUModelRunner.load_model()
— model is no longer passed in from run.py.  Matches vLLM lifecycle.
"""

from my_vllm.v1.distributed.parallel_state import init_distributed
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner


class Worker(WorkerBase):
    """GPU Worker owning ModelRunner — mirrors vLLM V1 Worker.

    C032: receives EngineConfig and creates GPUModelRunner internally.
    The worker calls model_runner.load_model() during init_device(),
    matching vLLM's deep model loading chain:
    Executor → Worker.init_device() → GPUModelRunner.load_model()
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
        super().__init__(rank=rank, world_size=world_size)

        self.model_name = model_name
        self.dtype_str = dtype_str
        self.eos_token_id = eos_token_id
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.max_num_reqs = max_num_reqs
        self._model_runner: GPUModelRunner | None = None

    def init_device(self):
        """Mirrors vLLM Worker.init_device(): init dist + load model."""
        init_distributed(
            world_size=self.world_size, rank=self.rank
        )
        self._model_runner = GPUModelRunner(
            model_name=self.model_name,
            dtype_str=self.dtype_str,
            eos_token_id=self.eos_token_id,
            block_size=self.block_size,
            max_num_blocks=self.max_num_blocks,
            max_num_reqs=self.max_num_reqs,
        )
        self._model_runner.load_model()

    def _runner(self) -> GPUModelRunner:
        if self._model_runner is None:
            self.init_device()
        return self._model_runner

    def execute_model(
        self, scheduler_output
    ) -> ModelRunnerOutput:
        return self._runner().execute_model(scheduler_output)

    def get_step(self, request_id: str) -> int:
        return self._runner().get_step(request_id)

    def cleanup(self, request_id: str):
        self._runner().cleanup(request_id)

"""GPU Worker — mirrors vLLM V1 v1/worker/gpu_worker.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
GPUModelRunner replaces per-request prefill/decode.
"""

import torch
import torch.nn as nn

from my_vllm.cache.kv_cache import KVCache
from my_vllm.distributed.parallel_state import init_distributed
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner


class Worker(WorkerBase):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eos_token_id: int = 2,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(rank=rank, world_size=world_size)
        self.model = model
        self.device = device
        self.eos_token_id = eos_token_id
        self._model_runner: GPUModelRunner | None = None

    def init_device(self):
        init_distributed(
            world_size=self.world_size, rank=self.rank
        )
        self.model.eval()
        self._model_runner = GPUModelRunner(
            self.model, self.device, eos_token_id=self.eos_token_id
        )

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

    # Legacy bridge
    def execute_prefill(
        self, request_id: str,
        input_ids: torch.Tensor, cache: KVCache
    ) -> int:
        return self._runner()._prefill_one(request_id, input_ids, cache)

    def execute_decode(
        self, request_id: str, cache: KVCache
    ) -> int:
        return self._runner()._decode_one_legacy(request_id, cache)

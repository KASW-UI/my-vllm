import torch
import torch.nn as nn

from my_vllm.worker.worker_base import WorkerBase
from my_vllm.worker.model_runner import ModelRunner
from my_vllm.cache.kv_cache import KVCache
from my_vllm.distributed.parallel_state import init_distributed


class Worker(WorkerBase):
    """GPU Worker — owns model and ModelRunner (C013).

    C013: accepts rank / world_size for Tensor Parallel.
    When world_size=1, init_distributed is a no-op and TP
    layers degrade to regular nn.Linear.
    """

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
        self._model_runner: ModelRunner | None = None

    def init_device(self):
        init_distributed(
            world_size=self.world_size, rank=self.rank
        )
        self.model.eval()
        self._model_runner = ModelRunner(
            self.model, self.device, eos_token_id=self.eos_token_id
        )

    def _runner(self) -> ModelRunner:
        if self._model_runner is None:
            self.init_device()
        return self._model_runner

    def execute_prefill(
        self, request_id: str,
        input_ids: torch.Tensor, cache: KVCache
    ) -> int:
        return self._runner().prefill(request_id, input_ids, cache)

    def execute_decode(
        self, request_id: str, cache: KVCache
    ) -> int:
        return self._runner().decode(request_id, cache)

    def get_step(self, request_id: str) -> int:
        return self._runner().get_step(request_id)

    def cleanup(self, request_id: str):
        self._runner().cleanup(request_id)

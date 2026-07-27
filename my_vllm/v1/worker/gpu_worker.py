"""GPU Worker — mirrors vLLM V1 v1/worker/gpu_worker.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C021: accepts ModelState for model lifecycle; backward compat
preserved via direct model+device constructor.
"""

import torch
import torch.nn as nn

from my_vllm.cache.kv_cache import KVCache
from my_vllm.distributed.parallel_state import init_distributed
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner
from my_vllm.v1.worker.gpu.model_states.interface import ModelState


class _DirectModelState(ModelState):
    """Internal wrapper when model + device are passed directly."""

    def __init__(self, model: nn.Module, device: torch.device):
        self._model = model
        self._device = device
        self._dtype = next(model.parameters()).dtype

    def load_model(self) -> nn.Module:
        return self._model

    def get_device(self) -> torch.device:
        return self._device

    def get_dtype(self) -> torch.dtype:
        return self._dtype


class Worker(WorkerBase):
    """GPU Worker owning ModelRunner.

    C021: accepts ModelState for model lifecycle management.
    Backward compat: can also receive model+device directly.
    """

    def __init__(
        self,
        model: nn.Module | None = None,
        device: torch.device | None = None,
        model_state: ModelState | None = None,
        eos_token_id: int = 2,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(rank=rank, world_size=world_size)

        if model_state is not None:
            self.model_state = model_state
        elif model is not None and device is not None:
            self.model_state = _DirectModelState(model, device)
        else:
            raise ValueError(
                "Provide model_state or (model, device)"
            )

        self.eos_token_id = eos_token_id
        self._model_runner: GPUModelRunner | None = None

    def init_device(self):
        init_distributed(
            world_size=self.world_size, rank=self.rank
        )
        model = self.model_state.load_model()
        device = self.model_state.get_device()
        self._model_runner = GPUModelRunner(
            model, device, eos_token_id=self.eos_token_id
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

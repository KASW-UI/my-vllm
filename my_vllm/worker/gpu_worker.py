import torch
import torch.nn as nn

from my_vllm.worker.worker_base import WorkerBase
from my_vllm.worker.model_runner import ModelRunner


class Worker(WorkerBase):
    """GPU Worker — owns model and ModelRunner (C011).

    Encapsulates device initialization and model lifecycle.
    StateManager accesses ModelRunner through Worker.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eos_token_id: int = 2,
    ):
        self.model = model
        self.device = device
        self.eos_token_id = eos_token_id
        self._model_runner: ModelRunner | None = None

    def init_device(self):
        self.model.eval()
        self._model_runner = ModelRunner(
            self.model, self.device, eos_token_id=self.eos_token_id
        )

    def get_model_runner(self) -> ModelRunner:
        if self._model_runner is None:
            self.init_device()
        return self._model_runner

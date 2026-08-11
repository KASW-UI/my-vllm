"""DefaultModelState — loads HuggingFace models.

Mirrors vLLM V1 worker/gpu/model_states/default.py.

C021: default implementation using runtime/model_loader + tokenizer.
"""

import torch
import torch.nn as nn

from my_vllm.model_executor.model_loader import load_model
from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.v1.worker.gpu.model_states.interface import ModelState


class DefaultModelState(ModelState):
    """Default model state for loading HuggingFace models."""

    def __init__(
        self,
        model_name: str,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        self.model_name = model_name
        self._dtype = dtype or torch.bfloat16
        self._model: nn.Module | None = None
        self._device: torch.device | None = device

    def load_model(self) -> nn.Module:
        if self._model is None:
            self._model = load_model(self.model_name)
        self._model.eval()
        if self._device is None:
            self._device = next(self._model.parameters()).device
        return self._model

    def get_device(self) -> torch.device:
        if self._device is None:
            self.load_model()
        return self._device

    def get_dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def tokenizer(self):
        return load_tokenizer(self.model_name)

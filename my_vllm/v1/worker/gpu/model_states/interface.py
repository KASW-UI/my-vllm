"""ModelState interface — mirrors vLLM V1 worker/gpu/model_states/.

C021: abstract interface for model lifecycle management.
Separates model loading and device initialization from Worker.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelState(ABC):
    """Abstract interface for model initialization state."""

    @abstractmethod
    def load_model(self) -> nn.Module:
        """Load the model and return it."""
        ...

    @abstractmethod
    def get_device(self) -> torch.device:
        """Return the target device."""
        ...

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Return the model dtype."""
        ...

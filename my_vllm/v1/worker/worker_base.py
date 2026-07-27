"""WorkerBase — abstract worker interface with execute_model().

Mirrors vLLM V1 v1/worker/worker_base.py:39.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput
as the primary execution interface.  Legacy methods retained
for backward compatibility.
"""

from abc import ABC, abstractmethod

import torch

from my_vllm.v1.outputs import ModelRunnerOutput


class WorkerBase(ABC):
    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rank = rank
        self.world_size = world_size

    @abstractmethod
    def init_device(self): ...

    @abstractmethod
    def execute_model(
        self, scheduler_output
    ) -> ModelRunnerOutput | None:
        """Execute a batch of requests through the model.

        Args:
            scheduler_output: structured scheduling output with
                new and cached request metadata.

        Returns:
            ModelRunnerOutput with sampled token IDs per request.
        """
        ...

    @abstractmethod
    def get_step(self, request_id: str) -> int: ...

    @abstractmethod
    def cleanup(self, request_id: str): ...

    # Legacy bridge — still referenced by old code.
    # Will be removed after C018 when full execute_model replaces them.

    def execute_prefill(
        self, request_id: str,
        input_ids: torch.Tensor, cache
    ) -> int:
        raise NotImplementedError

    def execute_decode(self, request_id: str, cache) -> int:
        raise NotImplementedError

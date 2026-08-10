"""WorkerBase — worker interface with execute_model().

Mirrors vLLM V1 v1/worker/worker_base.py:45.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput
as the primary execution interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from my_vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


class WorkerBase:
    """Worker interface for different hardware.

    Mirrors vLLM V1 WorkerBase — NOT an ABC. Uses raise NotImplementedError
    pattern for methods that subclasses must override.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self.device: torch.device | None = None
        self.model_runner = None

    def init_device(self):
        raise NotImplementedError

    def execute_model(
        self, scheduler_output
    ) -> ModelRunnerOutput | None:
        raise NotImplementedError

    def get_step(self, request_id: str) -> int:
        raise NotImplementedError

    def cleanup(self, request_id: str):
        raise NotImplementedError

    def execute_prefill(
        self, request_id: str,
        input_ids: torch.Tensor, cache
    ) -> int:
        raise NotImplementedError

    def execute_decode(self, request_id: str, cache) -> int:
        raise NotImplementedError
"""GPU Worker — mirrors vLLM V1 v1/worker/gpu_worker.py.

C017: adds execute_model(SchedulerOutput) → ModelRunnerOutput.
C021: accepts ModelState for model lifecycle.
C032: Worker loads model internally via GPUModelRunner.load_model()
— model is no longer passed in from run.py.  Matches vLLM lifecycle.
C150: `init_device()` now calls
`init_distributed_environment(...)` + `initialize_model_parallel(...)`
before `load_model()`. This is the canonical vLLM sequence so
`ColumnParallelLinear` / `RowParallelLinear` read the correct
`tp_rank` / `tp_size` at construction time (when the model is
constructed inside `load_model`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.worker_base import WorkerBase
from my_vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


# Compatibility export for legacy tests
_DirectModelState = None  # Removed; use DefaultModelState instead


class Worker(WorkerBase):
    """GPU Worker owning ModelRunner — mirrors vLLM V1 Worker.

    Receives VllmConfig plus distributed params, matching real vLLM's
    constructor pattern.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self._model_runner: GPUModelRunner | None = None

    def init_device(self):
        """Mirrors vLLM Worker.init_device(): init dist → init TP → load model.

        C150: the canonical vLLM sequence — `init_distributed_environment`
        must complete (NCCL rendezvous via `MASTER_ADDR`/`MASTER_PORT`)
        before `initialize_model_parallel` creates the TP group. Both
        must complete before `load_model()`, because the model's
        `ColumnParallelLinear` / `RowParallelLinear` layers size their
        weight partitions from `get_tp_world_size()` /
        `get_tp_rank()` at construction time.
        """
        # Step 1: NCCL rendezvous (env:// reads MASTER_ADDR / MASTER_PORT).
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            local_rank=self.local_rank,
            backend="nccl",
        )
        # Step 2: TP group construction (always runs even at TP=1,
        # so the `_TP` singleton is non-None and `get_tp_world_size()`
        # returns the correct size).
        initialize_model_parallel(
            tensor_model_parallel_size=self.parallel_config.tensor_parallel_size,
            backend="nccl",
        )
        self.device = self._resolve_device(self.local_rank)
        self._model_runner = GPUModelRunner(
            vllm_config=self.vllm_config,
            device=self.device,
        )
        self._model_runner.load_model()

    def _resolve_device(self, local_rank: int) -> torch.device:
        import torch
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")

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

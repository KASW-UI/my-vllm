"""MultiprocExecutor — multiprocess GPU executor.

Mirrors vLLM V1 v1/executor/multiproc_executor.py (education version).

C080: spawns one Worker process per GPU rank, dispatches SchedulerOutput
via pickle, collects ModelRunnerOutput.
C150: real NCCL rendezvous — opens a TCP port, exports
`MASTER_ADDR` / `MASTER_PORT` / `WORLD_SIZE` / `RANK` /
`LOCAL_RANK` env vars before spawning workers. Each worker's
`Worker.init_device()` calls `init_distributed_environment(...)
→ initialize_model_parallel(...)` (canonical vLLM sequence)
before loading the model.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import socket
from queue import Empty
from typing import TYPE_CHECKING

import torch

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu_worker import Worker

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


def _get_open_port() -> int:
    """Bind to port 0, read the assigned port, release.

    Mirrors vLLM's `vllm.utils.network_utils.get_open_port` —
    atomically grabs a free port from the kernel so concurrent
    executor instances don't collide.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MultiprocExecutor(Executor):
    """Multiprocess executor — one Worker process per GPU.

    Aligns with vLLM MultiprocExecutor:
    - spawns Worker processes with rank/local_rank
    - each worker calls init_device() → init_distributed_environment
      → initialize_model_parallel → load_model
    - execute_model() dispatches to all workers, returns output from rank 0
    """

    supports_pp: bool = True

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.world_size = vllm_config.parallel_config.world_size
        self.tensor_parallel_size = (
            vllm_config.parallel_config.tensor_parallel_size
        )

        self._workers: list[mp.Process] = []
        self._input_queues: list[mp.Queue] = []
        self._output_queues: list[mp.Queue] = []
        self._master_port: int | None = None

    def _init_executor(self) -> None:
        """Spawn worker processes — aligns with vLLM _init_executor.

        C150: allocates a free TCP port and exports NCCL rendezvous
        env vars (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`,
        `RANK`, `LOCAL_RANK`). Spawned workers read these via
        `env://` to join the same NCCL process group.
        """
        assert torch.cuda.is_available(), "MultiprocExecutor requires CUDA"

        self._master_port = _get_open_port()
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        for rank in range(self.world_size):
            input_q: mp.Queue = mp.Queue()
            output_q: mp.Queue = mp.Queue()

            # Per-rank env vars so workers don't inherit stale
            # values from a prior executor in the same process.
            os.environ["RANK"] = str(rank)
            os.environ["LOCAL_RANK"] = str(rank)

            proc = mp.Process(
                target=_worker_main,
                args=(rank, self.world_size, self.vllm_config,
                      input_q, output_q),
                daemon=True,
            )
            proc.start()
            self._workers.append(proc)
            self._input_queues.append(input_q)
            self._output_queues.append(output_q)

    def execute_model(self, scheduler_output) -> ModelRunnerOutput | None:
        """Dispatch to all workers, collect output from rank 0."""
        if not self._workers:
            return None

        for q in self._input_queues:
            q.put(scheduler_output)

        outputs: list[ModelRunnerOutput] = []
        for q in self._output_queues:
            outputs.append(q.get(timeout=60))

        return outputs[0]

    def get_worker(self):
        raise NotImplementedError(
            "MultiprocExecutor: use execute_model() directly"
        )

    def shutdown(self) -> None:
        for q in self._input_queues:
            q.put(None)  # sentinel
        for p in self._workers:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._workers.clear()
        self._input_queues.clear()
        self._output_queues.clear()


def _worker_main(
    rank: int,
    world_size: int,
    vllm_config: VllmConfig,
    input_q: mp.Queue,
    output_q: mp.Queue,
) -> None:
    """Child process entry point.

    Initializes CUDA device, creates Worker, loads model,
    then loops: receive SchedulerOutput → execute_model → send output.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.cuda.set_device(f"cuda:{rank}")

    worker = Worker(
        vllm_config=vllm_config,
        local_rank=rank,
        rank=rank,
        distributed_init_method="env://",
        is_driver_worker=(rank == 0),
    )
    worker.init_device()

    while True:
        try:
            scheduler_output = input_q.get(timeout=1)
        except Empty:
            continue
        if scheduler_output is None:  # shutdown sentinel
            break
        result = worker.execute_model(scheduler_output)
        output_q.put(result)

"""MultiprocExecutor — multiprocess GPU executor.

Mirrors vLLM V1 v1/executor/multiproc_executor.py (education version).

C080: spawns one Worker process per GPU rank, dispatches SchedulerOutput
via pickle, collects ModelRunnerOutput.
"""

import multiprocessing as mp
import signal
from queue import Empty

import torch

from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.worker.gpu_worker import Worker


class MultiprocExecutor(Executor):
    """Multiprocess executor — one Worker process per GPU.

    Aligns with vLLM MultiprocExecutor:
    - spawns Worker processes with rank/local_rank
    - each worker calls init_device() → init_distributed → load_model
    - execute_model() dispatches to all workers, returns output from rank 0
    """

    supports_pp: bool = True

    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        dtype_str: str = "bfloat16",
        eos_token_id: int = 2,
        block_size: int = 16,
        max_num_blocks: int = 256,
        max_num_reqs: int = 64,
        world_size: int = 1,
    ):
        super().__init__()
        self.model_name = model_name
        self.dtype_str = dtype_str
        self.eos_token_id = eos_token_id
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.max_num_reqs = max_num_reqs
        self.world_size = world_size

        self._workers: list[mp.Process] = []
        self._input_queues: list[mp.Queue] = []
        self._output_queues: list[mp.Queue] = []

    def _init_executor(self) -> None:
        """Spawn worker processes — aligns with vLLM _init_executor."""
        assert torch.cuda.is_available(), "MultiprocExecutor requires CUDA"

        for rank in range(self.world_size):
            device = f"cuda:{rank}"
            input_q: mp.Queue = mp.Queue()
            output_q: mp.Queue = mp.Queue()

            proc = mp.Process(
                target=_worker_main,
                args=(rank, self.world_size, device,
                      self.model_name, self.dtype_str, self.eos_token_id,
                      self.block_size, self.max_num_blocks, self.max_num_reqs,
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
    device: str,
    model_name: str,
    dtype_str: str,
    eos_token_id: int,
    block_size: int,
    max_num_blocks: int,
    max_num_reqs: int,
    input_q: mp.Queue,
    output_q: mp.Queue,
) -> None:
    """Child process entry point.

    Initializes CUDA device, creates Worker, loads model,
    then loops: receive SchedulerOutput → execute_model → send output.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.cuda.set_device(device)

    worker = Worker(
        model_name=model_name,
        dtype_str=dtype_str,
        eos_token_id=eos_token_id,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        max_num_reqs=max_num_reqs,
        rank=rank,
        world_size=world_size,
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

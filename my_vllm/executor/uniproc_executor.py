from abc import ABC, abstractmethod

from my_vllm.worker.worker_base import WorkerBase


class Executor(ABC):
    """Abstract executor — manages one or more Workers (C012).

    For multi-card (C013+): distributes work across workers.
    For single-card: UniProcExecutor holds one Worker.
    """

    @abstractmethod
    def get_worker(self) -> WorkerBase:
        ...


class UniProcExecutor(Executor):
    """Single-process executor with one Worker."""

    def __init__(self, worker: WorkerBase):
        self.worker = worker

    def get_worker(self) -> WorkerBase:
        return self.worker

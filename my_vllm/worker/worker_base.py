from abc import ABC, abstractmethod

from my_vllm.worker.model_runner import ModelRunner


class WorkerBase(ABC):
    """Abstract worker interface (C011).

    Owns the model and ModelRunner.  External code should only
    interact with the worker through this interface.
    """

    @abstractmethod
    def init_device(self):
        ...

    @abstractmethod
    def get_model_runner(self) -> ModelRunner:
        ...

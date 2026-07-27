from abc import ABC, abstractmethod

import torch

from my_vllm.cache.kv_cache import KVCache


class WorkerBase(ABC):
    """Abstract worker interface (C012).

    External code interacts with the worker only through
    execute_prefill / execute_decode — never through
    ModelRunner directly.
    """

    @abstractmethod
    def init_device(self):
        ...

    @abstractmethod
    def execute_prefill(
        self, request_id: str,
        input_ids: torch.Tensor, cache: KVCache
    ) -> int:
        ...

    @abstractmethod
    def execute_decode(
        self, request_id: str, cache: KVCache
    ) -> int:
        ...

    @abstractmethod
    def get_step(self, request_id: str) -> int:
        ...

    @abstractmethod
    def cleanup(self, request_id: str):
        ...

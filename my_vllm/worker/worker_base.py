from abc import ABC, abstractmethod

import torch

from my_vllm.cache.kv_cache import KVCache


class WorkerBase(ABC):
    """Abstract worker interface (C013).

    C013: adds rank / world_size for Tensor Parallel support.
    When world_size=1, the worker runs in single-card mode.
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.rank = rank
        self.world_size = world_size

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

"""Base class for attention-like layers — aligns with vLLM.

Architecture preserved: AttentionLayerBase (ABC), bind_kv_cache,
get_attn_backend, get_kv_cache_spec.
"""

from abc import ABC, abstractmethod

import torch


class AttentionLayerBase(ABC):
    """Base class for attention-like layers — aligns with vLLM."""

    impl: object
    supports_dcp: bool = True

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self):
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config):
        pass

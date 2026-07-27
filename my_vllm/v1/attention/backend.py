"""Attention backend interface — mirrors vLLM V1 v1/attention/backend.py.

C022: abstract interfaces (AttentionBackend, AttentionImpl) for
pluggable attention computation.  TorchAttentionBackend is the
default fallback using HF's native attention via model.forward().
"""

from abc import ABC, abstractmethod

import torch

from my_vllm.v1.attention.metadata import CommonAttentionMetadata


class AttentionBackend(ABC):
    """Abstract factory for attention implementations.

    Mirrors vLLM V1 AttentionBackend (simplified).
    """

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        raise NotImplementedError


class AttentionImpl(ABC):
    """Per-layer attention implementation.

    Mirrors vLLM V1 AttentionImpl forward signature.
    """

    @abstractmethod
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
    ): ...

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention: Q @ K^T / sqrt(d_k) * V.

        Args:
            query: [num_tokens, num_heads, head_size]
            key:   [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            kv_cache: physical KV cache tensor
            attn_metadata: per-batch metadata (block tables, slots, etc.)
            output: pre-allocated output buffer [num_tokens, num_heads, head_size]

        Returns:
            Attention output tensor.
        """
        raise NotImplementedError

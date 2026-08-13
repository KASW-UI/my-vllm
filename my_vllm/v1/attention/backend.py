"""Attention backend interface — mirrors vLLM V1 v1/attention/backend.py.

C022: abstract interfaces (AttentionBackend, AttentionImpl) for
pluggable attention computation.

C117 / C120: the default backend is now TritonAttentionBackend (real
vLLM Triton kernels). FlashAttentionBackend is kept as a fallback for
environments without Triton.

C131: CommonAttentionMetadata merged from the former metadata.py into
this file, matching vLLM's single-file layout
(vllm/v1/attention/backend.py:412).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """Per-batch metadata shared across all attention layers.

    Mirrors vLLM V1 v1/attention/backend.py:412.

    Fields assembled from:
      C017 InputBatch  → query_start_loc, seq_lens, positions
      C019 BlockTables → block_table_tensor, slot_mapping
      C020 RequestState → num_reqs, num_actual_tokens
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,) start location of each request in the query."""
    query_start_loc_cpu: torch.Tensor
    """CPU copy of query_start_loc."""

    seq_lens: torch.Tensor
    """(batch_size,) number of computed tokens per request."""
    seq_lens_cpu: torch.Tensor
    """CPU copy of seq_lens."""

    num_reqs: int

    num_actual_tokens: int
    """Total number of tokens in the batch."""

    max_query_len: int
    """Longest query in the batch."""

    max_seq_len: int
    """Longest context length."""

    block_table_tensor: torch.Tensor
    """[num_reqs, max_blocks] int32 — paged block table."""

    slot_mapping: torch.Tensor
    """[num_tokens] int64 — slot_id = block_id * block_size + offset."""

    positions: torch.Tensor | None = None
    """(num_actual_tokens,) token positions."""

    causal: bool = True


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

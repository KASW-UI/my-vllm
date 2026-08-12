"""TritonAttentionBackend — Triton-based paged attention backend.

Mirrors vllm/v1/attention/backends/triton_attn.py.

Uses real vLLM Triton kernels copied from the reference codebase:
  - triton_reshape_and_cache_flash  (KV cache write)
  - unified_attention                (paged attention compute)
"""

import torch

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from my_vllm.v1.attention.ops.triton_unified_attention import unified_attention


class TritonAttentionBackend(AttentionBackend):
    """Triton-based paged attention backend."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        # Flash layout matching triton_reshape_and_cache_flash kernel:
        # [num_blocks, block_size, num_kv_heads, head_size]
        return (num_blocks, block_size, num_kv_heads, head_size)


class TritonAttentionImpl(AttentionImpl):
    """Triton paged attention forward pass.

    Mirrors vllm/v1/attention/backends/triton_attn.py TritonAttentionImpl.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = None
        self.use_alibi_sqrt = False
        self.sliding_window = None
        self.logits_soft_cap = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute paged attention with real vLLM Triton kernels.

        Steps:
          1. Write new K/V into paged KV cache
          2. Compute paged attention via unified_attention kernel
        """
        num_tokens = query.shape[0]
        if num_tokens == 0:
            return output

        slot_mapping = attn_metadata.slot_mapping
        block_table = attn_metadata.block_table_tensor
        seq_lens = attn_metadata.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len

        kv_cache_dtype = "auto"
        k_scale = torch.tensor([1.0], device=query.device, dtype=torch.float32)
        v_scale = torch.tensor([1.0], device=query.device, dtype=torch.float32)

        # Step 1: Write K/V into paged cache
        triton_reshape_and_cache_flash(
            key=key.contiguous(),
            value=value.contiguous(),
            key_cache=kv_cache,
            value_cache=kv_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        # Step 2: Unified paged attention
        num_actual_tokens = attn_metadata.num_actual_tokens

        cu_seqlens_q = query_start_loc[:num_actual_tokens + 1]
        max_seqlen_q = max_query_len
        seqused_k = seq_lens
        max_seqlen_k = max_seq_len

        unified_attention(
            q=query[:num_actual_tokens].contiguous(),
            k=kv_cache,
            v=kv_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
        )

        return output

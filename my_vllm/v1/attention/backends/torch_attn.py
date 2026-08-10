"""TorchAttention — fallback backend using PyTorch SDPA.

Mirrors vLLM V1's torch-native attention path.

Uses paged KV cache + block_table for reading history KV,
matching real vLLM's interface pattern:
  attention(query, new_k, new_v, kv_cache, block_table, seq_lens)

Educational limitation:
  Still materializes contiguous K/V from paged cache via
  gather_cached_kv_pytorch() inside forward(), rather than
  reading directly inside the attention kernel.
  Real vLLM passes block_table to FlashAttention kernel
  which indexes paged cache natively without gathering.
"""

import torch
import torch.nn.functional as F

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)


class TorchAttentionBackend(AttentionBackend):
    """Torch SDPA backend — fallback when no hardware backend is available."""

    @staticmethod
    def get_name() -> str:
        return "torch_attn"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return TorchAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)


class TorchAttentionImpl(AttentionImpl):
    """Attention using paged KV cache + block_table.

    Mirrors vLLM FlashAttentionImpl.forward() interface:
    receives raw kv_cache (paged) + block_table, reads history
    K/V via block_table lookup, concatenates with new K/V,
    runs SDPA attention.
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Paged-attention forward with block_table lookup.

        query:  [num_tokens, num_heads, head_size]
        key:    [num_tokens, num_kv_heads, head_size] — this step's NEW K
        value:  [num_tokens, num_kv_heads, head_size] — this step's NEW V
        kv_cache: (key_cache, value_cache) paged tensors
        attn_metadata: block_table, seq_lens, slot_mapping, etc.
        output: pre-allocated [num_tokens, num_heads * head_size]

        Educational limitation: still materializes contiguous K/V
        via gather_cached_kv_pytorch(). Real vLLM passes block_table
        to FlashAttention kernel which indexes paged cache natively.
        """
        block_table = attn_metadata.block_table_tensor
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(ratio, dim=1)
            value = value.repeat_interleave(ratio, dim=1)

        is_decode = max_query_len == 1
        num_reqs = block_table.shape[0]

        if is_decode and seq_lens.numel() > 0:
            k_cache, v_cache = _unpack_kv_cache(kv_cache, self.head_size)
            history_k, history_v = gather_cached_kv_pytorch(
                k_cache, v_cache,
                block_table, seq_lens, kv_cache.shape[2],
            )
            history_k = history_k.repeat_interleave(ratio, dim=1)
            history_v = history_v.repeat_interleave(ratio, dim=1)
            full_k = torch.cat([
                history_k.permute(2, 0, 1, 3).reshape(-1, self.num_heads, self.head_size),
                key,
            ], dim=0)
            full_v = torch.cat([
                history_v.permute(2, 0, 1, 3).reshape(-1, self.num_heads_kv, self.head_size),
                value,
            ], dim=0)
            q = query.transpose(0, 1).unsqueeze(0)
            k = full_k.transpose(0, 1).unsqueeze(0)
            v = full_v.transpose(0, 1).unsqueeze(0)
        else:
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=attn_metadata.causal,
            scale=self.scale,
        )
        result = out.squeeze(0).transpose(0, 1)
        output.copy_(result)
        return output


def _unpack_kv_cache(
    kv_cache: torch.Tensor, head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split packed K+V cache into key_cache and value_cache.

    kv_cache: [num_blocks, num_kv_heads, block_size, 2*head_size] for packed
    or just the key_cache tensor. For educational project, kv_cache is
    passed as a (key_cache, value_cache) tuple handled externally.
    """
    if isinstance(kv_cache, tuple):
        return kv_cache[0], kv_cache[1]
    last_dim = kv_cache.shape[-1]
    if last_dim == head_size:
        return kv_cache, None
    k = kv_cache[..., :head_size]
    v = kv_cache[..., head_size:]
    return k, v
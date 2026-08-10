"""FlashAttention backend — mirrors vLLM V1 v1/attention/backends/flash_attn.py.

Educational note: uses PyTorch SDPA as the attention kernel rather than
the actual CUDA FlashAttention library.  The interface pattern (paged
KV cache + block_table + slot_mapping) matches real vLLM's FlashAttention.
"""

import torch
import torch.nn.functional as F

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)


class FlashAttentionImpl(AttentionImpl):

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
        block_table = attn_metadata.block_table_tensor
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(ratio, dim=1)
            value = value.repeat_interleave(ratio, dim=1)

        is_decode = max_query_len == 1

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
                history_v.permute(2, 0, 1, 3).reshape(-1, self.num_kv_heads, self.head_size),
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
    if isinstance(kv_cache, tuple):
        return kv_cache[0], kv_cache[1]
    last_dim = kv_cache.shape[-1]
    if last_dim == head_size:
        return kv_cache, None
    k = kv_cache[..., :head_size]
    v = kv_cache[..., head_size:]
    return k, v

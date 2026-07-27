"""TorchAttention — fallback backend using PyTorch SDPA.

Mirrors vLLM V1's torch-native attention path.

C022: uses torch.nn.functional.scaled_dot_product_attention
as the default backend.  No block table/paged KV support yet
(C023 adds PagedAttention).
"""

import torch

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata


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
    """Attention using torch.nn.functional.scaled_dot_product_attention."""

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
        """Standard attention: Q @ K^T → softmax → @ V."""
        # Reshape for SDPA if needed
        # query:  [num_tokens, num_heads, head_size]
        # key:    [num_tokens, num_kv_heads, head_size]
        # value:  [num_tokens, num_kv_heads, head_size]

        # For grouped-query attention, expand K/V to match Q heads
        if self.num_kv_heads != self.num_heads:
            ratio = self.num_heads // self.num_kv_heads
            key = key.repeat_interleave(ratio, dim=1)
            value = value.repeat_interleave(ratio, dim=1)

        # Transpose to [batch, heads, seq, head_dim] for SDPA
        q = query.transpose(0, 1).unsqueeze(0)     # [1, num_heads, tokens, dim]
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        is_causal = attn_metadata.causal

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        # [1, num_heads, tokens, dim] → [tokens, num_heads, dim]
        result = out.squeeze(0).transpose(0, 1)
        output.copy_(result)
        return output

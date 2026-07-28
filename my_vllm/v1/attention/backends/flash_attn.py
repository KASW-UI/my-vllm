"""FlashAttention backend — stub for future CUDA extension.

Mirrors vLLM V1 v1/attention/backends/flash_attn.py.

C031: placeholder for FlashAttention hardware backend.
For now, TorchAttentionBackend (torch SDPA) is the default.
"""

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata


class FlashAttentionBackend(AttentionBackend):
    """FlashAttention backend (stub — requires CUDA FlashAttention extension)."""

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
    """FlashAttention implementation (stub)."""

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
        query,
        key,
        value,
        kv_cache,
        attn_metadata: CommonAttentionMetadata,
        output,
    ):
        raise NotImplementedError(
            "FlashAttention requires CUDA FlashAttention extension. "
            "Use TorchAttentionBackend for PyTorch SDPA fallback."
        )

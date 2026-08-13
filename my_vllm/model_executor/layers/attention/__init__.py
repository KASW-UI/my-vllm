"""Attention layer — aligns with vLLM vllm/model_executor/layers/attention/attention.py.

Architecture preserved: Attention (nn.Module + AttentionLayerBase), constructor signature,
forward method reading attn_metadata from ForwardContext, impl-based attention dispatch.

Engineering details skipped: CUDA graph/compilation, unified_attention ops,
KV cache quantization (FP8), sliding window, alibi, custom op registration, query quant.
"""

import torch
import torch.nn as nn

from my_vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer — aligns with vLLM Attention.

    Takes query, key, value tensors as input. Stores KV cache for
    paged attention support. Dispatches to the attention backend impl.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: None = None,
        quant_config: None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = "DECODER",
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.alibi_slopes = alibi_slopes
        self.logits_soft_cap = logits_soft_cap
        self.sliding_window = per_layer_sliding_window
        self.layer_name = prefix
        self.attn_type = attn_type
        self.kv_cache = torch.tensor([])
        self.kv_cache_dtype = "auto"
        self.kv_cache_torch_dtype = torch.get_default_dtype()
        self.quant_config = quant_config
        self.calculate_kv_scales = False
        self.dtype = torch.get_default_dtype()
        self.use_direct_call = True

        self.attn_backend = self.get_attn_backend()
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
        )  # type: ignore[assignment]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        from my_vllm.forward_context import get_forward_context

        query_shape = query.shape
        num_tokens = query.numel() // (self.num_heads * self.head_size)
        if output_shape is None:
            output_shape = query_shape[:-1] + (self.num_heads * self.head_size,)

        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size)

        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata.get(
            self.layer_name, next(iter(ctx.attn_metadata.values()))
        )

        output = torch.empty(
            num_tokens, self.num_heads, self.head_size,
            dtype=query.dtype, device=query.device,
        )

        return self.impl.forward(
            query, key, value, self.kv_cache, attn_metadata, output,
        ).reshape(output_shape)

    def get_attn_backend(self):
        from my_vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config):
        from my_vllm.v1.kv_cache_interface import FullAttentionSpec
        block_size = vllm_config.cache_config.block_size if vllm_config else 16
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        self.kv_cache = kv_cache

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}"
        s += f", num_heads={self.num_heads}"
        s += f", num_kv_heads={self.num_kv_heads}"
        s += f", scale={self.scale}"
        return s

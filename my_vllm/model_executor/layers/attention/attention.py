"""Attention layer — aligns with vLLM vllm/model_executor/layers/attention/attention.py.

The Attention nn.Module sits between the QKV projection and the output
projection in each transformer block. It owns:
  - the AttentionImpl instance (provided by the AttentionBackend)
  - the bound KV cache tensor (set by the runner via bind_kv_cache)
  - the layer's kv_cache_spec (consumed by the engine KV cache allocator)

The forward pass reads `attn_metadata` and `slot_mapping` from the
global ForwardContext (set by GPUModelRunner via set_forward_context),
reshapes q/k/v into the impl's expected layout, and dispatches to
self.impl.forward.

Engineering details skipped (principled scope reduction):
- Cross-attention / encoder-only attention / MLA variants (single path only)
- CUDA graph / torch.compile registration (static_forward_context)
- KV cache dtype negotiation (FP8 quant deferred to P1 quantization path)
- Soft cap, ALiBi, sliding window, custom op registration
- Query/key/value quantization paths (fused_input_quant, etc.)
"""

import torch
import torch.nn as nn

from my_vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer — aligns with vLLM Attention.

    Constructor signature mirrors vllm Attention (subset):
        Attention(num_heads, head_size, scale, num_kv_heads=None,
                  alibi_slopes=None, cache_config=None, quant_config=None,
                  logits_soft_cap=None, per_layer_sliding_window=None,
                  prefix="", attn_type="DECODER", **extra_impl_args)

    Args:
        num_heads: total number of query heads (Q).
        head_size: per-head feature dimension.
        scale: softmax scale = 1/sqrt(head_size).
        num_kv_heads: total number of key/value heads. None → MHA (== num_heads).
        alibi_slopes: ALiBi slope per head (None for OPT).
        cache_config: VllmConfig.cache_config (None → use default block_size).
        quant_config: quantization config (None → no quantization).
        logits_soft_cap: soft cap for attention logits (None for OPT).
        per_layer_sliding_window: sliding-window size (None for OPT).
        prefix: fully-qualified layer name (used as `layer_name` and for
                static_forward_context registration in real vLLM).
        attn_type: "DECODER" | "ENCODER" | ... (only "DECODER" supported here).
        **extra_impl_args: passed to impl_cls.__init__.
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
        nn.Module.__init__(self)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

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
        self.attn_backend = self.get_attn_backend()
        impl_cls = self.attn_backend.get_impl_cls()

        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            **extra_impl_args,
        )
        self.kv_cache: torch.Tensor | tuple[torch.Tensor, ...] = torch.tensor([])
        self.kv_cache_dtype = "auto"
        self.kv_cache_torch_dtype = torch.get_default_dtype()
        self.quant_config = quant_config
        self.cache_config = cache_config
        self.calculate_kv_scales = False
        self.dtype = torch.get_default_dtype()
        self.use_direct_call = True

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Compute attention for one transformer block.

        Reads `attn_metadata[layer_name]` and `slot_mapping[layer_name]`
        from the active ForwardContext; the runner installs that context
        once per forward pass via my_vllm.forward_context.set_forward_context.

        Args:
            query: [num_tokens, num_heads * head_size] (flattened) or
                   [num_tokens, num_heads, head_size].
            key:   same as query, for KV.
            value: same as query, for KV.
            output_shape: optional reshape target for the impl output.
            output_dtype: optional cast target for the impl output.

        Returns:
            Attention output, shape == output_shape or query.shape[:-1] +
            (num_heads * head_size,).
        """
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
        if self.layer_name in ctx.attn_metadata:
            attn_metadata = ctx.attn_metadata[self.layer_name]
        else:
            attn_metadata = next(iter(ctx.attn_metadata.values()))

        output = torch.empty(
            num_tokens, self.num_heads, self.head_size,
            dtype=query.dtype, device=query.device,
        )

        out = self.impl.forward(
            query, key, value, self.kv_cache, attn_metadata, output,
        )
        if output_dtype is not None and out.dtype != output_dtype:
            out = out.to(output_dtype)
        return out.reshape(output_shape)

    def get_attn_backend(self) -> "type[object]":
        """Return the AttentionBackend class for this layer.

        Mirrors vLLM Attention.get_attn_backend(). Resolves via the
        v1/attention/selector.get_attn_backend() hook so the backend is
        configurable in one place.
        """
        from my_vllm.v1.attention.selector import get_attn_backend
        return get_attn_backend()

    def get_kv_cache_spec(self, vllm_config):
        """Return the FullAttentionSpec for this layer.

        Mirrors vLLM Attention.get_kv_cache_spec (DECODER subset).
        """
        from my_vllm.v1.kv_cache_interface import FullAttentionSpec
        block_size = (
            vllm_config.cache_config.block_size
            if vllm_config is not None and vllm_config.cache_config is not None
            else 16
        )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )

    def bind_kv_cache(
        self, kv_cache: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> None:
        """Bind the allocated KV cache tensor to this layer."""
        self.kv_cache = kv_cache

    def extra_repr(self) -> str:
        return (
            f"head_size={self.head_size}"
            f", num_heads={self.num_heads}"
            f", num_kv_heads={self.num_kv_heads}"
            f", scale={self.scale}"
        )
"""Base class for attention-like layers — aligns with vLLM.

Mirrors vllm/vllm/model_executor/layers/attention_layer_base.py.

Architecture preserved: AttentionLayerBase ABC with bind_kv_cache,
get_attn_backend, get_kv_cache_spec. Type-annotated against the
real vLLM AttentionImpl, AttentionBackend, KVCacheSpec interfaces.

Engineering details skipped (principled scope reduction):
- KV cache dtype negotiation (FP8 quant deferred to P1 quantization path)
- attn_type enum (only DECODER supported in my-vllm basic runtime)
- DCP (Disaggregated Cache Prefill) flag — kept as default True for
  forward-compat with vLLM but not exercised.
"""

from abc import ABC, abstractmethod

import torch


class AttentionLayerBase(ABC):
    """Base class for attention-like layers (Attention, Mamba, etc.)

    Subclasses must populate self.impl with an AttentionImpl instance
    (typically inside __init__, via the backend class returned by
    get_attn_backend()).
    """

    impl: "object"
    """The bound AttentionImpl instance (set in subclass __init__)."""

    supports_dcp: bool = True
    """Whether the layer supports Disaggregated Cache Prefill."""

    def bind_kv_cache(self, kv_cache: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        """Bind the allocated KV cache tensor to this layer.

        Default stores the cache view as-is. Subclasses (e.g. Mamba)
        override to unpack the raw buffer into per-state views.
        """
        self.kv_cache = kv_cache

    @abstractmethod
    def get_attn_backend(self) -> "type[object]":
        """Return the AttentionBackend class for this layer.

        Subclasses return a concrete backend (e.g. FlashAttentionBackend,
        TritonAttentionBackend).
        """
        pass

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: "object"):
        """Return the KVCacheSpec for this layer (or None if no KV cache).

        Subclasses return a concrete spec (e.g. FullAttentionSpec).
        """
        pass
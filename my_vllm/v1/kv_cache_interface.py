"""KV Cache interface — architecture boundary between Attention and KVCacheManager.

Mirrors vLLM V1 v1/kv_cache_interface.py.

C018: KVCacheSpec base class establishing the interface contract.
C033: Full hierarchy — AttentionSpec, FullAttentionSpec, SlidingWindowSpec,
      MLAAttentionSpec, KVCacheGroupSpec.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheSpec:
    """Base specification for a single KV cache layer group.

    Mirrors vLLM V1 kv_cache_interface.py KVCacheSpec.
    """

    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """Total bytes per block for this spec (K + V)."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    """Specification for an attention-based KV cache layer group.

    Mirrors vLLM V1 kv_cache_interface.py AttentionSpec.
    """

    num_kv_heads: int
    head_size: int
    dtype: str = "float16"

    @property
    def page_size_bytes(self) -> int:
        element_size = 2
        return (
            2
            * self.num_kv_heads
            * self.block_size
            * self.head_size
            * element_size
        )


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """Full attention KV cache spec — all tokens cached, full attention window.

    When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding window
    attention layers are regarded as full attention in the KV cache manager
    (blocks are allocated for all tokens), while computed as sliding window
    attention in the model runner.

    Mirrors vLLM V1 kv_cache_interface.py FullAttentionSpec.
    """

    head_size_v: int | None = None
    sliding_window: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def page_size_bytes(self) -> int:
        element_size = 2
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * element_size
        )


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    """Sliding window attention KV cache spec — only recent tokens retained.

    Mirrors vLLM V1 kv_cache_interface.py SlidingWindowSpec.
    """

    sliding_window: int
    head_size_v: int | None = None

    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    @property
    def page_size_bytes(self) -> int:
        element_size = 2
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * element_size
        )


@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    """Multi-head Latent Attention KV cache spec — compressed KV representation.

    Mirrors vLLM V1 kv_cache_interface.py MLAAttentionSpec.
    """

    cache_dtype_str: str | None = None


@dataclass
class KVCacheGroupSpec:
    """A group of model layers that share the same KV cache block table.

    Mirrors vLLM V1 kv_cache_interface.py KVCacheGroupSpec.
    """

    layer_names: list[str]
    kv_cache_spec: KVCacheSpec


@dataclass
class KVCacheConfig:
    """The KV cache configuration of a model.

    Mirrors vLLM V1 kv_cache_interface.py KVCacheConfig.
    C050: num_blocks + kv_cache_groups + kv_cache_tensors.
    C033: num_blocks + kv_cache_groups (minimal).
    """

    num_blocks: int
    kv_cache_groups: list[KVCacheGroupSpec]

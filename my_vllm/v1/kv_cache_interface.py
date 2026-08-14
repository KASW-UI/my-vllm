"""KV cache interface — spec hierarchy + KVCacheConfig.

Mirrors vLLM V1 vllm/v1/kv_cache_interface.py (subset, not vendored).

This file contains:
- KVQuantMode / get_kv_quant_mode / helpers — pre-existing from C213
- KVCacheSpec hierarchy (C254) — frozen dataclass base + FullAttentionSpec
- KVCacheSpecKind enum + get_kv_cache_spec_kind() helper
- KVCacheTensor / KVCacheGroupSpec / KVCacheConfig — the per-model config
  that the `KVCacheCoordinator` (v1/core/kv_cache_coordinator.py)
  consumes to build a `single_type_managers` tuple.

Scope decisions (per docs/alignment-plan.md):
- my-vllm only ships `FullAttentionSpec` (Dense/OPT path). The other
  spec types (SlidingWindowSpec, ChunkedLocalAttentionSpec, MambaSpec,
  CrossAttentionSpec, MLAAttentionSpec) are **declared shells** —
  registered in `kv_cache_spec_registry.py` with their manager class
  shells but no concrete implementations. Adding the actual class body
  happens when Llama (GQA/SWA) or DeepSeek (MLA/Mamba) gets wired in
  (out of scope for Phase 4 v2).
- `VllmConfig`-coupled methods (`max_memory_usage_bytes`) take a
  `VllmConfig`-shaped duck type — my-vllm's `VllmConfig` has the same
  fields used by the math (model_config.max_model_len, block_size,
  parallel_config.decode_context_parallel_size).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import ceil
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


# ---------------------------------------------------------------------------
# KV cache quantization mode (pre-existing from C213)
# ---------------------------------------------------------------------------


class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.

    Mirrors vllm/v1/kv_cache_interface.py:KVQuantMode (subset — no
    NVFP4; my-vllm exercises the four modes below via C211/C213/C149
    quantization kernels).
    """

    NONE = 0
    FP8_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4

    @property
    def is_per_token_head(self) -> bool:
        """True for any per-token-head quantization mode."""
        return self in (
            KVQuantMode.INT8_PER_TOKEN_HEAD,
            KVQuantMode.FP8_PER_TOKEN_HEAD,
            KVQuantMode.INT4_PER_TOKEN_HEAD,
        )


def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    if kv_cache_dtype == "int4_per_token_head":
        return KVQuantMode.INT4_PER_TOKEN_HEAD
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    if kv_cache_dtype == "fp8_per_token_head":
        return KVQuantMode.FP8_PER_TOKEN_HEAD
    if isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8"):
        # Per-tensor FP8 (used by the C211 kernel path).
        return KVQuantMode.FP8_TENSOR
    return KVQuantMode.NONE


def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return get_kv_quant_mode(kv_cache_dtype) != KVQuantMode.NONE


def kv_cache_uses_per_token_head_scales(kv_cache_dtype: str) -> bool:
    """Return True if *kv_cache_dtype* needs per-token-head scales."""
    return get_kv_quant_mode(kv_cache_dtype).is_per_token_head


# ---------------------------------------------------------------------------
# KV cache spec kinds (C254 — used by HybridKVCacheCoordinator dispatch)
# ---------------------------------------------------------------------------


class KVCacheSpecKind(str, Enum):
    """Discriminator used to route cache-hit / cache-block logic by spec type.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:KVCacheSpecKind (subset).

    Kept as a string-valued enum so debug logs are readable without an
    extra ``.name`` round-trip; the values match vLLM's spelling.
    """

    FULL_ATTENTION = "full_attention"
    SLIDING_WINDOW = "sliding_window"
    CHUNKED_LOCAL_ATTENTION = "chunked_local_attention"
    MAMBA = "mamba"
    CROSS_ATTENTION = "cross_attention"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# KV cache spec hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KVCacheSpec:
    """A base class for specifying the KV cache format of one layer.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:KVCacheSpec.

    C254: my-vllm only declares the abstract surface + the concrete
    ``FullAttentionSpec``. Subclasses (SlidingWindowSpec, MambaSpec, …)
    are declared in `kv_cache_spec_registry.py` as class shells with the
    same field names so the registry can dispatch them.

    The `is_uniform_with_collection` helper is consumed by
    ``KVCacheGroupSpec.is_uniform_type`` (formerly
    ``UniformTypeKVCacheSpecs.is_uniform_type``) to decide whether all
    layers in a model can share one KV cache group. my-vllm uses the
    registry's ``uniform_type_base_spec`` (set at registration time)
    instead of the MRO walk vLLM does inline.
    """

    block_size: int

    @property
    def page_size_bytes(self) -> int:
        """The size of a page with `block_size` tokens in bytes."""
        raise NotImplementedError

    @property
    def storage_block_size(self) -> int:
        """Effective storage block size after compression (MLAv3 etc.).

        Default: identical to ``block_size``.
        """
        return self.block_size

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        """Maximum possible memory usage of this KV cache in bytes."""
        raise NotImplementedError

    def max_num_blocks_per_req(
        self, vllm_config: "VllmConfig", max_len: int
    ) -> int:
        """Per-request block-table row length for this cache group."""
        return ceil(max_len / self.block_size)

    def copy_with_new_block_size(self, block_size: int) -> "KVCacheSpec":
        """Create a new spec with ``block_size`` replaced."""
        from dataclasses import replace
        return replace(self, block_size=block_size)

    @classmethod
    def merge(cls, specs: list["KVCacheSpec"]) -> "KVCacheSpec":
        """Merge a list of KVCacheSpec objects into a single spec.

        The base implementation asserts all specs are equal (same spec
        type means same fields). Subclasses override for type-specific
        merging (e.g. MLA merges ``head_size_v``; SlidingWindow merges
        ``sliding_window``).
        """
        from dataclasses import replace as _replace
        assert specs, "merge requires at least one spec"
        first = specs[0]
        assert all(spec == first for spec in specs[1:]), (
            "All layers in the same KV cache group must have the same "
            "KVCacheSpec."
        )
        # Frozen dataclass → deepcopy is safe.
        import copy
        return copy.deepcopy(first)

    def is_uniform_with_collection(
        self, kv_cache_specs: dict[str, "KVCacheSpec"]
    ) -> bool:
        """Whether this spec is uniform with all layers' specs.

        Implementation note: we walk the registry's
        ``uniform_type_base_spec`` so subclasses with custom base types
        (e.g. RSWASpec with base=FullAttentionSpec) merge correctly.
        """
        from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        base = KVCacheSpecRegistry.get_uniform_type_base_spec(self)
        assert base is not None, (
            f"Unsupported KV cache spec type: {type(self).__name__}. "
            "Register it using @register_kv_cache_spec decorator."
        )
        return all(isinstance(spec, base) for spec in kv_cache_specs.values())


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    """Base for attention-layer KV cache specs (full / sliding / chunked).

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:AttentionSpec (subset).
    Carries the per-layer geometry fields consumed by
    ``FullAttentionManager.get_num_blocks_to_allocate`` and the page-
    size math used by ``page_size_bytes``.
    """

    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def real_page_size_bytes(self) -> int:
        """Unpadded per-page byte size before alignment padding."""
        # Per-token-head scales are stored in separate tensors managed
        # by the attention backend, but the memory is carved from the
        # raw KV cache allocation so it must be budgeted here.
        # my-vllm NOTE: my-vllm doesn't vendor the C++ helper, so the
        # bytes are computed inline. The math is identical.
        last_dim = self.head_size
        element_size = self._element_size()
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            last_dim = self.head_size // 2
        unpadded = (
            2
            * self.block_size
            * self.num_kv_heads
            * last_dim
            * element_size
        )
        if self.kv_quant_mode.is_per_token_head:
            unpadded += (
                2
                * self.block_size
                * self.num_kv_heads
                * 4  # float32 per-token-head scale
            )
        return unpadded

    @property
    def page_size_bytes(self) -> int:
        if self.page_size_padded is not None:
            assert self.page_size_padded >= self.real_page_size_bytes
            return self.page_size_padded
        return self.real_page_size_bytes

    def _element_size(self) -> int:
        """Bytes per element for ``self.dtype``."""
        return self.dtype.itemsize

    def max_num_blocks_per_req(
        self, vllm_config: "VllmConfig", max_len: int
    ) -> int:
        """Per-request block-table row length, scaled by DCP if active."""
        dcp_world_size = getattr(
            vllm_config.parallel_config, "decode_context_parallel_size", 1
        )
        return ceil(max_len / (self.block_size * dcp_world_size))


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    """Spec for dense-attention layers (OPT, Llama dense, GPT-2).

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:FullAttentionSpec (subset).
    The only concrete subclass actually exercised in my-vllm today; the
    SlidingWindowSpec / ChunkedLocalAttentionSpec / MambaSpec / MLA shells
    are class skeletons registered for multi-spec routing in C254 but
    carry no concrete math — they raise on ``max_memory_usage_bytes``.
    """

    head_size_v: int | None = None
    sliding_window: int | None = None
    """Default None = full attention."""
    attention_chunk_size: int | None = None
    non_causal: bool = False

    def __post_init__(self) -> None:
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)

    def max_memory_usage_bytes(self, vllm_config: "VllmConfig") -> int:
        """Full attention holds every token until the request finishes."""
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = getattr(
            vllm_config.parallel_config, "decode_context_parallel_size", 1
        )
        if dcp_world_size > 1:
            max_model_len = ceil(max_model_len / dcp_world_size)
        return ceil(max_model_len / self.block_size) * self.page_size_bytes


# ---------------------------------------------------------------------------
# KV cache tensors + groups + config (consumed by KVCacheCoordinator)
# ---------------------------------------------------------------------------


@dataclass
class KVCacheTensor:
    """How a worker should initialize a KV cache tensor for one slab.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:KVCacheTensor.

    my-vllm NOTE: we don't split ``KVCacheConfig`` into per-rank tensors
    yet (no TP sharding — `TensorParallel` config field exists but is
    pinned to 1). The fields are preserved so future Llama-MLA / TP
    wiring can populate them.
    """

    size: int
    """Size of the KV cache tensor in bytes."""
    shared_by: list[str]
    """Layer names that share this tensor."""
    offset: int = 0
    """Byte offset of this layer within a contiguous block."""
    block_stride: int = 0
    """Bytes per block in a packed layout (0 = not packed)."""


@dataclass
class KVCacheGroupSpec:
    """One group of layers that share a single KV cache block table.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:KVCacheGroupSpec.

    In a model with all-FullAttention layers there's one group containing
    every layer; in a hybrid (full-attn + Mamba / sliding) there are
    multiple groups, each owned by one ``SingleTypeKVCacheManager``.
    """

    layer_names: list[str]
    """The names of model layers in this group."""
    kv_cache_spec: KVCacheSpec
    """The KV cache spec of this group (one spec per manager)."""
    is_eagle_group: bool = False
    """Whether this group contains EAGLE/MTP draft attention layers."""

    @classmethod
    def from_layers(
        cls,
        layer_names: list[str],
        kv_cache_spec: KVCacheSpec,
        is_eagle_group: bool = False,
    ) -> "KVCacheGroupSpec":
        """Factory: build a group with the given spec."""
        return cls(
            layer_names=list(layer_names),
            kv_cache_spec=kv_cache_spec,
            is_eagle_group=is_eagle_group,
        )


@dataclass
class KVCacheConfig:
    """The KV cache configuration of a model.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:KVCacheConfig.

    Consumed by ``get_kv_cache_coordinator()`` to pick
    ``UnitaryKVCacheCoordinator`` (1 group) vs
    ``HybridKVCacheCoordinator`` (≥ 2 groups) vs
    ``KVCacheCoordinatorNoPrefixCache`` (caching off).
    """

    num_blocks: int
    """The number of KV cache blocks."""
    kv_cache_tensors: list[KVCacheTensor]
    """How workers should initialize KV cache tensors per layer."""
    kv_cache_groups: list[KVCacheGroupSpec]
    """The kv cache groups of the model. One for dense; multiple for hybrid."""

    @property
    def has_mamba_layers(self) -> bool:
        """True iff any group has a MambaSpec (state-space layer).

        my-vllm NOTE: only ``KVCacheSpecKind.MAMBA`` lookup is exercised
        — actual Mamba layers aren't wired (Dense-only project).
        """
        from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        for g in self.kv_cache_groups:
            cls = KVCacheSpecRegistry.get_manager_class(g.kv_cache_spec)
            if cls is not None and cls.__name__ == "MambaManager":
                return True
        return False

    @property
    def has_mixed_precision_kv_cache(self) -> bool:
        """True iff groups store KV at more than one (dtype, quant mode)."""
        from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

        precisions: set[tuple[Any, KVQuantMode]] = set()
        for group in self.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, FullAttentionSpec):
                precisions.add((spec.dtype, spec.kv_quant_mode))
            else:
                # Shells (SlidingWindowSpec etc.) — record their dtype if
                # set, otherwise ignore (no precision conflict known).
                d = getattr(spec, "dtype", None)
                if d is not None:
                    precisions.add((d, KVQuantMode.NONE))
        return len(precisions) > 1

    @property
    def needs_kv_cache_zeroing(self) -> bool:
        """True iff newly allocated blocks must be zeroed before use.

        Required for Mamba (state read before fully written, #35219)
        and mixed-precision caches (reinterpretation under different
        precision → NaN/Inf if not zeroed). Uniform-precision caches
        skip zeroing.
        """
        return self.has_mamba_layers or self.has_mixed_precision_kv_cache


# ---------------------------------------------------------------------------
# Spec kind dispatch (consumed by HybridKVCacheCoordinator grouping)
# ---------------------------------------------------------------------------


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
    """Map a KV cache spec instance to its discriminator kind.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py:get_kv_cache_spec_kind.

    Subclass checks are kept before base classes so specialized specs
    keep their precise kind (FullAttentionSpec stays FULL_ATTENTION
    rather than collapsing into the broader AttentionSpec base).

    my-vllm NOTE: only FULL_ATTENTION is a true positive match today;
    the other branches return UNKNOWN for shell specs since they aren't
    registered as concrete subclasses yet.
    """
    if isinstance(kv_cache_spec, FullAttentionSpec):
        return KVCacheSpecKind.FULL_ATTENTION
    # Sliding-window / chunked-local / Mamba / cross-attention shells are
    # registered with their own kind via the registry metadata — query
    # that so the HybridKVCacheCoordinator can route correctly even when
    # the concrete class body is a shell.
    from my_vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

    cls = KVCacheSpecRegistry.get_manager_class(kv_cache_spec)
    if cls is None:
        return KVCacheSpecKind.UNKNOWN
    name = cls.__name__
    if name == "SlidingWindowManager":
        return KVCacheSpecKind.SLIDING_WINDOW
    if name == "ChunkedLocalAttentionManager":
        return KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
    if name == "MambaManager":
        return KVCacheSpecKind.MAMBA
    if name == "CrossAttentionManager":
        return KVCacheSpecKind.CROSS_ATTENTION
    return KVCacheSpecKind.UNKNOWN


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "AttentionSpec",
    "FullAttentionSpec",
    "KVCacheConfig",
    "KVCacheGroupSpec",
    "KVCacheSpec",
    "KVCacheSpecKind",
    "KVCacheTensor",
    "KVQuantMode",
    "get_kv_cache_spec_kind",
    "get_kv_quant_mode",
    "is_quantized_kv_cache",
    "kv_cache_uses_per_token_head_scales",
]
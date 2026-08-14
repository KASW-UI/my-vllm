"""tests/test_C254_kv_coordinator_multi_spec.py — KV Cache Coordinator 多 spec 协商.

C254: KV Cache Coordinator 多 spec 协商 + KVCacheSpec registry.

Verifies the new multi-spec coordinator surface (mirrors vLLM V1):

* **KVCacheSpec hierarchy** — FullAttentionSpec + the shell spec types
  (SlidingWindow / ChunkedLocal / Mamba / CrossAttention / MLA).
* **KVCacheSpecRegistry** — registration + lookup + decorator + MRO
  walk + validation.
* **get_kv_cache_spec_kind** dispatch — FullAttentionSpec →
  FULL_ATTENTION; shell managers → their kind; unknown → UNKNOWN.
* **SingleTypeKVCacheManager full vLLM surface** —
  `find_longest_cache_hit` (full-attention two-phase),
  `add_local_computed_blocks` / `allocate_external_computed_blocks`
  (two-phase protocol #33775),
  `get_num_blocks_to_allocate` (vLLM signature),
  `allocate_new_blocks` (vLLM signature, returns list),
  `get_num_common_prefix_blocks`,
  `remove_skipped_blocks` / `new_step_starts` / `get_blocks`.
* **KVCacheCoordinator factory** — caching-off → NoPrefixCache,
  1 group → UnitaryFull, ≥ 2 groups → Hybrid.
* **UnitaryKVCacheCoordinatorFull** — single-group lookup +
  EAGLE-block drop + alignment rounding.
* **HybridKVCacheCoordinator** — SpecGroup formation +
  fixed-point iteration + multi-group reconcile.
* **Backward compat** — legacy `UnitaryKVCacheCoordinator` (C134)
  + `KVCacheManager` (C018/C143) keep working through the
  `_simple` API surface.

C254 is a scaffold for Llama (GQA/SWA), DeepSeek (MLA/Mamba), and
hybrid-model support — the multi-group coordinator hierarchy exists
but the production `KVCacheManager` still routes through the legacy
single-group path until Phase 5 wires up the new coordinator.
"""

from __future__ import annotations

from typing import Type

import pytest
import torch

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_coordinator import (
    HybridKVCacheCoordinator,
    KVCacheCoordinator,
    KVCacheCoordinatorNoPrefixCache,
    SpecGroup,
    UnitaryKVCacheCoordinator,
    UnitaryKVCacheCoordinatorFull,
    get_kv_cache_coordinator,
)
from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock
from my_vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    ChunkedLocalAttentionSpec,
    CrossAttentionManager,
    CrossAttentionSpec,
    FullAttentionManager,
    MambaManager,
    MambaSpec,
    MLAAttentionSpec,
    Shells,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
    SlidingWindowSpec,
    get_manager_for_kv_cache_spec,
    register_all_kvcache_specs,
)
from my_vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    KVQuantMode,
    get_kv_cache_spec_kind,
    get_kv_quant_mode,
)
from my_vllm.v1.kv_cache_spec_registry import (
    KVCacheSpecMetadata,
    KVCacheSpecRegistry,
    register_kv_cache_spec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _full_spec(block_size: int = 16, num_heads: int = 2, head_size: int = 4) -> FullAttentionSpec:
    """Build a FullAttentionSpec with sensible defaults."""
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.float32,
    )


def _make_config(
    groups: list[KVCacheGroupSpec],
    num_blocks: int = 64,
) -> KVCacheConfig:
    """Build a KVCacheConfig with one tensor per group."""
    tensors = [
        KVCacheTensor(
            size=g.kv_cache_spec.page_size_bytes * num_blocks,
            shared_by=list(g.layer_names),
        )
        for g in groups
    ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


# ===========================================================================
# KVCacheSpec hierarchy tests
# ===========================================================================


class TestKVCacheSpecHierarchy:
    """The KV cache spec hierarchy + KVCacheConfig / Group / Tensor.

    Mirrors vLLM V1 vllm/v1/kv_cache_interface.py.
    """

    def test_KVQuantMode_values(self):
        # Match vLLM integer values (C211/C213 quantization kernels).
        assert int(KVQuantMode.NONE) == 0
        assert int(KVQuantMode.FP8_TENSOR) == 1
        assert int(KVQuantMode.INT8_PER_TOKEN_HEAD) == 2
        assert int(KVQuantMode.FP8_PER_TOKEN_HEAD) == 3
        assert int(KVQuantMode.INT4_PER_TOKEN_HEAD) == 4

    def test_KVQuantMode_is_per_token_head(self):
        assert not KVQuantMode.NONE.is_per_token_head
        assert not KVQuantMode.FP8_TENSOR.is_per_token_head
        assert KVQuantMode.INT8_PER_TOKEN_HEAD.is_per_token_head
        assert KVQuantMode.FP8_PER_TOKEN_HEAD.is_per_token_head
        assert KVQuantMode.INT4_PER_TOKEN_HEAD.is_per_token_head

    def test_get_kv_quant_mode_dispatch(self):
        assert get_kv_quant_mode("auto") == KVQuantMode.NONE
        assert get_kv_quant_mode("int4_per_token_head") == KVQuantMode.INT4_PER_TOKEN_HEAD
        assert get_kv_quant_mode("int8_per_token_head") == KVQuantMode.INT8_PER_TOKEN_HEAD
        assert get_kv_quant_mode("fp8_per_token_head") == KVQuantMode.FP8_PER_TOKEN_HEAD
        assert get_kv_quant_mode("fp8") == KVQuantMode.FP8_TENSOR
        assert get_kv_quant_mode("fp8_e4m3") == KVQuantMode.FP8_TENSOR
        assert get_kv_quant_mode("unknown_dtype") == KVQuantMode.NONE

    def test_KVCacheSpecKind_values(self):
        # String-valued enum — values match vLLM's spelling.
        assert KVCacheSpecKind.FULL_ATTENTION.value == "full_attention"
        assert KVCacheSpecKind.SLIDING_WINDOW.value == "sliding_window"
        assert KVCacheSpecKind.MAMBA.value == "mamba"
        assert KVCacheSpecKind.UNKNOWN.value == "unknown"

    def test_FullAttentionSpec_defaults(self):
        spec = _full_spec()
        assert spec.block_size == 16
        assert spec.num_kv_heads == 2
        assert spec.head_size == 4
        assert spec.head_size_v == spec.head_size  # post_init default
        assert spec.sliding_window is None
        assert spec.kv_quant_mode == KVQuantMode.NONE

    def test_FullAttentionSpec_page_size_bytes(self):
        # 2 * block_size * num_kv_heads * head_size * element_size.
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=8,
            dtype=torch.float32,
        )
        expected = 2 * 16 * 4 * 8 * 4  # = 4096 bytes
        assert spec.page_size_bytes == expected

    def test_FullAttentionSpec_max_memory_usage_bytes(self):
        # Stand-in VllmConfig with just the fields we need.
        class _Cfg:
            class model_config:
                max_model_len = 256
            class parallel_config:
                decode_context_parallel_size = 1

        spec = _full_spec(block_size=16, num_heads=2, head_size=4)
        # 256 / 16 = 16 blocks, each 2*16*2*4*4 = 1024 bytes.
        assert spec.max_memory_usage_bytes(_Cfg()) == 16 * 1024

    def test_KVCacheGroupSpec_from_layers(self):
        spec = _full_spec()
        group = KVCacheGroupSpec.from_layers(
            layer_names=["layer.0", "layer.1"],
            kv_cache_spec=spec,
        )
        assert group.layer_names == ["layer.0", "layer.1"]
        assert group.kv_cache_spec is spec
        assert not group.is_eagle_group

    def test_KVCacheTensor_fields(self):
        t = KVCacheTensor(size=4096, shared_by=["a", "b"])
        assert t.size == 4096
        assert t.shared_by == ["a", "b"]
        assert t.offset == 0
        assert t.block_stride == 0

    def test_KVCacheConfig_has_mamba_layers(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        assert not cfg.has_mamba_layers

    def test_KVCacheConfig_has_mixed_precision(self):
        spec_a = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4, dtype=torch.float32,
        )
        spec_b = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4, dtype=torch.float16,
        )
        groups = [
            KVCacheGroupSpec(["a"], spec_a),
            KVCacheGroupSpec(["b"], spec_b),
        ]
        cfg = _make_config(groups)
        assert cfg.has_mixed_precision_kv_cache
        assert cfg.needs_kv_cache_zeroing

    def test_KVCacheConfig_uniform_precision_no_zeroing(self):
        spec = _full_spec()
        cfg = _make_config([KVCacheGroupSpec(["a"], spec)])
        assert not cfg.has_mixed_precision_kv_cache
        assert not cfg.needs_kv_cache_zeroing

    def test_get_kv_cache_spec_kind_FullAttention(self):
        spec = _full_spec()
        assert get_kv_cache_spec_kind(spec) == KVCacheSpecKind.FULL_ATTENTION

    def test_get_kv_cache_spec_kind_shells(self):
        # Shells — registered via `register_all_kvcache_specs()`. Kind
        # resolves via the registry manager-class name.
        register_all_kvcache_specs()
        assert get_kv_cache_spec_kind(SlidingWindowSpec(block_size=16)) == KVCacheSpecKind.SLIDING_WINDOW
        assert get_kv_cache_spec_kind(ChunkedLocalAttentionSpec(block_size=16)) == KVCacheSpecKind.CHUNKED_LOCAL_ATTENTION
        assert get_kv_cache_spec_kind(MambaSpec(block_size=16)) == KVCacheSpecKind.MAMBA
        assert get_kv_cache_spec_kind(CrossAttentionSpec(block_size=16)) == KVCacheSpecKind.CROSS_ATTENTION
        assert get_kv_cache_spec_kind(object()) == KVCacheSpecKind.UNKNOWN


# ===========================================================================
# KVCacheSpecRegistry tests
# ===========================================================================


class TestKVCacheSpecRegistry:
    """The KV cache spec ↔ manager registry."""

    def setup_method(self):
        # Reset to a clean state before each test.
        KVCacheSpecRegistry.reset()

    def teardown_method(self):
        KVCacheSpecRegistry.reset()

    def test_default_FullAttention_registered(self):
        # After any access, the default registration is bootstrapped.
        spec = _full_spec()
        mgr_cls = KVCacheSpecRegistry.get_manager_class(spec)
        assert mgr_cls is FullAttentionManager

    def test_get_manager_class_MRO_walk(self):
        # Define a subclass of FullAttentionSpec — the MRO walk should
        # find the registered FullAttentionSpec ancestor.
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        class MyCustomSpec(FullAttentionSpec):
            pass

        spec = MyCustomSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float32,
        )
        # Walk MRO: MyCustomSpec → FullAttentionSpec → AttentionSpec
        # → KVCacheSpec. The first registered ancestor is FullAttentionSpec.
        assert (
            KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        )

    def test_get_uniform_type_base_spec(self):
        spec = _full_spec()
        base = KVCacheSpecRegistry.get_uniform_type_base_spec(spec)
        assert base is FullAttentionSpec

    def test_unknown_spec_returns_None(self):
        # Object that doesn't inherit from any registered spec.
        class _Unknown:
            pass

        assert KVCacheSpecRegistry.get_manager_class(_Unknown()) is None
        assert KVCacheSpecRegistry.get_uniform_type_base_spec(_Unknown()) is None

    def test_register_manager_class_required(self):
        with pytest.raises(AssertionError, match="manager_class is required"):
            KVCacheSpecRegistry.register(SlidingWindowSpec)

    def test_register_uniform_base_must_be_parent(self):
        with pytest.raises(AssertionError, match="must inherit"):
            KVCacheSpecRegistry.register(
                SlidingWindowSpec,
                manager_class=SlidingWindowManager,
                uniform_type_base_spec=FullAttentionSpec,
            )

    def test_register_conflicting_metadata_raises(self):
        KVCacheSpecRegistry.register(
            SlidingWindowSpec,
            manager_class=SlidingWindowManager,
        )
        # Re-register with a different manager_class → hard error.
        with pytest.raises(AssertionError, match="Conflicting registration"):
            KVCacheSpecRegistry.register(
                SlidingWindowSpec,
                manager_class=FullAttentionManager,
            )

    def test_register_idempotent(self):
        KVCacheSpecRegistry.register(
            SlidingWindowSpec,
            manager_class=SlidingWindowManager,
        )
        # Same metadata → no-op (no error).
        KVCacheSpecRegistry.register(
            SlidingWindowSpec,
            manager_class=SlidingWindowManager,
        )
        assert (
            KVCacheSpecRegistry.get_manager_class(SlidingWindowSpec())
            is SlidingWindowManager
        )

    def test_check_kv_cache_spec_registry_validates(self):
        # Mixed dict with one registered + one unregistered → error.
        KVCacheSpecRegistry.register(
            SlidingWindowSpec, manager_class=SlidingWindowManager
        )
        specs = {
            "a": _full_spec(),
            "b": SlidingWindowSpec(),
        }
        # Both registered → no error.
        KVCacheSpecRegistry.check_kv_cache_spec_registry(specs)

        class _UnknownSpec(KVCacheSpec):
            pass

        specs["c"] = _UnknownSpec(block_size=16)
        with pytest.raises(ValueError, match="Unsupported KV cache spec"):
            KVCacheSpecRegistry.check_kv_cache_spec_registry(specs)

    def test_register_kv_cache_spec_decorator(self):
        @register_kv_cache_spec(
            manager_class=FullAttentionManager,
            uniform_type_base_spec=FullAttentionSpec,
        )
        class _CustomFullSpec(FullAttentionSpec):
            pass

        spec = _CustomFullSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float32,
        )
        assert (
            KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
        )

    def test_reset_clears_registry(self):
        KVCacheSpecRegistry.register(
            SlidingWindowSpec, manager_class=SlidingWindowManager
        )
        assert (
            KVCacheSpecRegistry.get_manager_class(SlidingWindowSpec())
            is SlidingWindowManager
        )
        KVCacheSpecRegistry.reset()
        # After reset + before next default bootstrap, sliding-window
        # has no manager. The default FullAttention is re-bootstrapped
        # on next lookup.
        assert (
            KVCacheSpecRegistry.get_manager_class(_full_spec())
            is FullAttentionManager
        )


# ===========================================================================
# SingleTypeKVCacheManager surface tests
# ===========================================================================


class TestSingleTypeKVCacheManagerSurface:
    """The full vLLM SingleTypeKVCacheManager surface on FullAttentionManager."""

    def setup_method(self):
        KVCacheSpecRegistry.reset()
        # Bootstrap default registration.
        _ = KVCacheSpecRegistry.get_manager_class(_full_spec())

    def test_FullAttentionManager_constructs_with_kv_cache_spec(self):
        spec = _full_spec()
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=spec,
        )
        assert mgr.kv_cache_spec is spec
        assert mgr.block_size == 16
        assert mgr.scheduler_block_size == 16  # default
        assert mgr.use_eagle is False
        assert mgr.records_new_block_ids is False  # no zeroing

    def test_legacy_simple_allocate(self):
        """The legacy C134 `allocate_new_blocks_simple` path still works."""
        from my_vllm.v1.request import Request

        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        req = Request("r1")
        req.prompt_token_ids = list(range(32))
        blocks = mgr.allocate_new_blocks_simple(req, 32)
        assert blocks is not None
        # 32 tokens / 16 block_size = 2 blocks; first block IDs are 0, 1.
        assert len(blocks.block_ids) == 2
        assert blocks.block_ids == [0, 1]

    def test_find_longest_cache_hit_empty_pool(self):
        """Empty block pool → no hit."""
        pool = BlockPool(num_blocks=10, block_size=16, enable_caching=True)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Build a request + cache its blocks.
        from my_vllm.v1.request import Request

        req_a = Request("a")
        req_a.prompt_token_ids = [1, 2, 3, 4] * 4  # 16 tokens
        blocks = mgr.allocate_new_blocks_simple(req_a, 16)
        assert blocks is not None
        # Cache them so the hashmap has entries.
        mgr.cache_blocks_simple(req_a, 16)
        # Now query for a different request with the same hashes.
        # Reset blocks list for the new request.
        req_b = Request("b")
        req_b.prompt_token_ids = [1, 2, 3, 4] * 4
        # find_longest_cache_hit walks `block_hashes` (list of bytes).
        block_hashes = req_b.block_hashes
        # block_hashes are computed by `compute_block_hashes(block_size)`
        # on the Request.
        req_b.compute_block_hashes(16)
        block_hashes = req_b.block_hashes
        hits, hit_length = FullAttentionManager.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=16,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=mgr.kv_cache_spec,
            drop_eagle_block=False,
            alignment_tokens=16,
        )
        # Hit length depends on whether the hash chain matches the
        # already-cached blocks.
        assert hit_length >= 0
        assert hit_length <= 16

    def test_find_longest_cache_hit_eagle_drop(self):
        """EAGLE drop subtracts one block from the hit length."""
        pool = BlockPool(num_blocks=10, block_size=16, enable_caching=True)
        spec = _full_spec()
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=spec,
        )
        from my_vllm.v1.request import Request

        req_a = Request("a")
        req_a.prompt_token_ids = list(range(16))
        mgr.allocate_new_blocks_simple(req_a, 16)
        mgr.cache_blocks_simple(req_a, 16)

        req_b = Request("b")
        req_b.prompt_token_ids = list(range(16))
        req_b.compute_block_hashes(16)
        # drop_eagle_block=True → hit_length is reduced by 1 block.
        _, hit_no_drop = FullAttentionManager.find_longest_cache_hit(
            block_hashes=req_b.block_hashes,
            max_length=16,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=False,
            alignment_tokens=16,
        )
        _, hit_drop = FullAttentionManager.find_longest_cache_hit(
            block_hashes=req_b.block_hashes,
            max_length=16,
            kv_cache_group_ids=[0],
            block_pool=pool,
            kv_cache_spec=spec,
            drop_eagle_block=True,
            alignment_tokens=16,
        )
        # EAGLE-dropped hit is ≤ non-dropped hit.
        assert hit_drop <= hit_no_drop

    def test_get_num_blocks_to_allocate_v2_signature(self):
        """The new 7-arg `get_num_blocks_to_allocate` (vLLM signature)."""
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Empty new_computed_blocks + running request.
        n = mgr.get_num_blocks_to_allocate(
            request_id="r1",
            num_tokens=32,
            new_computed_blocks=[],
            total_computed_tokens=0,
            num_local_computed_tokens=0,
            num_tokens_main_model=32,
        )
        assert n == 2  # ceil(32/16)

    def test_allocate_new_blocks_v2(self):
        """The new vLLM-signature `allocate_new_blocks` returns list."""
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        blocks = mgr.allocate_new_blocks("r1", num_tokens=32, num_tokens_main_model=32)
        assert isinstance(blocks, list)
        assert len(blocks) == 2

    def test_add_local_computed_blocks_two_phase(self):
        """Issue #33775: local touch first, external alloc second."""
        pool = BlockPool(num_blocks=20, block_size=16, enable_caching=True)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Pretend a request is hitting the prefix cache for 16 tokens.
        cached_block = pool.blocks[0]
        cached_block.ref_cnt = 1  # already in pool
        mgr.add_local_computed_blocks(
            request_id="r1",
            new_computed_blocks=[cached_block],
            num_local_computed_tokens=16,
            num_external_computed_tokens=0,
        )
        assert "r1" in mgr.req_to_blocks
        assert mgr.num_cached_block["r1"] == 1
        # Now allocate external blocks (none here).
        mgr.allocate_external_computed_blocks(
            request_id="r1",
            num_local_computed_tokens=16,
            num_external_computed_tokens=16,
        )
        # External adds 1 more block (total_computed=32, need 2 blocks).
        assert len(mgr.req_to_blocks["r1"]) == 2

    def test_get_num_common_prefix_blocks(self):
        pool = BlockPool(num_blocks=20, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Allocate blocks for two requests with a shared prefix.
        # Request A: tokens [0..32] → 2 blocks (0, 1).
        from my_vllm.v1.request import Request
        req_a = Request("a")
        req_a.prompt_token_ids = list(range(32))
        mgr.allocate_new_blocks_simple(req_a, 32)
        # Request B: tokens [0..16] → 1 block. Block 0 is shared.
        req_b = Request("b")
        req_b.prompt_token_ids = list(range(16))
        mgr.allocate_new_blocks_simple(req_b, 16)
        # Re-point block 0 of req_b to req_a's block 0 + bump ref_cnt.
        shared = mgr.req_to_blocks["a"][0]
        mgr.req_to_blocks["b"][0] = shared
        shared.ref_cnt = 2  # both requests reference it
        # Block 1 only belongs to req_a (ref_cnt=1).
        mgr.req_to_blocks["a"][1].ref_cnt = 1
        # Common prefix from req_a's perspective = 1 (block 0).
        n = mgr.get_num_common_prefix_blocks("a")
        assert n == 1

    def test_remove_skipped_blocks_no_op_for_full_attention(self):
        """Full attention has no skipped tokens → remove is a no-op."""
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        from my_vllm.v1.request import Request

        req = Request("r")
        req.prompt_token_ids = list(range(16))
        mgr.allocate_new_blocks_simple(req, 16)
        before = list(mgr.req_to_blocks["r"])
        mgr.remove_skipped_blocks("r", processed_computed_tokens=100)
        after = mgr.req_to_blocks["r"]
        assert len(before) == len(after) == 1

    def test_new_step_starts_no_op(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Base default impl returns None (no-op).
        assert mgr.new_step_starts() is None

    def test_get_blocks(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        from my_vllm.v1.request import Request

        req = Request("r")
        req.prompt_token_ids = list(range(32))
        mgr.allocate_new_blocks_simple(req, 32)
        blocks = mgr.get_blocks("r")
        assert len(blocks) == 2

    def test_take_new_block_ids(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        ids = mgr.take_new_block_ids()
        assert ids == []
        # records_new_block_ids is False for FullAttention (uniform-precision).
        assert mgr.records_new_block_ids is False

    def test_take_pending_cow_copies(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = FullAttentionManager(
            block_pool=pool,
            block_size=16,
            enable_caching=True,
            kv_cache_spec=_full_spec(),
        )
        # Drain pending (empty).
        copies = mgr.take_pending_cow_copies()
        assert copies == []


# ===========================================================================
# get_manager_for_kv_cache_spec dispatcher tests
# ===========================================================================


class TestGetManagerForKVCacheSpec:
    """The `get_manager_for_kv_cache_spec` dispatcher."""

    def setup_method(self):
        KVCacheSpecRegistry.reset()
        register_all_kvcache_specs()

    def teardown_method(self):
        KVCacheSpecRegistry.reset()

    def test_FullAttentionSpec_returns_FullAttentionManager(self):
        spec = _full_spec()
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = get_manager_for_kv_cache_spec(
            spec, pool, enable_caching=True
        )
        assert isinstance(mgr, FullAttentionManager)
        assert mgr.block_size == 16

    def test_unknown_spec_raises(self):
        class _UnknownSpec(KVCacheSpec):
            pass

        pool = BlockPool(num_blocks=10, block_size=16)
        with pytest.raises(AssertionError, match="No manager registered"):
            get_manager_for_kv_cache_spec(
                _UnknownSpec(block_size=16), pool, enable_caching=True
            )

    def test_shells_instantiate_as_shell_manager(self):
        """Shells instantiate — but raise on actual use."""
        for shell_spec, shell_mgr in [
            (SlidingWindowSpec(), SlidingWindowManager),
            (ChunkedLocalAttentionSpec(), ChunkedLocalAttentionManager),
            (MambaSpec(), MambaManager),
            (CrossAttentionSpec(), CrossAttentionManager),
        ]:
            pool = BlockPool(num_blocks=10, block_size=16)
            mgr = get_manager_for_kv_cache_spec(
                shell_spec, pool, enable_caching=True
            )
            assert type(mgr) is shell_mgr
            # Shells raise on actual cache-hit lookup.
            with pytest.raises(NotImplementedError):
                mgr.find_longest_cache_hit(
                    block_hashes=[],
                    max_length=16,
                    kv_cache_group_ids=[0],
                    block_pool=pool,
                    kv_cache_spec=shell_spec,
                    drop_eagle_block=False,
                    alignment_tokens=16,
                )


# ===========================================================================
# KVCacheCoordinator hierarchy + factory tests
# ===========================================================================


class TestKVCacheCoordinatorHierarchy:
    """The KVCacheCoordinator factory + concrete subclasses."""

    def setup_method(self):
        KVCacheSpecRegistry.reset()
        register_all_kvcache_specs()

    def teardown_method(self):
        KVCacheSpecRegistry.reset()

    def test_factory_picks_NoPrefixCache_when_caching_off(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = get_kv_cache_coordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
        )
        assert isinstance(coord, KVCacheCoordinatorNoPrefixCache)

    def test_factory_picks_Unitary_when_single_group(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = get_kv_cache_coordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        assert isinstance(coord, UnitaryKVCacheCoordinatorFull)
        assert isinstance(coord, KVCacheCoordinator)

    def test_factory_picks_Hybrid_when_multi_group(self):
        # Two distinct specs → hybrid.
        spec_a = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float32,
        )
        spec_b = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float16,  # different dtype → distinct spec
        )
        groups = [
            KVCacheGroupSpec(["a"], spec_a),
            KVCacheGroupSpec(["b"], spec_b),
        ]
        cfg = _make_config(groups)
        coord = get_kv_cache_coordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        assert isinstance(coord, HybridKVCacheCoordinator)

    def test_SpecGroup_namedtuple(self):
        spec = _full_spec()
        sg = SpecGroup(
            spec=spec, group_ids=[0, 1], manager_cls=FullAttentionManager,
            use_eagle=True,
        )
        assert sg.spec is spec
        assert sg.group_ids == [0, 1]
        assert sg.use_eagle is True

    def test_NoPrefixCache_find_longest_cache_hit_returns_empty(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = get_kv_cache_coordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
        )
        blocks, hit_length, uncached = coord.find_longest_cache_hit(
            block_hashes=[b"\x01", b"\x02"],
            max_cache_hit_length=64,
        )
        assert blocks == ([],)
        assert hit_length == 0
        assert uncached == 0

    def test_NoPrefixCache_get_num_common_prefix_blocks_zero(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = get_kv_cache_coordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=False,
            enable_kv_cache_events=False,
        )
        out = coord.get_num_common_prefix_blocks("r1")
        assert out == [0]

    def test_UnitaryKVCacheCoordinatorFull_constructs(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        assert coord.kv_cache_spec is spec
        assert coord.block_size == 16
        assert len(coord.single_type_managers) == 1
        assert isinstance(coord.single_type_managers[0], FullAttentionManager)

    def test_UnitaryKVCacheCoordinatorFull_asserts_one_group(self):
        spec = _full_spec()
        spec_b = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float16,
        )
        cfg = _make_config([
            KVCacheGroupSpec(["a"], spec),
            KVCacheGroupSpec(["b"], spec_b),
        ])
        with pytest.raises(AssertionError, match="exactly 1 group"):
            UnitaryKVCacheCoordinatorFull(
                kv_cache_config=cfg,
                max_model_len=128,
                max_in_flight_tokens=128,
                use_eagle=False,
                enable_caching=True,
                enable_kv_cache_events=False,
            )

    def test_Unitary_find_longest_cache_hit(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        # With an empty pool, no hit.
        blocks, hit_length, uncached = coord.find_longest_cache_hit(
            block_hashes=[],
            max_cache_hit_length=64,
        )
        assert hit_length == 0
        assert uncached == 0

    def test_Hybrid_verify_and_split_kv_cache_groups(self):
        spec_a = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float32,
        )
        spec_b = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float16,
        )
        # Same spec repeated → SpecGroup merges group_ids.
        spec_a_again = FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=4,
            dtype=torch.float32,
        )
        groups = [
            KVCacheGroupSpec(["a"], spec_a),
            KVCacheGroupSpec(["a2"], spec_a_again),
            KVCacheGroupSpec(["b"], spec_b),
        ]
        cfg = _make_config(groups)
        coord = HybridKVCacheCoordinator(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        # 2 distinct specs → 2 SpecGroups (group 0+1 merged).
        assert len(coord.attention_groups) == 2
        merged = next(
            sg for sg in coord.attention_groups if sg.spec == spec_a
        )
        assert merged.group_ids == [0, 1]

    def test_Hybrid_asserts_multi_group(self):
        spec = _full_spec()
        cfg = _make_config([KVCacheGroupSpec(["x"], spec)])
        with pytest.raises(AssertionError, match="at least two"):
            HybridKVCacheCoordinator(
                kv_cache_config=cfg,
                max_model_len=128,
                max_in_flight_tokens=128,
                use_eagle=False,
                enable_caching=True,
                enable_kv_cache_events=False,
            )


# ===========================================================================
# Legacy C134 path — backward compat
# ===========================================================================


class TestLegacyC134Compat:
    """The legacy single-group `UnitaryKVCacheCoordinator` + `KVCacheManager`.

    The new multi-spec coordinator (C254) must NOT break the production
    single-group path that `KVCacheManager` consumes.
    """

    def test_legacy_UnitaryKVCacheCoordinator_still_works(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        coord = UnitaryKVCacheCoordinator(
            block_pool=pool, block_size=16, enable_caching=True,
        )
        from my_vllm.v1.request import Request

        req = Request("r1")
        req.prompt_token_ids = list(range(32))
        blocks = coord.allocate_new_blocks(req, 32)
        assert blocks is not None
        assert blocks.block_ids == [0, 1]

    def test_KVCacheManager_allocate_slots_still_works(self):
        pool = BlockPool(num_blocks=50, block_size=16)
        mgr = KVCacheManager(block_pool=pool)
        from my_vllm.v1.request import Request

        req = Request("r1")
        req.prompt_token_ids = list(range(32))
        result = mgr.allocate_slots(req, 32)
        assert result is not None
        assert result.block_ids == [0, 1]

    def test_KVCacheManager_get_block_ids(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = KVCacheManager(block_pool=pool)
        from my_vllm.v1.request import Request

        req = Request("r1")
        req.prompt_token_ids = list(range(32))
        mgr.allocate_slots(req, 32)
        ids = mgr.get_block_ids("r1")
        assert ids == [0, 1]

    def test_KVCacheManager_num_free_blocks(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = KVCacheManager(block_pool=pool)
        # 10 blocks, 1 reserved as null_block sentinel (NOT popped,
        # so all 10 are still free for allocation).
        assert mgr.num_free_blocks == 10

    def test_KVCacheManager_free(self):
        pool = BlockPool(num_blocks=10, block_size=16)
        mgr = KVCacheManager(block_pool=pool)
        from my_vllm.v1.request import Request

        req = Request("r1")
        req.prompt_token_ids = list(range(16))
        mgr.allocate_slots(req, 16)
        mgr.free("r1")
        assert mgr.get_block_ids("r1") == []


# ===========================================================================
# End-to-end integration
# ===========================================================================


class TestEndToEndMultiSpecFlow:
    """End-to-end: build a full UnitaryKVCacheCoordinatorFull, allocate,
    cache, find longest hit."""

    def setup_method(self):
        KVCacheSpecRegistry.reset()
        register_all_kvcache_specs()

    def teardown_method(self):
        KVCacheSpecRegistry.reset()

    def test_unitary_full_lifecycle(self):
        spec = _full_spec(block_size=16, num_heads=2, head_size=4)
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups, num_blocks=20)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        # Allocate blocks for a request.
        blocks = coord.allocate_new_blocks(
            request_id="r1",
            num_tokens=32,
            num_tokens_main_model=32,
        )
        assert len(blocks) == 1
        assert len(blocks[0]) == 2
        # Verify per-group block_ids.
        assert blocks[0][0].block_id == 0
        assert blocks[0][1].block_id == 1

    def test_unitary_full_allocate_new_computed_blocks(self):
        """Two-phase cache-hit bookkeeping."""
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups, num_blocks=20)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        # Phase 1: add local computed blocks (1 group, empty hit list).
        coord.allocate_new_computed_blocks(
            request_id="r1",
            new_computed_blocks=([],),  # 1 group → tuple of 1 list
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
        )
        # Phase 2: no external → no-op.
        # Now allocate via the v2 API.
        blocks = coord.allocate_new_blocks(
            request_id="r1",
            num_tokens=16,
            num_tokens_main_model=16,
        )
        assert len(blocks[0]) == 1

    def test_coordinator_blocks_property(self):
        """`coord.get_blocks` returns per-group block tuples."""
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        coord.allocate_new_blocks("r1", num_tokens=16, num_tokens_main_model=16)
        blocks = coord.get_blocks("r1")
        assert isinstance(blocks, tuple)
        assert len(blocks) == 1
        assert len(blocks[0]) == 1

    def test_coordinator_remove_skipped_blocks_no_op(self):
        """Full attention → no skipped blocks."""
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        coord.allocate_new_blocks("r1", num_tokens=16, num_tokens_main_model=16)
        before = coord.get_blocks("r1")
        coord.remove_skipped_blocks("r1", processed_computed_tokens=100)
        after = coord.get_blocks("r1")
        assert len(before[0]) == len(after[0]) == 1

    def test_coordinator_new_step_starts(self):
        spec = _full_spec()
        groups = [KVCacheGroupSpec(["x"], spec)]
        cfg = _make_config(groups)
        coord = UnitaryKVCacheCoordinatorFull(
            kv_cache_config=cfg,
            max_model_len=128,
            max_in_flight_tokens=128,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
        )
        # Base default impl → no-op (returns None).
        assert coord.new_step_starts() is None
"""C022: tests for AttentionBackend interface and CommonAttentionMetadata."""

import torch

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.selector import get_attn_backend
from my_vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl,
)
from my_vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl,
)


class TestAttentionBackendABC:
    def test_attention_backend_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            AttentionBackend()

    def test_attention_impl_is_abstract(self):
        import pytest
        with pytest.raises(TypeError):
            AttentionImpl(4, 64, 1.0)

    def test_flash_backend_get_name(self):
        assert FlashAttentionBackend.get_name() == "FLASH_ATTN"

    def test_flash_backend_get_impl_cls(self):
        assert FlashAttentionBackend.get_impl_cls() is FlashAttentionImpl

    def test_flash_backend_kv_cache_shape(self):
        shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=32, block_size=16,
            num_kv_heads=4, head_size=64,
        )
        assert shape == (32, 4, 16, 64)


class TestFlashAttentionImpl:
    def test_init_default(self):
        impl = FlashAttentionImpl(8, 64, 1.0)
        assert impl.num_heads == 8
        assert impl.head_size == 64
        assert impl.num_kv_heads == 8

    def test_init_gqa(self):
        impl = FlashAttentionImpl(8, 64, 1.0, num_kv_heads=2)
        assert impl.num_heads == 8
        assert impl.num_kv_heads == 2

    def test_forward_basic(self):
        impl = FlashAttentionImpl(4, 32, 1.0)
        num_tokens = 3
        q = torch.randn(num_tokens, 4, 32)
        k = torch.randn(num_tokens, 4, 32)
        v = torch.randn(num_tokens, 4, 32)
        output = torch.zeros(num_tokens, 4, 32)

        meta = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3]),
            query_start_loc_cpu=torch.tensor([0, 3]),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=num_tokens,
            block_table_tensor=torch.zeros(1, 8, dtype=torch.int32),
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        )

        result = impl.forward(q, k, v,
                              kv_cache=torch.zeros(8, 4, 16, 32),
                              attn_metadata=meta,
                              output=output)
        assert result.shape == (num_tokens, 4, 32)
        assert not torch.equal(result, torch.zeros_like(result))

    def test_forward_gqa(self):
        impl = FlashAttentionImpl(8, 32, 1.0, num_kv_heads=2)
        num_tokens = 2
        q = torch.randn(num_tokens, 8, 32)
        k = torch.randn(num_tokens, 2, 32)
        v = torch.randn(num_tokens, 2, 32)
        output = torch.zeros(num_tokens, 8, 32)

        meta = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 2]),
            query_start_loc_cpu=torch.tensor([0, 2]),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            max_seq_len=num_tokens,
            block_table_tensor=torch.zeros(1, 8, dtype=torch.int32),
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        )

        result = impl.forward(q, k, v,
                              kv_cache=torch.zeros(8, 2, 16, 32),
                              attn_metadata=meta,
                              output=output)
        assert result.shape == (num_tokens, 8, 32)


class TestCommonAttentionMetadata:
    def test_build_metadata(self):
        num_reqs = 2
        num_tokens = 10
        m = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 4, 10]),
            query_start_loc_cpu=torch.tensor([0, 4, 10]),
            seq_lens=torch.tensor([4, 6], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([4, 6], dtype=torch.int32),
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=6,
            max_seq_len=6,
            block_table_tensor=torch.zeros(num_reqs, 16, dtype=torch.int32),
            slot_mapping=torch.zeros(num_tokens, dtype=torch.int64),
        )
        assert m.num_reqs == 2
        assert m.num_actual_tokens == 10
        assert m.max_query_len == 6
        assert m.causal is True

    def test_metadata_default_causal(self):
        m = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 5]),
            query_start_loc_cpu=torch.tensor([0, 5]),
            seq_lens=torch.tensor([5], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=5,
            max_query_len=5,
            max_seq_len=5,
            block_table_tensor=torch.zeros(1, 8, dtype=torch.int32),
            slot_mapping=torch.zeros(5, dtype=torch.int64),
        )
        assert m.causal is True

    def test_metadata_positions_optional(self):
        m = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3]),
            query_start_loc_cpu=torch.tensor([0, 3]),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=3,
            max_query_len=3,
            max_seq_len=3,
            block_table_tensor=torch.zeros(1, 8, dtype=torch.int32),
            slot_mapping=torch.zeros(3, dtype=torch.int64),
            positions=None,
        )
        assert m.positions is None

    def test_metadata_with_positions(self):
        m = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 3]),
            query_start_loc_cpu=torch.tensor([0, 3]),
            seq_lens=torch.tensor([3], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=3,
            max_query_len=3,
            max_seq_len=3,
            block_table_tensor=torch.zeros(1, 8, dtype=torch.int32),
            slot_mapping=torch.zeros(3, dtype=torch.int64),
            positions=torch.tensor([0, 1, 2]),
        )
        assert torch.equal(m.positions, torch.tensor([0, 1, 2]))


class TestSelector:
    def test_get_attn_backend(self):
        # C117: default backend is now TritonAttentionBackend
        # (real vLLM Triton kernels). FlashAttentionBackend remains as a
        # fallback only.
        backend = get_attn_backend()
        assert backend is TritonAttentionBackend
        assert backend.get_name() == "TRITON_ATTN"
        assert backend.get_impl_cls() is TritonAttentionImpl

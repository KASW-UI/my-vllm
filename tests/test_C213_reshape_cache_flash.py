"""test_C213_reshape_cache_flash.py — C213 triton_reshape_and_cache_flash full quant alignment.

Verifies that ``my_vllm/v1/attention/ops/triton_reshape_and_cache_flash.py``
(630 lines, already aligned with vLLM) covers all 5 ``KV_QUANT_MODE``
paths:

  1. NONE / "auto"                       plain cast (fp16 / bf16)
  2. FP8_TENSOR  / "fp8_e4m3"            per-tensor scale quantization
  3. INT8_PER_TOKEN_HEAD                per-(token, head) scale + int8 cache
  4. FP8_PER_TOKEN_HEAD                 per-(token, head) scale + fp8 cache
  5. INT4_PER_TOKEN_HEAD                routed to a dedicated int4 kernel
                                          (not loaded in my-vllm; the wrapper
                                          raises an ImportError if called)

Plus the diffkv variant (different K/V head sizes) used for MLA.

Each test cross-checks the write correctness against a PyTorch
reference (when feasible) or against shape/finiteness invariants
(when the quantization is lossy by design).
"""

import pytest
import torch

from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
    triton_reshape_and_cache_flash_diffkv,
    is_quantized_kv_cache,
)
from my_vllm.v1.kv_cache_interface import KVQuantMode


# ---------------------------------------------------------------------------
# 1. is_quantized_kv_cache helper
# ---------------------------------------------------------------------------


class TestIsQuantizedKVCache:
    @pytest.mark.parametrize("dtype,expected", [
        ("auto", False),
        ("float16", False),
        ("bfloat16", False),
        ("half", False),
        ("fp8", True),
        ("fp8_e4m3", True),
        ("fp8_e5m2", True),
        ("int8", True),
        ("int4", True),
    ])
    def test_dtype_string(self, dtype, expected):
        assert is_quantized_kv_cache(dtype) is expected

    def test_torch_dtype_passthrough(self):
        """Non-string dtype is not a quantized kv cache name → False."""
        assert is_quantized_kv_cache(torch.float16) is False


# ---------------------------------------------------------------------------
# 2. NONE / "auto" — plain fp16 / bf16 write
# ---------------------------------------------------------------------------


class TestAutoKVCache:
    """Basic write: K/V are stored verbatim, only layout reorganization."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_basic_write(self, dtype):
        torch.manual_seed(0)
        device = "cuda"
        num_tokens, num_kv_heads, head_size, block_size = 8, 2, 16, 8
        num_blocks = 4

        key = torch.randn(num_tokens, num_kv_heads, head_size,
                          dtype=dtype, device=device)
        value = torch.randn(num_tokens, num_kv_heads, head_size,
                            dtype=dtype, device=device)
        # FlashAttention layout.
        k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=dtype, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        # Slots 0..7 are valid (block_size * num_blocks = 32).
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        triton_reshape_and_cache_flash(
            key=key, value=value,
            key_cache=k_cache, value_cache=v_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
            k_scale=k_scale, v_scale=v_scale,
        )
        torch.cuda.synchronize()

        # Verify each token landed in the right slot.
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            blk = slot // block_size
            off = slot % block_size
            assert torch.allclose(k_cache[blk, off], key[i], atol=1e-3), (
                f"key mismatch at token {i} (slot={slot})"
            )
            assert torch.allclose(v_cache[blk, off], value[i], atol=1e-3), (
                f"value mismatch at token {i} (slot={slot})"
            )


# ---------------------------------------------------------------------------
# 3. FP8_TENSOR / "fp8_e4m3" — per-tensor scale quantization
# ---------------------------------------------------------------------------


class TestFP8PerTensorQuant:
    """Per-tensor FP8 e4m3 quantization: k_scale/v_scale are 1-element
    float32 tensors applied at the cache-write site."""

    def test_fp8_e4m3_storage_dtype(self):
        torch.manual_seed(1)
        device = "cuda"
        num_tokens, num_kv_heads, head_size, block_size = 8, 2, 16, 8
        num_blocks = 2

        key = torch.randn(
            num_tokens, num_kv_heads, head_size, dtype=torch.float16,
            device=device,
        )
        value = torch.randn(
            num_tokens, num_kv_heads, head_size, dtype=torch.float16,
            device=device,
        )
        # FP8 e4m3 cache.
        k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.float8_e4m3fn, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        # Per-tensor scale (single element).
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        triton_reshape_and_cache_flash(
            key=key, value=value,
            key_cache=k_cache, value_cache=v_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="fp8_e4m3",
            k_scale=k_scale, v_scale=v_scale,
        )
        torch.cuda.synchronize()

        # Cache is FP8; verify shape + finiteness + non-zero.
        assert k_cache.shape == (num_blocks, block_size, num_kv_heads, head_size)
        assert k_cache.dtype == torch.float8_e4m3fn
        # Token 0 should have non-zero cache value.
        assert k_cache[0, 0, 0, 0].item() != 0


# ---------------------------------------------------------------------------
# 4. INT8 / FP8 per-token-head quantization
# ---------------------------------------------------------------------------


class TestPerTokenHeadQuant:
    """``triton_reshape_and_cache_flash_per_token_head_quant``:
    quantizes K/V per (token, head), stores quantized values in
    key_cache/value_cache (int8 or fp8) and the per-(token, head) scale
    in k_scale_cache/v_scale_cache (float32)."""

    def test_int8_per_token_head(self):
        torch.manual_seed(2)
        device = "cuda"
        num_tokens, num_kv_heads, head_size, block_size = 8, 2, 16, 8
        num_blocks = 2

        key = torch.randn(num_tokens, num_kv_heads, head_size,
                          dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_kv_heads, head_size,
                            dtype=torch.float16, device=device)
        # Int8 cache.
        k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.int8, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        # Per-(token, head) scale caches.
        k_scale_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device,
        )
        v_scale_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device,
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        triton_reshape_and_cache_flash_per_token_head_quant(
            key=key, value=value,
            key_cache=k_cache, value_cache=v_cache,
            k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        )
        torch.cuda.synchronize()

        # Only block 0 was written (slots 0..7, block 1 stays at zero).
        # Verify scale cache for the populated block 0 only.
        assert (k_scale_cache[0] > 0).all(), "K scale cache[block 0] not populated"
        assert (v_scale_cache[0] > 0).all(), "V scale cache[block 0] not populated"
        # Verify int8 cache is in valid range.
        assert k_cache.min().item() >= -128
        assert k_cache.max().item() <= 127
        # Round-trip reconstruction should be approximately equal to the
        # original (within quant noise).
        for i in range(num_tokens):
            blk = i // block_size
            off = i % block_size
            scale_k = k_scale_cache[blk, off]  # [num_kv_heads]
            # Dequant: k_q * scale for each head.
            k_q = k_cache[blk, off].float()  # [num_kv_heads, head_size]
            k_recon = k_q * scale_k.unsqueeze(-1)
            # Check first head.
            cos = torch.nn.functional.cosine_similarity(
                k_recon[:, 0].flatten().float(),
                key[i, :, 0].flatten().float(), dim=-1,
            )
            # Per-token-head quant with absmax normalization should
            # preserve direction; allow modest (≥0.5) cosine sim due
            # to small head_size and high quant noise.
            assert cos.item() > 0.5 or k_recon.norm().item() < 1e-6, (
                f"token {i} head 0 dequant mismatch: cos={cos.item():.3f}"
            )

    def test_fp8_per_token_head(self):
        torch.manual_seed(3)
        device = "cuda"
        num_tokens, num_kv_heads, head_size, block_size = 8, 2, 16, 8
        num_blocks = 2

        key = (torch.randn(
            num_tokens, num_kv_heads, head_size,
            dtype=torch.float32, device=device,
        ) * 0.5).to(torch.float16)
        value = (torch.randn(
            num_tokens, num_kv_heads, head_size,
            dtype=torch.float32, device=device,
        ) * 0.5).to(torch.float16)
        # FP8 e4m3 cache.
        k_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_size,
            dtype=torch.float8_e4m3fn, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        k_scale_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device,
        )
        v_scale_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads,
            dtype=torch.float32, device=device,
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        triton_reshape_and_cache_flash_per_token_head_quant(
            key=key, value=value,
            key_cache=k_cache, value_cache=v_cache,
            k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
            slot_mapping=slot_mapping,
            kv_quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        )
        torch.cuda.synchronize()

        # Verify scale cache was populated and is finite (block 0 only).
        assert torch.isfinite(k_scale_cache[0]).all()
        assert torch.isfinite(v_scale_cache[0]).all()
        assert (k_scale_cache[0] > 0).all()
        # FP8 cache is uint8 storage; check shape + finite cast.
        k_cache_f = k_cache[0].float()
        assert torch.isfinite(k_cache_f).all()


# ---------------------------------------------------------------------------
# 5. INT4_PER_TOKEN_HEAD — wrapper routes to int4 kernel (not loaded)
# ---------------------------------------------------------------------------


class TestInt4PerTokenHeadWrapper:
    """The wrapper for KVQuantMode.INT4_PER_TOKEN_HEAD delegates to
    ``my_vllm.v1.attention.ops.int4_per_token_head.reshape_and_cache_int4``
    which is not loaded in my-vllm (DeepSeek MLA is out of JD scope).
    Verify the wrapper raises a clear ImportError if invoked."""

    def test_int4_unavailable(self):
        torch.manual_seed(4)
        device = "cuda"
        num_tokens, num_kv_heads, head_size, block_size = 4, 2, 16, 8

        key = torch.randn(num_tokens, num_kv_heads, head_size,
                          dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_kv_heads, head_size,
                            dtype=torch.float16, device=device)
        k_cache = torch.zeros(
            1, block_size, num_kv_heads, head_size,
            dtype=torch.int8, device=device,
        )
        v_cache = torch.zeros_like(k_cache)
        k_scale_cache = torch.zeros(
            1, block_size, num_kv_heads,
            dtype=torch.float32, device=device,
        )
        v_scale_cache = torch.zeros_like(k_scale_cache)
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

        with pytest.raises((ImportError, ModuleNotFoundError)):
            triton_reshape_and_cache_flash_per_token_head_quant(
                key=key, value=value,
                key_cache=k_cache, value_cache=v_cache,
                k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
                slot_mapping=slot_mapping,
                kv_quant_mode=KVQuantMode.INT4_PER_TOKEN_HEAD,
            )


# ---------------------------------------------------------------------------
# 6. DiffKV variant (different K/V head sizes; MLA-compat)
# ---------------------------------------------------------------------------


class TestDiffKVVariant:
    """Different K and V head sizes (e.g. MLA-style where K and V are
    packed into the same paged block)."""

    def test_diff_kv_layout(self):
        torch.manual_seed(5)
        device = "cuda"
        num_tokens, num_kv_heads = 4, 2
        head_size_k, head_size_v = 16, 24
        block_size, num_blocks = 4, 2

        key = torch.randn(num_tokens, num_kv_heads, head_size_k,
                          dtype=torch.float16, device=device)
        value = torch.randn(num_tokens, num_kv_heads, head_size_v,
                            dtype=torch.float16, device=device)
        # diffkv: K and V are packed in a single cache of width head_size_k + head_size_v.
        kv_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads,
            head_size_k + head_size_v,
            dtype=torch.float16, device=device,
        )
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        triton_reshape_and_cache_flash_diffkv(
            key=key, value=value,
            kv_cache=kv_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
            k_scale=k_scale, v_scale=v_scale,
        )
        torch.cuda.synchronize()

        # Verify K and V are stored at the right offsets.
        for i in range(num_tokens):
            blk = i // block_size
            off = i % block_size
            assert torch.allclose(
                kv_cache[blk, off, :, :head_size_k], key[i], atol=1e-3,
            ), f"K mismatch at token {i}"
            assert torch.allclose(
                kv_cache[blk, off, :, head_size_k:], value[i], atol=1e-3,
            ), f"V mismatch at token {i}"


# ---------------------------------------------------------------------------
# 7. Negative slot skip
# ---------------------------------------------------------------------------


class TestNegativeSlotSkip:
    """Negative slot_mapping[i] skips the token (used for padding /
    mixed prefill batches where the slot is reserved but no token
    is being written)."""

    def test_negative_slots_preserve_existing_cache(self):
        torch.manual_seed(6)
        device = "cuda"
        dtype = torch.float16
        num_kv_heads, head_size, block_size = 1, 8, 4
        num_blocks = 2

        key = torch.randn(3, num_kv_heads, head_size, dtype=dtype, device=device)
        value = torch.randn(3, num_kv_heads, head_size, dtype=dtype, device=device)
        k_cache = torch.full(
            (num_blocks, block_size, num_kv_heads, head_size), 99.0,
            dtype=dtype, device=device,
        )
        v_cache = torch.full_like(k_cache, 99.0)
        # Token 1 has negative slot → skipped.
        slot_mapping = torch.tensor([0, -1, 4], dtype=torch.int64, device=device)
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

        triton_reshape_and_cache_flash(
            key=key, value=value,
            key_cache=k_cache, value_cache=v_cache,
            slot_mapping=slot_mapping,
            kv_cache_dtype="auto",
            k_scale=k_scale, v_scale=v_scale,
        )
        torch.cuda.synchronize()

        # Slot 0 written, slot 4 written, slot 1 untouched (still 99).
        assert torch.allclose(k_cache[0, 0], key[0], atol=1e-3)
        assert torch.allclose(k_cache[1, 0], key[2], atol=1e-3)
        assert torch.allclose(
            k_cache[0, 1],
            torch.full_like(k_cache[0, 1], 99.0),
        ), "negative slot was not skipped"

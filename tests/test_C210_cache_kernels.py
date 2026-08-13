"""test_C210_cache_kernels.py — C210 cache_kernels.cu 多 op + 多 dtype 升级

覆盖 ``my_vllm/_C`` 暴露的全部 7 个 cache block op:

    reshape_and_cache        fp16 / bf16  写入（HND layout）
    reshape_and_cache_flash  fp16 / bf16  写入（FlashAttention layout）
    swap_blocks              任意 dtype   cudaMemcpyAsync 字节交换
    copy_blocks              fp16 / bf16  GPU 端 block → block 复制
    cp_gather_cache          fp16 / bf16  非连续 block 收集
    convert_fp8              fp32/fp16/bf16 ↔ uint8 (FP8 e4m3)
    concat_and_cache_mla     fp16 / bf16  MLA 压缩 KV 写入（stub）

所有测试在 CUDA 不可用或扩展未编译时跳过；扩展未编译的情况下仅做
存在性校验。  hot path 仍由 Triton kernel (C149) 提供；CUDA op 提供
CUDA-graph fallback / FP8 转换 / prefix-cache CoW gather / MLA API 对称。
"""

import pytest
import torch

from my_vllm._custom_ops import _C_CACHE_OPS


# ---------------------------------------------------------------------------
# 编译可用性 / 公共 API 校验
# ---------------------------------------------------------------------------

class TestCacheKernelBuild:
    def test_c_extension_compiled(self):
        assert _C_CACHE_OPS is not None, (
            "my_vllm._C 扩展未编译 — 运行 `python setup.py build_ext "
            "--inplace -j3` 后重跑本测试。"
        )

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_all_ops_exposed(self):
        expected = {
            "reshape_and_cache",
            "reshape_and_cache_flash",
            "swap_blocks",
            "copy_blocks",
            "cp_gather_cache",
            "convert_fp8",
            "concat_and_cache_mla",
        }
        actual = {n for n in dir(_C_CACHE_OPS) if not n.startswith("_")}
        missing = expected - actual
        assert not missing, f"missing cache ops: {missing}"


# ---------------------------------------------------------------------------
# reshape_and_cache — multi-dtype cache write (HND layout)
# ---------------------------------------------------------------------------

class TestReshapeAndCache:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_basic_write(self, dtype):
        num_tokens, num_heads, head_size, block_size = 4, 2, 8, 4
        num_blocks = 8

        key = torch.randn(num_tokens, num_heads, head_size,
                          dtype=dtype, device="cuda")
        value = torch.randn(num_tokens, num_heads, head_size,
                            dtype=dtype, device="cuda")
        # HND layout: [num_blocks, num_heads, block_size, head_size]
        k_cache = torch.zeros(num_blocks, num_heads, block_size, head_size,
                              dtype=dtype, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        # slot_mapping[i] = block_id * block_size + offset
        slot_mapping = torch.tensor(
            [0, 1, 4, 5], dtype=torch.int64, device="cuda",
        )
        k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        _C_CACHE_OPS.reshape_and_cache(
            key, value, k_cache, v_cache, slot_mapping,
            "auto", k_scale, v_scale,
        )

        for i in range(num_tokens):
            blk = slot_mapping[i].item() // block_size
            off = slot_mapping[i].item() % block_size
            assert torch.allclose(
                k_cache[blk, :, off, :], key[i], atol=1e-3,
            ), f"key mismatch at token {i}"
            assert torch.allclose(
                v_cache[blk, :, off, :], value[i], atol=1e-3,
            ), f"value mismatch at token {i}"

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_negative_slot_skipped(self):
        dtype = torch.float16
        num_heads, head_size, block_size = 1, 4, 4
        num_blocks = 4

        key = torch.randn(3, num_heads, head_size, dtype=dtype, device="cuda")
        value = torch.randn(3, num_heads, head_size, dtype=dtype, device="cuda")
        k_cache = torch.full(
            (num_blocks, num_heads, block_size, head_size), 99.0,
            dtype=dtype, device="cuda",
        )
        v_cache = torch.full_like(k_cache, 99.0)
        # token 1 has negative slot → skipped.
        slot_mapping = torch.tensor([0, -1, 4], dtype=torch.int64, device="cuda")
        k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        _C_CACHE_OPS.reshape_and_cache(
            key, value, k_cache, v_cache, slot_mapping,
            "auto", k_scale, v_scale,
        )

        # slot 0 written, slot 4 written, slot 1 untouched (still 99).
        assert torch.allclose(k_cache[0, :, 0, :], key[0], atol=1e-3)
        assert torch.allclose(k_cache[1, :, 0, :], key[2], atol=1e-3)
        # Block 0 offset 1 should still be sentinel 99.
        assert torch.allclose(
            k_cache[0, :, 1, :],
            torch.full_like(k_cache[0, :, 1, :], 99.0),
        )

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_fp8_storage_rejected(self):
        """FP8 storage is owned by the Triton kernel (C149); CUDA path rejects it."""
        dtype = torch.float16
        num_heads, head_size, block_size = 1, 4, 4
        num_blocks = 2

        key = torch.zeros(1, num_heads, head_size, dtype=dtype, device="cuda")
        value = torch.zeros(1, num_heads, head_size, dtype=dtype, device="cuda")
        k_cache = torch.zeros(
            num_blocks, num_heads, block_size, head_size, dtype=torch.uint8,
            device="cuda",
        )
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.tensor([0], dtype=torch.int64, device="cuda")
        k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError, match="FP8 cache path|Triton"):
            _C_CACHE_OPS.reshape_and_cache(
                key, value, k_cache, v_cache, slot_mapping,
                "fp8_e4m3", k_scale, v_scale,
            )


# ---------------------------------------------------------------------------
# reshape_and_cache_flash — FlashAttention layout variant
# ---------------------------------------------------------------------------

class TestReshapeAndCacheFlash:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_flash_layout_write(self, dtype):
        num_tokens, num_heads, head_size, block_size = 3, 2, 8, 4
        num_blocks = 4

        key = torch.randn(num_tokens, num_heads, head_size,
                          dtype=dtype, device="cuda")
        value = torch.randn(num_tokens, num_heads, head_size,
                            dtype=dtype, device="cuda")
        # FlashAttention layout: [num_blocks, block_size, num_heads, head_size]
        k_cache = torch.zeros(num_blocks, block_size, num_heads, head_size,
                              dtype=dtype, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.tensor([0, 1, 8], dtype=torch.int64, device="cuda")
        k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        _C_CACHE_OPS.reshape_and_cache_flash(
            key, value, k_cache, v_cache, slot_mapping,
            "auto", k_scale, v_scale,
        )

        for i in range(num_tokens):
            blk = slot_mapping[i].item() // block_size
            off = slot_mapping[i].item() % block_size
            assert torch.allclose(
                k_cache[blk, off, :, :], key[i], atol=1e-3,
            ), f"flash key mismatch at token {i}"

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_flash_layout_fp8_rejected(self):
        dtype = torch.float16
        num_heads, head_size, block_size = 1, 4, 4

        key = torch.zeros(1, num_heads, head_size, dtype=dtype, device="cuda")
        value = torch.zeros(1, num_heads, head_size, dtype=dtype, device="cuda")
        k_cache = torch.zeros(1, block_size, num_heads, head_size,
                              dtype=dtype, device="cuda")
        v_cache = torch.zeros_like(k_cache)
        slot_mapping = torch.tensor([0], dtype=torch.int64, device="cuda")
        k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError, match="not supported|Triton"):
            _C_CACHE_OPS.reshape_and_cache_flash(
                key, value, k_cache, v_cache, slot_mapping,
                "fp8_e4m3", k_scale, v_scale,
            )


# ---------------------------------------------------------------------------
# swap_blocks — cudaMemcpyAsync over (src_block, dst_block) pairs
# ---------------------------------------------------------------------------

class TestSwapBlocks:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_basic_swap(self):
        # Single-layer cache: 4 blocks of 16 elements each.
        num_blocks, elem_per_block = 4, 16
        src = torch.arange(num_blocks * elem_per_block,
                           dtype=torch.float32, device="cuda")
        # Copy contents from src to dst verbatim, then swap one pair.
        dst = src.clone()
        block_size_bytes = elem_per_block * src.element_size()

        # Swap block 0 and block 3.
        block_mapping = torch.tensor(
            [[0, 3], [3, 0]], dtype=torch.int64, device="cpu",
        )
        _C_CACHE_OPS.swap_blocks(src, dst, block_size_bytes, block_mapping)

        # After swap, dst[0:16] should hold what was in src[48:64], and vice
        # versa.  Note: cudaMemcpyAsync is asynchronous; we need sync.
        torch.cuda.synchronize()
        assert torch.allclose(dst[0:16], src[48:64])
        assert torch.allclose(dst[48:64], src[0:16])

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_block_mapping_must_be_cpu(self):
        src = torch.zeros(16, dtype=torch.float32, device="cuda")
        dst = torch.zeros_like(src)
        block_mapping = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")

        with pytest.raises(RuntimeError, match="block_mapping must be on CPU|block_mapping must live on CPU"):
            _C_CACHE_OPS.swap_blocks(src, dst, 16 * src.element_size(), block_mapping)


# ---------------------------------------------------------------------------
# copy_blocks — GPU-side block-to-block KV copy
# ---------------------------------------------------------------------------

class TestCopyBlocks:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_basic_copy(self, dtype):
        num_layers, num_blocks, num_heads, block_size, head_size = 2, 4, 1, 4, 4

        k_caches = [
            torch.randn(num_blocks, num_heads, block_size, head_size,
                        dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        v_caches = [
            torch.randn(num_blocks, num_heads, block_size, head_size,
                        dtype=dtype, device="cuda")
            for _ in range(num_layers)
        ]
        # Copy block 0 → block 2, block 1 → block 3 (all layers).
        block_mapping = torch.tensor(
            [[0, 2], [1, 3]], dtype=torch.int64, device="cpu",
        )
        _C_CACHE_OPS.copy_blocks(k_caches, v_caches, block_mapping)
        torch.cuda.synchronize()

        for layer_idx in range(num_layers):
            k_src_0 = k_caches[layer_idx][0].clone()
            v_src_0 = v_caches[layer_idx][0].clone()
            # After copy, dst block 2 should equal src block 0.
            assert torch.allclose(k_caches[layer_idx][2], k_src_0, atol=1e-3)
            assert torch.allclose(v_caches[layer_idx][2], v_src_0, atol=1e-3)


# ---------------------------------------------------------------------------
# cp_gather_cache — gather non-contiguous blocks → contiguous output
# ---------------------------------------------------------------------------

class TestCpGatherCache:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_two_requests(self, dtype):
        num_blocks, block_size, entry_size = 8, 4, 6
        # Cache layout: [num_blocks, block_size, entry_size]
        src_cache = torch.randn(num_blocks, block_size, entry_size,
                                dtype=dtype, device="cuda")

        # Two requests: req0 uses blocks [3, 1], req1 uses block [5].
        block_table = torch.tensor(
            [[3, 1, -1], [5, -1, -1]], dtype=torch.int32, device="cuda",
        )
        cu_seq_lens = torch.tensor([0, 7, 9], dtype=torch.int32, device="cuda")
        batch_size = 2

        total_tokens = cu_seq_lens[-1].item()
        dst = torch.zeros(total_tokens, entry_size,
                          dtype=dtype, device="cuda")

        _C_CACHE_OPS.cp_gather_cache(
            src_cache, dst, block_table, cu_seq_lens, batch_size,
        )
        torch.cuda.synchronize()

        # Verify by manual gather.
        for req_idx in range(batch_size):
            start = cu_seq_lens[req_idx].item()
            end = cu_seq_lens[req_idx + 1].item()
            slen = end - start
            for off in range(slen):
                blk_idx = off // block_size
                off_in_blk = off % block_size
                phys_block = block_table[req_idx, blk_idx].item()
                expected = src_cache[phys_block, off_in_blk, :]
                assert torch.allclose(dst[start + off], expected, atol=1e-3), (
                    f"req{req_idx} token{off} mismatch"
                )

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_empty_batch(self):
        dtype = torch.float16
        src_cache = torch.zeros(2, 4, 4, dtype=dtype, device="cuda")
        dst = torch.zeros(0, 4, dtype=dtype, device="cuda")
        block_table = torch.zeros(0, 0, dtype=torch.int32, device="cuda")
        cu_seq_lens = torch.tensor([0], dtype=torch.int32, device="cuda")
        # batch_size=0 → no-op.
        _C_CACHE_OPS.cp_gather_cache(
            src_cache, dst, block_table, cu_seq_lens, 0,
        )


# ---------------------------------------------------------------------------
# convert_fp8 — per-block FP8 ↔ BF16/FP16 conversion
# ---------------------------------------------------------------------------

class TestConvertFP8:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_bf16_to_fp8(self):
        # Build a 2-block BF16 cache (16 elems per block) and convert to FP8 e4m3.
        block_stride = 16
        src = torch.tensor([
            0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0,
            0.25, 0.75, -0.5, 0.5, 4.0, -4.0, 64.0, -64.0,
            0.0, 0.5, 1.0, -1.0, 2.0, -2.0, 100.0, -100.0,
            0.25, 0.75, -0.5, 0.5, 4.0, -4.0, 64.0, -64.0,
        ], dtype=torch.bfloat16, device="cuda").reshape(2, block_stride)
        dst = torch.zeros(2, block_stride, dtype=torch.uint8, device="cuda")

        # scale=1.0 means: clamp then direct encode (no rescale).
        _C_CACHE_OPS.convert_fp8(dst, src, 1.0, "fp8_e4m3")
        torch.cuda.synchronize()

        # Spot-check: value 0.0 → 0x00, value 1.0 → ~0x38 (FP8 e4m3 1.0).
        # Note: BF16→F32→FP8 conversion goes via float; 1.0 encodes exactly.
        assert dst[0, 0].item() == 0x00 or dst[0, 0].item() == 0x80  # ±0
        # 1.0 in e4m3 = 0b00111000 = 0x38 (sign=0, exp=7-7=0? actually
        # bias=7: 1.0 = 2^0 → exp=7 → (7<<3)|0 = 0x38).
        # Allow ±1 LSB rounding slack.
        assert abs(dst[0, 2].item() - 0x38) <= 1
        # -1.0 → 0xB8.
        assert abs((dst[0, 3].item() & 0x7F) - 0x38) <= 1
        assert (dst[0, 3].item() & 0x80) != 0  # sign bit set

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_fp8_to_bf16(self):
        block_stride = 8
        # 0x38 = 1.0 in FP8 e4m3; 0x00 = 0.0; 0xC0 = -2.0 (sign=1, exp=8, mant=0).
        src = torch.tensor(
            [0x38, 0x00, 0xC0, 0x38, 0x38, 0x00, 0xC0, 0x38],
            dtype=torch.uint8, device="cuda",
        ).reshape(1, block_stride)
        dst = torch.zeros(1, block_stride, dtype=torch.bfloat16, device="cuda")

        _C_CACHE_OPS.convert_fp8(dst, src, 1.0, "auto")
        torch.cuda.synchronize()

        # 1.0 round-trips (BF16 represents 1.0 exactly).
        assert abs(dst[0, 0].item() - 1.0) < 1e-3
        assert dst[0, 1].item() == 0.0
        # -2.0 round-trips (within BF16 precision).
        assert abs(dst[0, 2].item() - (-2.0)) < 1e-2

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_unsupported_dtype_rejected(self):
        src = torch.zeros(1, 8, dtype=torch.float64, device="cuda")
        dst = torch.zeros(1, 8, dtype=torch.uint8, device="cuda")
        with pytest.raises(RuntimeError, match="unsupported src dtype"):
            _C_CACHE_OPS.convert_fp8(dst, src, 1.0, "fp8_e4m3")


# ---------------------------------------------------------------------------
# concat_and_cache_mla — MLA compressed KV write (fp16/bf16 only)
# ---------------------------------------------------------------------------

class TestConcatAndCacheMLA:
    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_basic_concat(self, dtype):
        num_tokens, kv_lora_rank, pe_dim = 3, 4, 4
        block_size = 4
        num_blocks = 2

        kv_c = torch.randn(num_tokens, kv_lora_rank,
                           dtype=dtype, device="cuda")
        k_pe = torch.randn(num_tokens, pe_dim,
                           dtype=dtype, device="cuda")
        # MLA cache: [num_blocks, block_size, kv_lora_rank + pe_dim]
        kv_cache = torch.zeros(num_blocks, block_size, kv_lora_rank + pe_dim,
                               dtype=dtype, device="cuda")
        slot_mapping = torch.tensor([0, 1, 4], dtype=torch.int64, device="cuda")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        _C_CACHE_OPS.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, "auto", scale,
        )
        torch.cuda.synchronize()

        for i in range(num_tokens):
            blk = slot_mapping[i].item() // block_size
            off = slot_mapping[i].item() % block_size
            assert torch.allclose(
                kv_cache[blk, off, :kv_lora_rank], kv_c[i], atol=1e-3,
            ), f"kv_c mismatch at token {i}"
            assert torch.allclose(
                kv_cache[blk, off, kv_lora_rank:], k_pe[i], atol=1e-3,
            ), f"k_pe mismatch at token {i}"

    @pytest.mark.skipif(_C_CACHE_OPS is None, reason="需要 my_vllm._C")
    def test_fp8_mla_rejected(self):
        dtype = torch.float16
        kv_c = torch.zeros(1, 4, dtype=dtype, device="cuda")
        k_pe = torch.zeros(1, 4, dtype=dtype, device="cuda")
        kv_cache = torch.zeros(1, 4, 8, dtype=dtype, device="cuda")
        slot_mapping = torch.tensor([0], dtype=torch.int64, device="cuda")
        scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        with pytest.raises(RuntimeError, match="only kv_cache_dtype='auto'"):
            _C_CACHE_OPS.concat_and_cache_mla(
                kv_c, k_pe, kv_cache, slot_mapping, "fp8_ds_mla", scale,
            )

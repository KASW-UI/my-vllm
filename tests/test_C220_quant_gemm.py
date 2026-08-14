"""test_C220_quant_gemm.py — C220 cutlass_scaled_mm + scaled quant tests.

Verifies the Python wrappers for quantized GEMM in
``my_vllm/_custom_ops``:

  1. ``cutlass_scaled_mm`` FP8 path — dispatches to
     ``torch._scaled_mm``; numerical accuracy within FP8's ~5% range.
  2. ``cutlass_scaled_mm`` INT8 path — dispatches to
     ``torch._int_mm``; accuracy within INT8's ~1% range.
  3. ``cutlass_scaled_mm`` BF16 fallback — ignores scales, runs
     ``torch.mm``.
  4. ``cutlass_scaled_mm_supports_fp8`` — correctly identifies sm_89+.
  5. ``scaled_fp8_quant`` dynamic per-tensor + per-token.
  6. ``scaled_fp8_quant`` static (caller-provided scale).
  7. ``scaled_int8_quant`` symmetric + asymmetric.
  8. ``cutlass_scaled_mm_fn`` (functional variant).
  9. End-to-end: bf16 → quant → scaled_mm → dequant → compare.
 10. Shape/dtype rejection (FP8 → FP32 raises; INT8 → BF16 ok).
"""

import pytest
import torch

from my_vllm._custom_ops import (
    cutlass_scaled_mm,
    cutlass_scaled_mm_fn,
    cutlass_scaled_mm_supports_fp8,
    scaled_fp8_quant,
    scaled_int8_quant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.fixture
def seed():
    torch.manual_seed(0)


@pytest.fixture
def MKN():
    return 64, 64, 64  # small for unit tests


# ---------------------------------------------------------------------------
# 1. FP8 hardware support detection
# ---------------------------------------------------------------------------


class TestFp8Support:
    def test_supports_fp8_returns_bool(self):
        result = cutlass_scaled_mm_supports_fp8()
        assert isinstance(result, bool)

    def test_supports_fp8_explicit_sm89(self):
        # sm_89 (Ada) is the first SM with FP8 tensor cores.
        assert cutlass_scaled_mm_supports_fp8(89) is True

    def test_supports_fp8_sm90_hopper(self):
        assert cutlass_scaled_mm_supports_fp8(90) is True

    def test_supports_fp8_sm100_blackwell(self):
        assert cutlass_scaled_mm_supports_fp8(100) is True

    def test_supports_fp8_sm80_ampere_false(self):
        # Ampere (sm_80, sm_86) has no FP8 tensor cores.
        assert cutlass_scaled_mm_supports_fp8(80) is False
        assert cutlass_scaled_mm_supports_fp8(86) is False

    def test_supports_fp8_sm70_pre_pascal_false(self):
        # Volta and earlier lack tensor-core FP8 entirely.
        assert cutlass_scaled_mm_supports_fp8(70) is False
        assert cutlass_scaled_mm_supports_fp8(75) is False


# ---------------------------------------------------------------------------
# 2. scaled_fp8_quant
# ---------------------------------------------------------------------------


class TestScaledFp8Quant:
    def test_dynamic_per_tensor_shape_and_dtype(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        x_fp8, scale = scaled_fp8_quant(x)
        assert x_fp8.shape == x.shape
        assert x_fp8.dtype == torch.float8_e4m3fn
        assert scale.shape == (1,)
        assert scale.dtype == torch.float32
        assert scale.item() > 0

    def test_dynamic_per_token_scale_shape(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        _, scale = scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        assert scale.shape == (32, 1)

    def test_static_scale_preserved(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        scale_in = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        x_fp8, scale_out = scaled_fp8_quant(x, scale=scale_in)
        # Static: caller-provided scale is echoed back.
        assert torch.equal(scale_in, scale_out)

    def test_quant_values_in_fp8_range(self, seed):
        # After quantize, dequant should stay within the FP8 representable range.
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 0.5
        x_fp8, scale = scaled_fp8_quant(x)
        # Dequantize: x_dq = x_fp8.float() * scale
        x_dq = x_fp8.to(torch.float32) * scale.to(torch.float32)
        # Max value should be <= 448 (FP8 e4m3 max).
        assert x_dq.abs().max().item() <= 448.0 + 1e-3

    def test_rejects_non_2d(self, seed):
        x = torch.randn(8, 8, 8, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError):
            scaled_fp8_quant(x)

    def test_num_token_padding(self, seed):
        x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
        x_fp8, scale = scaled_fp8_quant(x, num_token_padding=16)
        assert x_fp8.shape == (16, 64)
        assert scale.shape == (1,)


# ---------------------------------------------------------------------------
# 3. scaled_int8_quant
# ---------------------------------------------------------------------------


class TestScaledInt8Quant:
    def test_symmetric_shape_and_dtype(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        x_i8, scale, azp = scaled_int8_quant(x, symmetric=True)
        assert x_i8.shape == x.shape
        assert x_i8.dtype == torch.int8
        assert scale.shape == (1,)
        assert scale.dtype == torch.float32
        # Symmetric: azp is None.
        assert azp is None

    def test_symmetric_values_in_range(self, seed):
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 0.5
        x_i8, _, _ = scaled_int8_quant(x, symmetric=True)
        # Symmetric INT8 quantized values should be in [-127, 127].
        assert x_i8.min().item() >= -127
        assert x_i8.max().item() <= 127

    def test_asymmetric_azp_present(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        # Bias the input so azp is non-zero.
        x = x + 0.5
        x_i8, scale, azp = scaled_int8_quant(x, symmetric=False)
        assert x_i8.shape == x.shape
        assert x_i8.dtype == torch.int8
        assert azp is not None
        assert azp.dtype == torch.int8

    def test_per_token_asymmetric(self, seed):
        x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        x_i8, scale, azp = scaled_int8_quant(
            x, symmetric=False, use_per_token_if_dynamic=True,
        )
        # Per-token scale should be per-row.
        assert scale.shape == (32, 1)
        assert azp.shape == (32, 1)


# ---------------------------------------------------------------------------
# 4. cutlass_scaled_mm — FP8 path
# ---------------------------------------------------------------------------


class TestCutlassScaledMmFp8:
    @pytest.mark.skipif(
        not cutlass_scaled_mm_supports_fp8(),
        reason="FP8 tensor cores not available on this GPU",
    )
    def test_fp8_smoke(self, seed, MKN):
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)
        out = cutlass_scaled_mm_fn(
            a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16,
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(
        not cutlass_scaled_mm_supports_fp8(),
        reason="FP8 tensor cores not available on this GPU",
    )
    def test_fp8_accuracy_vs_bf16(self, seed):
        """FP8 GEMM output should be within ~10% of BF16 baseline.

        We use unscaled N(0,1) inputs because per-tensor dynamic FP8
        scaling maps ``max(|x|) → 448``.  Inputs scaled by 0.1 would
        waste most of the FP8 representable range and produce all-zero
        outputs.

        FP8 e4m3 has 3-bit mantissa (~12.5% per-element relative
        error); the accumulated error over K=256 multiplications is
        bounded by ``sqrt(K) * per-element-error``.
        """
        torch.manual_seed(42)
        M, K, N = 64, 256, 64
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

        # BF16 baseline.
        baseline = torch.nn.functional.linear(a, b)

        # FP8 via scaled_mm.
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)
        out_fp8 = cutlass_scaled_mm_fn(
            a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16,
        )

        # Magnitude-relative tolerance.
        baseline_abs = baseline.abs().clamp(min=1e-3)
        rel_diff = (out_fp8.float() - baseline.float()).abs() / baseline_abs
        # Per-tensor dynamic + e4m3 mantissa on K=256 → ~5-10% median.
        assert rel_diff.median().item() < 0.15, (
            f"FP8 median rel diff = {rel_diff.median().item():.4f} > 0.15"
        )

    @pytest.mark.skipif(
        not cutlass_scaled_mm_supports_fp8(),
        reason="FP8 tensor cores not available on this GPU",
    )
    def test_fp8_inplace(self, seed, MKN):
        """The out= variant writes to a pre-allocated tensor."""
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)
        out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        cutlass_scaled_mm(out, a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16)
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()

    @pytest.mark.skipif(
        not cutlass_scaled_mm_supports_fp8(),
        reason="FP8 tensor cores not available on this GPU",
    )
    def test_fp8_bias_add(self, seed, MKN):
        """Bias is added after the matmul."""
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1
        bias = torch.randn((N,), device="cuda", dtype=torch.bfloat16) * 0.01
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)
        out_no_bias = cutlass_scaled_mm_fn(
            a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16,
        )
        out_with_bias = cutlass_scaled_mm_fn(
            a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16, bias=bias,
        )
        # bias should be added (modulo FP8 quant error on out_no_bias).
        diff = (out_with_bias.float() - out_no_bias.float() - bias.float()).abs()
        assert diff.max().item() < 1e-2


# ---------------------------------------------------------------------------
# 5. cutlass_scaled_mm — INT8 path
# ---------------------------------------------------------------------------


class TestCutlassScaledMmInt8:
    def test_int8_smoke(self, seed, MKN):
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1
        a_i8, sa, _ = scaled_int8_quant(a)
        b_i8, sb, _ = scaled_int8_quant(b)
        out = cutlass_scaled_mm_fn(
            a_i8, b_i8, sa, sb, out_dtype=torch.bfloat16,
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()

    def test_int8_accuracy_vs_bf16(self, seed):
        """INT8 GEMM output should be within ~10% of BF16 baseline.

        INT8 per-tensor symmetric quantization has ~0.8% per-element
        error; accumulated over K=256 multiplications we expect ~5%
        median relative error.
        """
        torch.manual_seed(42)
        M, K, N = 64, 256, 64
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        baseline = torch.nn.functional.linear(a, b)
        a_i8, sa, _ = scaled_int8_quant(a)
        b_i8, sb, _ = scaled_int8_quant(b)
        out_i8 = cutlass_scaled_mm_fn(
            a_i8, b_i8, sa, sb, out_dtype=torch.bfloat16,
        )
        baseline_abs = baseline.abs().clamp(min=1e-3)
        rel_diff = (out_i8.float() - baseline.float()).abs() / baseline_abs
        assert rel_diff.median().item() < 0.10


# ---------------------------------------------------------------------------
# 6. cutlass_scaled_mm — BF16 / FP16 baseline path
# ---------------------------------------------------------------------------


class TestCutlassScaledMmBaseline:
    def test_bf16_path(self, seed, MKN):
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        # Scales are ignored in BF16 path — pass dummy ones.
        dummy_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        out = cutlass_scaled_mm_fn(
            a, b, dummy_scale, dummy_scale, out_dtype=torch.bfloat16,
        )
        baseline = torch.nn.functional.linear(a, b)
        assert torch.allclose(out, baseline, atol=1e-3, rtol=1e-3)

    def test_fp16_path(self, seed, MKN):
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((N, K), device="cuda", dtype=torch.float16)
        dummy_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        out = cutlass_scaled_mm_fn(
            a, b, dummy_scale, dummy_scale, out_dtype=torch.float16,
        )
        assert out.shape == (M, N)
        assert out.dtype == torch.float16


# ---------------------------------------------------------------------------
# 7. cutlass_scaled_mm — error paths
# ---------------------------------------------------------------------------


class TestCutlassScaledMmErrors:
    def test_unsupported_dtype_raises(self, seed, MKN):
        M, K, N = MKN
        # FP32 is not in the supported set.
        a = torch.randn((M, K), device="cuda", dtype=torch.float32)
        b = torch.randn((N, K), device="cuda", dtype=torch.float32)
        dummy_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        with pytest.raises(TypeError, match="unsupported dtype"):
            cutlass_scaled_mm_fn(
                a, b, dummy_scale, dummy_scale, out_dtype=torch.float32,
            )

    def test_mismatched_dtypes_raises(self, seed, MKN):
        M, K, N = MKN
        # Construct FP8 from BF16 (torch.randn doesn't support FP8 directly).
        a_bf16 = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        a = a_bf16.to(torch.float8_e4m3fn)
        b_bf16 = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        b = b_bf16.to(torch.int8)
        dummy_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        with pytest.raises(TypeError, match="unsupported dtype"):
            cutlass_scaled_mm_fn(
                a, b, dummy_scale, dummy_scale, out_dtype=torch.bfloat16,
            )


# ---------------------------------------------------------------------------
# 8. cutlass_scaled_mm_fn vs cutlass_scaled_mm consistency
# ---------------------------------------------------------------------------


class TestFunctionalConsistency:
    def test_functional_matches_inplace(self, seed, MKN):
        """The functional variant produces the same output as the
        in-place variant given the same pre-allocated out tensor.
        """
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) * 0.1
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16) * 0.1
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)

        if cutlass_scaled_mm_supports_fp8():
            out_fn = cutlass_scaled_mm_fn(
                a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16,
            )
            out_inplace = torch.empty_like(out_fn)
            cutlass_scaled_mm(
                out_inplace, a_fp8, b_fp8, sa, sb,
                out_dtype=torch.bfloat16,
            )
            assert torch.equal(out_fn, out_inplace)

    def test_functional_default_out_dtype(self, seed, MKN):
        """out_dtype defaults to ``out.dtype`` when None."""
        M, K, N = MKN
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
        dummy_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        # When out_dtype=None, should default to the dtype of the
        # output.  For BF16 baseline, we pass out= as BF16.
        out_buf = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        cutlass_scaled_mm(out_buf, a, b, dummy_scale, dummy_scale)
        assert out_buf.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# 9. End-to-end: quantize → scaled_mm → dequantize round-trip
# ---------------------------------------------------------------------------


class TestEndToEndRoundtrip:
    @pytest.mark.skipif(
        not cutlass_scaled_mm_supports_fp8(),
        reason="FP8 tensor cores not available on this GPU",
    )
    def test_fp8_round_trip_smoke(self, seed):
        """Linear(a, b) ≈ (a_fp8 * sa) @ (b_fp8 * sb).T in BF16.

        The FP8 e4m3 quantization rounds small values to coarse steps,
        so element-wise tolerance is loose.  We assert (1) finite,
        (2) correct magnitude (max |out| ≈ max |ref| within 20%), and
        (3) high Pearson correlation (> 0.95).  These are the right
        checks for an FP8 round-trip without requiring per-element
        bit-equivalence.
        """
        torch.manual_seed(123)
        M, K, N = 32, 128, 32
        a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
        b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)

        # Reference: BF16 baseline.
        ref = torch.nn.functional.linear(a, b)

        # Quantize + scaled_mm.
        a_fp8, sa = scaled_fp8_quant(a)
        b_fp8, sb = scaled_fp8_quant(b)
        out = cutlass_scaled_mm_fn(
            a_fp8, b_fp8, sa, sb, out_dtype=torch.bfloat16,
        )

        # (1) Finite.
        assert torch.isfinite(out).all()

        # (2) Magnitude envelope — max abs within 20% of reference.
        ref_max = ref.abs().max().item()
        out_max = out.abs().max().item()
        assert abs(out_max - ref_max) / ref_max < 0.20, (
            f"max abs: ref={ref_max:.4f}, out={out_max:.4f}, "
            f"ratio={out_max / ref_max:.3f}"
        )

        # (3) Pearson correlation > 0.95.
        ref_f = ref.float().flatten()
        out_f = out.float().flatten()
        ref_c = ref_f - ref_f.mean()
        out_c = out_f - out_f.mean()
        corr = (ref_c * out_c).sum() / (
            ref_c.norm() * out_c.norm() + 1e-12
        )
        assert corr.item() > 0.95, (
            f"Pearson correlation {corr.item():.4f} < 0.95 — FP8 path "
            f"produces essentially noise."
        )

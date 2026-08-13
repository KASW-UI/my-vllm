"""FP8 utility helpers — pure functions + Triton quant kernels.

Mirrors vLLM V1 model_executor/layers/quantization/utils/fp8_utils.py.

C140: minimal subset — the pure helpers (`is_fp8`,
`get_fp8_min_max`, `per_tensor_quantize`, `per_tensor_dequantize`)
and one Triton per-token quant kernel
(`per_token_quant_fp8`). The vLLM file is 1454 lines covering
DeepGemm integration, block-wise quantization,
`per_token_group_quant_fp8`, `w8a8_utils`, and many more
helpers — most are out of scope for my-vllm's single-KV-cache-
quant path. Adding more is a follow-up once the FP8 path is
actually exercised end-to-end.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ── FP8 dtype checks ──────────────────────────────────────────────────


def is_fp8(dtype: torch.dtype | torch.Tensor) -> bool:
    """Return True if `dtype` is an FP8 variant.

    Mirrors vLLM V1 .../utils/fp8_utils.py:is_fp8 (L42).
    Supports both `float8_e4m3fn` (NVIDIA) and `float8_e4m3fnuz`
    (AMD) variants.
    """
    if isinstance(dtype, torch.Tensor):
        dtype = dtype.dtype
    return dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def get_fp8_min_max(dtype: torch.dtype = torch.float8_e4m3fn) -> tuple[float, float]:
    """Return `(min, max)` representable in the given FP8 dtype.

    Mirrors vLLM V1 .../utils/quant_utils.py:get_fp8_min_max (L41).
    `float8_e4m3fn` and `float8_e4m3fnuz` share the same range
    for our purposes (the fnuz variant differs by ±NaN
    encoding, not the representable range).
    """
    # E4M3: 4 exponent bits, 3 mantissa bits + 1 implicit.
    # Max = 1.111₂ × 2^(7-biased=8) = 448.0
    # Min (positive) = 2^-9 ≈ 0.001953125
    return (0.001953125, 448.0)


# ── Per-tensor quantize / dequantize ──────────────────────────────────


def per_tensor_quantize(
    x: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> torch.Tensor:
    """Quantize a tensor to FP8 with a single per-tensor scale.

    Args:
        x: input tensor (any float dtype).
        scale: scalar scale factor.
        dtype: target FP8 dtype (default `float8_e4m3fn`).

    Returns:
        FP8-quantized tensor with the same shape as `x`.
    """
    fp8_min, fp8_max = get_fp8_min_max(dtype)
    scaled = x.float() / scale.float()
    clamped = scaled.clamp(min=fp8_min, max=fp8_max)
    return clamped.to(dtype)


def per_tensor_dequantize(
    x_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an FP8 tensor with a single per-tensor scale."""
    return x_fp8.to(out_dtype) * scale.to(out_dtype)


# ── Triton per-token quant kernel ─────────────────────────────────────


@triton.jit
def _per_token_quant_fp8_kernel(
    x_ptr,           # [numel] input bf16/fp16
    out_ptr,         # [numel] output fp8
    scale_ptr,       # [num_tokens] per-token scale (output)
    absmax_ptr,      # [num_tokens] per-token absmax (output)
    numel_per_token,
    fp8_min,
    fp8_max,
):
    """Per-token FP8 quant: `out = (x / absmax).clamp(fp8_min, fp8_max)`.

    One program per token. Computes the per-token absmax, then
    quantizes each element. The scale is `absmax / fp8_max` so
    that dequant `out.to(fp32) * scale` recovers the original.
    """
    token_idx = tl.program_id(0)
    base = token_idx * numel_per_token

    # Pass 1: compute absmax.
    absmax = tl.zeros((), dtype=tl.float32)
    for i in tl.static_range(numel_per_token):
        v = tl.load(x_ptr + base + i).to(tl.float32)
        absmax = tl.maximum(absmax, tl.abs(v))
    tl.store(absmax_ptr + token_idx, absmax)

    # Compute scale = absmax / fp8_max (or 1.0 if absmax == 0).
    scale = tl.where(absmax > 0.0, absmax / fp8_max, 1.0)
    tl.store(scale_ptr + token_idx, scale)

    # Pass 2: quantize each element.
    inv_scale = 1.0 / scale
    for i in tl.static_range(numel_per_token):
        v = tl.load(x_ptr + base + i).to(tl.float32) * inv_scale
        v = tl.minimum(tl.maximum(v, fp8_min), fp8_max)
        tl.store(out_ptr + base + i, v.to(out_ptr.dtype.element_ty))


def per_token_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `x` to FP8 with one scale per token (row).

    Args:
        x: input tensor of shape `[num_tokens, numel_per_token]`,
            dtype `bfloat16` / `float16`.
        dtype: target FP8 dtype.

    Returns:
        `(x_fp8, scales)` where:
          - `x_fp8` is FP8-quantized with shape `[num_tokens, numel_per_token]`.
          - `scales` is `float32` of shape `[num_tokens]`, with
            `scales[t] = absmax(x[t]) / fp8_max`.
    """
    assert x.ndim == 2, f"per_token_quant_fp8 expects 2-D input, got {x.ndim}-D"
    num_tokens, numel_per_token = x.shape
    fp8_min, fp8_max = get_fp8_min_max(dtype)

    x_fp8 = torch.empty_like(x, dtype=dtype)
    scales = torch.empty(num_tokens, dtype=torch.float32, device=x.device)
    absmax = torch.empty(num_tokens, dtype=torch.float32, device=x.device)

    _per_token_quant_fp8_kernel[(num_tokens,)](
        x,
        x_fp8,
        scales,
        absmax,
        numel_per_token=numel_per_token,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
    )
    return x_fp8, scales


__all__ = [
    "get_fp8_min_max",
    "is_fp8",
    "per_tensor_dequantize",
    "per_tensor_quantize",
    "per_token_quant_fp8",
]
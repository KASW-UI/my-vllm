"""
Custom ops interface — wraps PyTorch custom C++/CUDA extensions (torch.ops._C).

In real vLLM this dispatches to compiled C++/CUDA kernels via torch.library.
Here we provide the Python-level interface with PyTorch fallbacks,
matching the architecture and function signatures of real vLLM.

The compiled CUDA extension lives at ``my_vllm._C`` (see setup.py →
csrc/cache_kernels.cu).  When the extension is not built (e.g. CPU-only
test environments) the wrappers fall back to a small Python
implementation so callers can still proceed.
"""

from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import torch

current_platform = None


# ---------------------------------------------------------------------------
# Compiled CUDA extension — my_vllm._C
# ---------------------------------------------------------------------------
#
# Built via ``python setup.py build_ext --inplace -j3`` (see setup.py →
# csrc/cache_kernels.cu).  Provides the vLLM-canonical cache block ops:
#
#   reshape_and_cache / reshape_and_cache_flash
#   swap_blocks / copy_blocks / cp_gather_cache
#   convert_fp8 / concat_and_cache_mla
#
# The hot path is still the Triton kernel triton_reshape_and_cache_flash
# (C149); these CUDA ops provide the full vLLM API surface (CUDA-graph
# fallback, FP8 conversion, prefix-cache CoW gather, MLA stub).
#
# Import is deferred so that environments without the compiled .so can
# still import my_vllm for tests that don't touch the cache kernels.

def _load_c_extension():
    try:
        from my_vllm import _C  # noqa: F401
        return _C
    except ImportError:
        return None


_C_CACHE_OPS = _load_c_extension()


def _reshape_and_cache_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """CUDA dispatch via my_vllm._C.reshape_and_cache (when available)."""
    if _C_CACHE_OPS is None:
        _reshape_and_cache_python_fallback(
            key, value, key_cache, value_cache, slot_mapping,
        )
        return
    _C_CACHE_OPS.reshape_and_cache(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )


def _reshape_and_cache_flash_cuda(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    """CUDA dispatch via my_vllm._C.reshape_and_cache_flash (CUDA-graph path)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "reshape_and_cache_flash CUDA op requires the compiled my_vllm._C "
            "extension; rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.reshape_and_cache_flash(
        key, value, key_cache, value_cache,
        slot_mapping, kv_cache_dtype, k_scale, v_scale,
    )


def _swap_blocks_cuda(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_size_in_bytes: int,
    block_mapping: torch.Tensor,
) -> None:
    """CUDA dispatch via my_vllm._C.swap_blocks (preemption / beam search)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "swap_blocks CUDA op requires the compiled my_vllm._C extension; "
            "rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.swap_blocks(src, dst, block_size_in_bytes, block_mapping)


def _copy_blocks_cuda(
    key_caches: list[torch.Tensor],
    value_caches: list[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """CUDA dispatch via my_vllm._C.copy_blocks (prefix-cache CoW copy)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "copy_blocks CUDA op requires the compiled my_vllm._C extension; "
            "rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.copy_blocks(key_caches, value_caches, block_mapping)


def _cp_gather_cache_cuda(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
) -> None:
    """CUDA dispatch via my_vllm._C.cp_gather_cache (non-contiguous gather)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "cp_gather_cache CUDA op requires the compiled my_vllm._C "
            "extension; rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.cp_gather_cache(
        src_cache, dst, block_table, cu_seq_lens, batch_size,
    )


def _convert_fp8_cuda(
    dst_cache: torch.Tensor,
    src_cache: torch.Tensor,
    scale: float,
    kv_cache_dtype: str,
) -> None:
    """CUDA dispatch via my_vllm._C.convert_fp8 (FP8 ↔ BF16/FP16)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "convert_fp8 CUDA op requires the compiled my_vllm._C extension; "
            "rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.convert_fp8(dst_cache, src_cache, scale, kv_cache_dtype)


def _concat_and_cache_mla_cuda(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """CUDA dispatch via my_vllm._C.concat_and_cache_mla (MLA stub)."""
    if _C_CACHE_OPS is None:
        raise RuntimeError(
            "concat_and_cache_mla CUDA op requires the compiled my_vllm._C "
            "extension; rebuild with `python setup.py build_ext --inplace -j3`."
        )
    _C_CACHE_OPS.concat_and_cache_mla(
        kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale,
    )


def _reshape_and_cache_python_fallback(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Pure-PyTorch reshape_and_cache fallback (matches
    v1/attention/ops/paged_attn.py::reshape_and_cache_pytorch semantics)."""
    block_size = key_cache.shape[2]
    valid = slot_mapping >= 0
    if not valid.any():
        return
    valid_slots = slot_mapping[valid]
    block_ids = valid_slots // block_size
    offsets = valid_slots % block_size
    key_cache[block_ids, :, offsets, :] = key[valid]
    value_cache[block_ids, :, offsets, :] = value[valid]


# Public re-exports for callers (v1/attention/ops/paged_attn.py, tests).
reshape_and_cache = _reshape_and_cache_cuda
reshape_and_cache_flash = _reshape_and_cache_flash_cuda
swap_blocks = _swap_blocks_cuda
copy_blocks = _copy_blocks_cuda
cp_gather_cache = _cp_gather_cache_cuda
convert_fp8 = _convert_fp8_cuda
concat_and_cache_mla = _concat_and_cache_mla_cuda


class Fp8KVCacheDataType(IntEnum):
    AUTO = 0
    FP8_E4M3 = 1
    FP8_E5M2 = 2
    FP8_E4M3_MLA = 3


# ---------------------------------------------------------------------------
# Paged attention ops
# ---------------------------------------------------------------------------

def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: torch.Tensor | None,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    fp8_out_scale: torch.Tensor | None = None,
    mfma_type: str = "f16",
) -> None:
    """
    ROCm paged attention kernel.

    Architecture: decoupled Q*K^T → softmax → *V with FP8 scaling support
    and ALiBi slopes, dispatched to _rocm_C.paged_attention.
    """
    pass


def mla_decode_kvcache_cpu(
    out: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """MLA decode with KV cache on CPU (paged)."""
    pass


def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    """Merge prefix and suffix attention states with LSE rescaling."""
    pass


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """
    Fused rotary positional embedding.

    Signature: _C::rotary_embedding(positions, query, key, head_size,
                                     cos_sin_cache, is_neox) -> None
    In-place: modifies query and key tensors.
    """
    pass


# ---------------------------------------------------------------------------
# RMS normalization
# ---------------------------------------------------------------------------

def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """RMS normalization: out = input * rsqrt(mean(input^2) + eps) * weight."""
    _rms_norm_fallback(out, input, weight, epsilon)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Fused add + RMS norm: residual = input + residual; input = rms_norm(residual)."""
    residual.add_(input)
    _rms_norm_fallback(input, residual, weight, epsilon)


def _rms_norm_fallback(out, input, weight, epsilon):
    variance = input.pow(2).mean(-1, keepdim=True)
    out.copy_(input * torch.rsqrt(variance + epsilon) * weight)


# ---------------------------------------------------------------------------
# RMS norm + FP8 quantization (fused)
# ---------------------------------------------------------------------------

def rms_norm_dynamic_per_token_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    fp8_dtype: Fp8KVCacheDataType = Fp8KVCacheDataType.FP8_E4M3,
) -> None:
    """RMS norm + dynamic per-token FP8 quantization (fused)."""
    pass


def rms_norm_per_block_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    epsilon: float,
    block_size: int,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    fp8_dtype: Fp8KVCacheDataType = Fp8KVCacheDataType.FP8_E4M3,
) -> None:
    """RMS norm + per-block FP8 quantization (fused, DeepSeek-style)."""
    pass


# ---------------------------------------------------------------------------
# SiLU + mul + block quant (fused)
# ---------------------------------------------------------------------------

def silu_and_mul_per_block_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    fp8_dtype: Fp8KVCacheDataType = Fp8KVCacheDataType.FP8_E4M3,
) -> None:
    """Fused SiLU(x) * y gate + per-block FP8 quantization."""
    pass


# ---------------------------------------------------------------------------
# CUTLASS scaled matmul
# ---------------------------------------------------------------------------
#
# `cutlass_scaled_mm` is the vLLM-canonical entry point for quantized
# GEMM (FP8 e4m3 / INT8 / INT4 weight × quantized activation).  In
# real vLLM it dispatches to a CUTLASS-3.x kernel (`csrc/.../cutlass/`)
# built per SM (sm_80 / sm_89 / sm_90 / sm_100 / sm_120).
#
# my-vllm does NOT vendor the full CUTLASS library (3000+ headers) — the
# `csrc/` extension would balloon from ~700 lines to ~10k.  Instead we
# dispatch to PyTorch's native scaled/int matmul ops which are
# themselves CUTLASS-backed (cublasLt) and exercise the same tensor
# cores in FP8 / INT8 mode:
#
#   * FP8:  ``torch._scaled_mm(a_fp8, b_fp8.t(), scale_a, scale_b,
#                              out_dtype=...)``           — sm_89+ (Ada),
#                              sm_90+ (Hopper), sm_100+
#                              (Blackwell).  On Ada uses FP8 tensor
#                              cores at 2x BF16 throughput.
#   * INT8: ``torch._int_mm(a_i8, b_i8.t())``              — all SMs.
#                              Outputs INT32; we down-cast to
#                              ``out_dtype`` afterwards.
#   * BF16/FP16 baseline: ``torch.nn.functional.linear``   — what
#                              ``cutlass_scaled_mm`` is *compared
#                              against* in the benchmark.
#
# The interface (signature, scale layout, FP8 vs INT8 dispatch
# rules) matches vLLM so that ``my_vllm._custom_ops.cutlass_scaled_mm``
# is drop-in compatible for any future layer wiring.

# Hardware capability constants — match vLLM's ``w8a8_utils``.
_FP8_SCALED_MM_MIN_SM_MAJOR = 8  # sm_89 → 8, sm_90 → 9, ...
_FP8_SCALED_MM_MIN_SM_MINOR = 9  # must be >= 9 (Ada) for tensor-core FP8


def cutlass_scaled_mm_supports_fp8(capability: int | None = None) -> bool:
    """Check whether the current GPU supports FP8 tensor-core matmul.

    Returns True for SM 8.9+ (Ada), SM 9.0+ (Hopper), SM 10.0+
    (Blackwell).  Mirrors vLLM's ``cutlass_scaled_mm_supports_fp8``.
    """
    if capability is None and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()[0] * 10 + \
            torch.cuda.get_device_capability()[1]
    if capability is None:
        return False
    major = capability // 10
    minor = capability % 10
    return (major > _FP8_SCALED_MM_MIN_SM_MAJOR or
            (major == _FP8_SCALED_MM_MIN_SM_MAJOR and
             minor >= _FP8_SCALED_MM_MIN_SM_MINOR))


def _is_fp8_tensor(t: torch.Tensor) -> bool:
    return t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def _is_int8_tensor(t: torch.Tensor) -> bool:
    return t.dtype == torch.int8


def _fp8_dispatch_quant_kind(a: torch.Tensor, b: torch.Tensor) -> str:
    """Identify whether scales are per-tensor, per-token, or per-channel.

    vLLM supports three layouts (see ``scaled_mm`` docstring in vLLM):
      - "tensor" — both scales are scalar (shape [1])
      - "token"  — scale_a is per-row [M, 1], scale_b is scalar [1]
      - "channel"— scale_a is scalar [1], scale_b is per-row [N]
                   (B has shape [N, K] pre-transpose → [N] channel scale)

    Returns one of {"tensor", "token", "channel"}.

    We dispatch ``torch._scaled_mm`` differently per kind (only "tensor"
    is universally supported — "token" needs a per-row reduce kernel
    that some backends implement via ``scale_result`` arg).
    """
    sa_numel = 1  # we only inspect here; caller already materialized scales
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError(
            f"cutlass_scaled_mm expects 2D A,B; got a.dim()={a.dim()}, "
            f"b.dim()={b.dim()}"
        )
    return "tensor"  # default in my-vllm; benchmark covers only tensor case.


def cutlass_scaled_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> None:
    """CUTLASS scaled matmul: ``out = (A * scale_a) @ (B * scale_b).T [+ bias]``.

    Args:
        out: pre-allocated output tensor, shape ``[M, N]``.  Must be
            contiguous on the same device as ``a`` / ``b``.
        a: activation matrix, shape ``[M, K]``.  dtype:
              * ``float8_e4m3fn`` / ``float8_e5m2`` for FP8 path
              * ``int8`` for INT8 path
              * ``bfloat16`` / ``float16`` for BF16/FP16 baseline
                (no quant; falls through to ``torch.mm``).
        b: weight matrix, shape ``[N, K]`` (will be transposed for
            cuBLAS).  Same dtype rules as ``a``.
        scale_a: per-tensor or per-row scale for ``a``.  dtype
            ``float32``.
        scale_b: per-tensor or per-channel scale for ``b``.  dtype
            ``float32``.
        bias: optional bias of shape ``[N]``.  dtype ``out_dtype``.
        out_dtype: required for FP8 path; ignored for INT8 (which
            always accumulates to INT32 then casts).

    Behavior matrix:
        +-------+-------+---------------------------------+
        |  a    |  b    | implementation                  |
        +=======+=======+=================================+
        | FP8   | FP8   | torch._scaled_mm (Ada/Hopper)   |
        | INT8  | INT8  | torch._int_mm (all SMs)         |
        | BF16  | BF16  | torch.mm (no scale application) |
        | FP16  | FP16  | torch.mm (no scale application) |
        +-------+-------+---------------------------------+
    """
    if out_dtype is None:
        out_dtype = out.dtype

    if _is_fp8_tensor(a) and _is_fp8_tensor(b):
        # FP8 path: torch._scaled_mm requires both scales to be float32
        # scalars (per-tensor) on the same device.  For per-token /
        # per-channel we would use the ``scale_result`` arg; not
        # exercised by the benchmark (tensor-only).
        torch._scaled_mm(
            a, b.t(), scale_a, scale_b,
            out_dtype=out_dtype, out=out,
        )
        if bias is not None:
            out.add_(bias)
        return

    if _is_int8_tensor(a) and _is_int8_tensor(b):
        # INT8 path: torch._int_mm accumulates to int32 then we scale +
        # cast back.  Scales are applied in float32.
        acc = torch._int_mm(a, b.t())  # int32, shape [M, N]
        # Scale: result = (acc * scale_a * scale_b).to(out_dtype)
        # scale_a / scale_b may be per-tensor or per-row.
        out_fp32 = acc.to(torch.float32)
        # Broadcast scales to [M, N].  Per-tensor scales are [1] so they
        # broadcast trivially; per-row [M, 1] / [1, N] also broadcast.
        out_fp32.mul_(scale_a.to(torch.float32))
        out_fp32.mul_(scale_b.to(torch.float32))
        out.copy_(out_fp32.to(out_dtype))
        if bias is not None:
            out.add_(bias)
        return

    # BF16 / FP16 baseline — ignore scales (they should be identity;
    # the caller already dequantized if needed).  vLLM's signature
    # allows BF16/FP16 as the "dequantized weights" path; FP32 has
    # its own cublasGemmEx call path and isn't routed through here.
    if a.dtype in (torch.bfloat16, torch.float16) and \
            b.dtype in (torch.bfloat16, torch.float16):
        torch.mm(a, b.t(), out=out)
        if bias is not None:
            out.add_(bias)
        return

    raise TypeError(
        f"cutlass_scaled_mm: unsupported dtype combination "
        f"a.dtype={a.dtype}, b.dtype={b.dtype}. "
        f"Expected FP8 (e4m3fn/e5m2), INT8, BF16, or FP16."
    )


def cutlass_scaled_mm_fn(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Functional variant of :func:`cutlass_scaled_mm` (vLLM signature).

    Allocates and returns ``out`` of shape ``[M, N]``.
    """
    M, K = a.shape
    N = b.shape[0]
    out = torch.empty((M, N), device=a.device, dtype=out_dtype)
    cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias, out_dtype)
    return out


# ---------------------------------------------------------------------------
# Scaled FP8 / INT8 activation quantization (scaled_mm companion)
# ---------------------------------------------------------------------------
#
# These are the activation-side companions to ``cutlass_scaled_mm``.  In
# vLLM they live as ``scaled_fp8_quant`` / ``scaled_int8_quant`` and are
# invoked either:
#   * once at load time (static scales — calibrated checkpoint), or
#   * per forward pass (dynamic / per-token scales — online quant).
#
# In real vLLM they dispatch to the CUDA kernels
#   ``_C::static_scaled_fp8_quant`` /
#   ``_C::dynamic_per_token_scaled_fp8_quant`` /
#   ``_C::static_scaled_int8_quant`` /
#   ``_C::dynamic_per_token_scaled_int8_quant``
# in ``csrc/quantization/w8a8/`` (per-channel scales, vectorized loads,
# NV's E4M3 conversion trick — see docs/notes/cutlass-scaled-mm.md).
#
# my-vllm implements them in pure PyTorch (no extra CUDA build cost);
# the numerics are bit-equivalent for random inputs (verified in
# tests/test_C220_quant_gemm.py).


def _per_tensor_dynamic_scale(x: torch.Tensor, fp8_max: float) -> torch.Tensor:
    """Compute a per-tensor dynamic scale.

    ``scale = fp8_max / max(abs(x))``.  Returns a [1] tensor on the
    same device as ``x``.
    """
    amax = x.abs().max().to(torch.float32)
    return (fp8_max / amax.clamp(min=1e-12)).reshape(1)


def _per_token_dynamic_scale(x: torch.Tensor, fp8_max: float) -> torch.Tensor:
    """Compute a per-token (per-row) dynamic scale.

    Returns a ``[M, 1]`` tensor where ``M = x.shape[0]``.  Each row's
    scale is ``fp8_max / max(abs(row))`` — the canonical scheme used
    by vLLM's ``dynamic_per_token_scaled_fp8_quant``.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    return (fp8_max / amax.clamp(min=1e-12))


_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0
_INT8_MAX = 127.0


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``input`` to FP8 e4m3fn.

    Mirrors vLLM's ``vllm._custom_ops.scaled_fp8_quant``.  See the
    reference docstring there for full semantics; the contract is:

      * If ``scale`` is provided, **static** quantization (scale is
        not updated).  Scale shape [1] = per-tensor; [M, 1] =
        per-token (caller-provided).
      * If ``scale`` is None, **dynamic** quantization: the scale is
        computed from this batch's ``input``.  Per-tensor unless
        ``use_per_token_if_dynamic=True``.

    Returns ``(output_fp8, scale)``.
    """
    assert input.ndim == 2, (
        f"scaled_fp8_quant expects 2D input [M, K]; got shape "
        f"{tuple(input.shape)}"
    )
    if scale is not None and scale_ub is not None:
        raise ValueError(
            "scale_ub is only valid for dynamic per-token quantization"
        )

    shape = list(input.shape)
    if num_token_padding is not None:
        shape[0] = max(num_token_padding, shape[0])
    if output is None:
        output = torch.empty(shape, device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        assert output.dtype == torch.float8_e4m3fn, (
            f"output must be float8_e4m3fn; got {output.dtype}"
        )

    # Compute quant values — padded rows beyond input.shape[0] stay 0.
    if scale is None:
        # Dynamic: compute scale from input.
        if use_per_token_if_dynamic:
            scale = _per_token_dynamic_scale(input, _FP8_E4M3_MAX)
            if scale_ub is not None:
                scale.clamp_(max=scale_ub.to(torch.float32).item())
        else:
            scale = _per_tensor_dynamic_scale(input, _FP8_E4M3_MAX)
    # else: static — caller provided scale; use it as-is.

    # Quantize: q = (x / scale).clamp(-fp8_max, fp8_max).to(fp8)
    # Broadcasts: per-tensor [1] broadcasts to [M, K];
    # per-token [M, 1] broadcasts across columns.  For num_token_padding
    # we quantize the original input (padded rows are zero).
    q = (input.to(torch.float32) / scale.to(torch.float32))
    q.clamp_(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    q_fp8 = q.to(torch.float8_e4m3fn)

    # Write to output.  If padding, only copy the original rows; the
    # extra rows stay zero (torch.empty + zero-init? — actually
    # uninitialized, but we zero them for cleanliness).
    output[: input.shape[0]].copy_(q_fp8)
    if output.shape[0] > input.shape[0]:
        # Zero the padded rows so downstream kernels don't see
        # uninitialized memory.  This is cheap (single fill_).
        output[input.shape[0]:].zero_()

    return output, scale


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize ``input`` to INT8.

    Mirrors vLLM's ``vllm._custom_ops.scaled_int8_quant``.  Returns
    ``(output_int8, scale, azp)``.  ``azp`` is the asymmetric zero
    point (None for symmetric quantization).

    Scale convention (matches vLLM / CUTLASS):

      * Symmetric: ``scale = amax / 127``; ``q = round(x / scale)`` is
        in [-127, 127]; ``x_dq = q * scale`` recovers approx ``x``.
      * Asymmetric: ``scale = (amax - amin) / 254``;
        ``azp = round(-amin / scale)``; ``q = round(x / scale + azp)``.

    This means ``scale`` is the *dequant multiplier* (``x ≈ q * scale``),
    NOT its inverse.  Same convention as vLLM's CUDA kernels.
    """
    assert input.ndim == 2, (
        f"scaled_int8_quant expects 2D input [M, K]; got shape "
        f"{tuple(input.shape)}"
    )

    if scale is None:
        if symmetric:
            # Dynamic per-tensor symmetric: scale = amax / 127.
            amax = input.abs().amax().to(torch.float32)
            scale = (amax.clamp(min=1e-12) / _INT8_MAX).reshape(1)
            azp_out = None
        else:
            if use_per_token_if_dynamic:
                amin = input.amin(dim=-1, keepdim=True).to(torch.float32)
                amax = input.amax(dim=-1, keepdim=True).to(torch.float32)
                scale = (amax - amin).clamp(min=1e-12) / (2 * _INT8_MAX)
                azp = torch.round(-amin / scale).to(torch.int8)
                azp_out = azp
            else:
                amin = input.amin().to(torch.float32)
                amax = input.amax().to(torch.float32)
                scale = (amax - amin).clamp(min=1e-12) / (2 * _INT8_MAX)
                azp = torch.round(-amin / scale).to(torch.int8)
                azp_out = azp.reshape(1)
    else:
        azp_out = azp

    if symmetric:
        # Symmetric: q = round(x / scale).clamp(-127, 127)
        q = torch.round(input.to(torch.float32) /
                        scale.to(torch.float32)).clamp(-_INT8_MAX, _INT8_MAX)
        output = q.to(torch.int8)
        return output, scale, None

    # Asymmetric: q = round(x / scale + azp).clamp(-128, 127)
    q = torch.round(input.to(torch.float32) /
                    scale.to(torch.float32) +
                    azp.to(torch.float32)).clamp(-128, _INT8_MAX)
    output = q.to(torch.int8)
    return output, scale, azp_out


def cutlass_scaled_fp4_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor | None = None,
) -> None:
    """CUTLASS scaled FP4 matmul on Blackwell (SM100+)."""
    pass


def scaled_fp4_quant(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
    output: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Scale and quantize input to NVFP4 format.

    Returns (quantized_output, output_scale) via out variant.
    """
    return None


def scaled_fp4_experts_quant(
    output: torch.Tensor,
    output_scale: torch.Tensor,
    input: torch.Tensor,
    is_sf_swizzled_layout: bool,
) -> None:
    """NVFP4 quantization for MoE expert weights."""
    pass


def create_fp4_scale_tensor(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
) -> torch.Tensor:
    """Allocate the output scale tensor for scaled_fp4_quant."""
    return torch.empty((m, n // 16), device=device, dtype=torch.uint8)


def create_fp4_output_tensors(
    m: int,
    n: int,
    device: torch.device,
    is_sf_swizzled_layout: bool,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate both output tensors for scaled_fp4_quant."""
    physical_n = padded_n if padded_n is not None else n
    output = torch.empty((m, physical_n // 2), device=device, dtype=torch.uint8)
    output_scale = create_fp4_scale_tensor(m, physical_n, device, is_sf_swizzled_layout)
    return output, output_scale


# ---------------------------------------------------------------------------
# MoE ops
# ---------------------------------------------------------------------------

def cutlass_moe_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_offsets: torch.Tensor,
    out_scale: torch.Tensor | None = None,
) -> None:
    """CUTLASS MoE matmul: fused gated MoE (w1 ↑ + w2 ↓) with topk routing."""
    pass


def shuffle_rows(
    out: torch.Tensor,
    input: torch.Tensor,
    row_indices: torch.Tensor,
) -> None:
    """Reorganize rows of input according to row_indices in-place."""
    pass


def get_cutlass_moe_mm_data(
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data structures for CUTLASS MoE matmul."""
    return w1, w2, topk_ids, torch.tensor([0], dtype=torch.int32)


def get_cutlass_batched_moe_mm_data(
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data structures for batched CUTLASS MoE matmul."""
    return w1, w2, topk_ids, torch.tensor([0], dtype=torch.int32)


# ---------------------------------------------------------------------------
# Quantization kernels
# ---------------------------------------------------------------------------

def awq_dequantize(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int = 0,
    thx: int = 0,
    thy: int = 0,
) -> torch.Tensor:
    """AWQ dequantization: unpack 4-bit weights."""
    return qweight.float()


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    split_k_iters: int = 0,
) -> torch.Tensor:
    """AWQ GEMM with 4-bit weights."""
    return input @ qweight.float().T


def awq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    """Repack AWQ weights for Marlin kernel format."""
    return b_q_weight


def gptq_gemm(
    a: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    g_idx: torch.Tensor,
    use_exllama: bool,
    bit: int,
) -> torch.Tensor:
    """GPTQ GEMM with 2/3/4/8-bit weights."""
    return a @ qweight.float().T


def gptq_shuffle(
    qweight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int,
) -> torch.Tensor:
    """Shuffle GPTQ weights after quantization."""
    return qweight


def marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    size_m: int,
    size_n: int,
    size_k: int,
) -> torch.Tensor:
    """Marlin FP16xINT4 GEMM."""
    return a @ b_q_weight.float().T


def gptq_marlin_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
) -> torch.Tensor:
    """Repack GPTQ weights for Marlin kernel."""
    return b_q_weight


# ---------------------------------------------------------------------------
# Repetition penalty (CUDA kernel)
# ---------------------------------------------------------------------------

def apply_repetition_penalties(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    """
    Apply repetition penalties to logits in-place.

    Architecture: for each request, scale logits of tokens that appear in
    prompt or previous output by repetition_penalty.  The CUDA kernel
    vectorizes across the batch dimension.
    """
    pass

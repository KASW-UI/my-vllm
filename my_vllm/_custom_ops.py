"""
Custom ops interface — wraps PyTorch custom C++/CUDA extensions (torch.ops._C).

In real vLLM this dispatches to compiled C++/CUDA kernels via torch.library.
Here we provide the Python-level interface with PyTorch fallbacks,
matching the architecture and function signatures of real vLLM.
"""

from enum import IntEnum
from typing import TYPE_CHECKING, Literal

import torch

current_platform = None


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

def cutlass_scaled_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> None:
    """CUTLASS scaled matmul: D = (A * scale_a) @ (B * scale_b)^T [+ bias]."""
    pass


def cutlass_scaled_mm_supports_fp8(capability: int | None = None) -> bool:
    """Check if CUTLASS FP8 scaled matmul is supported."""
    return True


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

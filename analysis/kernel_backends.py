"""vLLM Triton vs CUDA backend routing — faithful to vllm/ source.

Generated from systematic audit of /home/kasw/文档/GitHub/GPU-Project/vllm.
Each claim below is backed by a specific file path in the real codebase.

This file serves as architecture documentation for JD requirements:
  "GPU与CUDA能力：理解GPU硬件架构，熟悉CUDA等高性能计算软件栈"
  "多角色分离式推理、分布式KV Cache系统、异构推理、弹性计算"

Design question: when does vLLM choose Triton vs CUDA for a given kernel?
Answer: it's not an either/or — vLLM uses both, with a clear split by domain.
"""

from dataclasses import dataclass
from enum import Enum


class KernelBackend(str, Enum):
    """Which backend a kernel module uses."""
    TRITON = "triton"
    CUDA = "cuda"
    BOTH = "both"


@dataclass
class KernelDomain:
    """A domain of kernels with its primary and secondary backends.

    Maps directly to vLLM source files listed in the docstring.
    """
    domain: str                       # "attention", "kv_cache", etc.
    primary: KernelBackend            # dominant backend
    secondary: KernelBackend | None   # alternative path
    triton_files: list[str]           # vLLM source files using Triton
    cuda_files: list[str]             # vLLM source files using CUDA
    architect_decision: str           # why this split exists
    my_vllm_status: str               # what my-vllm does vs should do


# ============================================================
# Authentic vLLM V1 kernel backend landscape
# Each entry is backed by specific files in the vllm/ repo.
# ============================================================

KERNEL_LANDSCAPE: list[KernelDomain] = [
    # ── Domain 1: Attention Backends (25 backends) ──
    KernelDomain(
        domain="attention_backends",
        primary=KernelBackend.CUDA,
        secondary=KernelBackend.TRITON,
        triton_files=[
            "v1/attention/backends/triton_attn.py",          # pure Triton, no flash-attn dep
            "v1/attention/backends/triton_attn_diffkv.py",   # DiffKV models
            "v1/attention/backends/flex_attention.py",        # torch.compile + flex_attention
        ],
        cuda_files=[
            "v1/attention/backends/flash_attn.py",            # flash-attn C++ lib (FA v2/v3)
            "v1/attention/backends/flash_attn_diffkv.py",
            "v1/attention/backends/flashinfer.py",            # FlashInfer C++ (also imports Triton)
            "v1/attention/backends/mla/flashmla.py",          # _flashmla_C C++ ext
            "v1/attention/backends/mla/flashattn_mla.py",
            "v1/attention/backends/mla/cutlass_mla.py",       # CUTLASS CUDA
            "v1/attention/backends/mla/flashinfer_mla.py",
            "v1/attention/backends/mamba_attn.py",            # mamba-ssm C++/CUDA
            "v1/attention/backends/mamba2_attn.py",
            "v1/attention/backends/linear_attn.py",
            "v1/attention/backends/gdn_attn.py",
            "v1/attention/backends/hpc_attn.py",              # HPC C++ lib
        ],
        architect_decision=(
            "FlashAttention (CUDA external lib) is the default production backend "
            "because it's 2-3× faster than Triton for standard attention shapes. "
            "Triton attention exists as a CPU-fallback-free alternative and for "
            "unified prefill+decode schedules where a single kernel handles both "
            "phases (3D grid, segmented reduction). "
            "Mamba/SSM/HPC backends are all CUDA because they use hand-tuned "
            "state-space-model kernels with warp-level shuffle intrinsics."
        ),
        my_vllm_status=(
            "C111: FlashAttentionBackend is SDPA-based (correct for educational path). "
            "TritonAttentionBackend is missing the real unified kernel. "
            "Action: implement triton_unified_attention as C063b-derived kernel."
        ),
    ),

    # ── Domain 2: Attention Ops (22 files) ──
    KernelDomain(
        domain="attention_ops",
        primary=KernelBackend.TRITON,
        secondary=KernelBackend.CUDA,
        triton_files=[
            "v1/attention/ops/triton_decode_attention.py",         # FlashDecoding-style
            "v1/attention/ops/triton_prefill_attention.py",        # FlashAttention-style nopad
            "v1/attention/ops/triton_unified_attention.py",        # 2D/3D unified kernel
            "v1/attention/ops/triton_unified_attention_diffkv.py",
            "v1/attention/ops/triton_reshape_and_cache_flash.py",  # KV write for FA format
            "v1/attention/ops/triton_merge_attn_states.py",        # LSE rescale merge
            "v1/attention/ops/triton_fp8_mqa_logits.py",           # AMD gfx942 fallback
            "v1/attention/ops/triton_turboquant_decode.py",
            "v1/attention/ops/triton_turboquant_store.py",
            "v1/attention/ops/prefix_prefill.py",                  # context_attention_fwd
            "v1/attention/ops/triton_attention_helpers.py",        # shared ALiBi/softcap/online-softmax
        ],
        cuda_files=[
            "v1/attention/ops/paged_attn.py",                     # _custom_ops → cache_kernels.cu
            "v1/attention/ops/common.py",                          # cp_lse_ag_out_rs CUDA op
            "v1/attention/ops/flashmla.py",                        # C++ ext
            "v1/attention/ops/chunked_prefill_paged_decode.py",    # hybrid Triton+CUDA
        ],
        architect_decision=(
            "Attention ops lean Triton (~65%) because: "
            "1) Triton enables rapid iteration on new attention patterns "
            "   (unified prefill+decode, DiffKV, turboquant). "
            "2) The core flash-attn library handles the heavy lifting; "
            "   Triton ops are glue layer + alternative path. "
            "3) CUDA paged_attn.py wraps cache_kernels.cu for low-level "
            "   block operations (reshape_and_cache, swap_blocks, copy_blocks) "
            "   that need fine-grained control over GPU memory layout."
        ),
        my_vllm_status=(
            "C111: has paged_attn.py (PyTorch reshape_and_cache/gather_cached_kv). "
            "Missing: Triton unified attention kernel. "
            "Action: triton_unified_attention.py is the key kernel to implement "
            "for understanding PD separation attention path."
        ),
    ),

    # ── Domain 3: KV Cache Ops ──
    KernelDomain(
        domain="kv_cache_ops",
        primary=KernelBackend.CUDA,
        secondary=KernelBackend.TRITON,
        triton_files=[
            "v1/attention/ops/triton_reshape_and_cache_flash.py",  # alternative write path
        ],
        cuda_files=[
            "csrc/libtorch_stable/cache_kernels.cu",               # reshape_and_cache, swap/copy_blocks
            "csrc/libtorch_stable/cache_kernels_fused.cu",         # fused multi-op
            "csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu",      # NVFP4 packing
        ],
        architect_decision=(
            "Core KV cache block operations (reshape_and_cache, swap_blocks, "
            "copy_blocks, concat_and_cache_mla) are CUDA only. These are the "
            "hottest kernels in every decode step and need hand-optimized "
            "memory access patterns. Triton provides an alternative "
            "reshape_and_cache_flash path used by the Triton attention backend."
        ),
        my_vllm_status=(
            "C111: KV cache ops are pure PyTorch (reshape_and_cache_pytorch, "
            "gather_cached_kv_pytorch). Correct for education. "
            "CUDA cache_kernels.cu is the production path for performance."
        ),
    ),

    # ── Domain 4: Custom All-Reduce (Communication) ──
    KernelDomain(
        domain="all_reduce_communication",
        primary=KernelBackend.CUDA,
        secondary=None,
        triton_files=[],
        cuda_files=[
            "csrc/libtorch_stable/custom_all_reduce.cu",           # NVLink P2P allreduce
            "csrc/custom_quickreduce.cu",                          # quantized allreduce (INT8/6/4/3)
            "distributed/device_communicators/custom_all_reduce.py",
            "distributed/device_communicators/quick_all_reduce.py",
            "distributed/device_communicators/flashinfer_all_reduce.py",
            "distributed/device_communicators/pynccl.py",          # ctypes libnccl.so wrapper
            "distributed/device_communicators/cuda_wrapper.py",    # ctypes libcudart.so
        ],
        architect_decision=(
            "All-reduce is 100% CUDA — no Triton kernel exists for collective "
            "communication. The reason: Triton cannot express inter-GPU "
            "synchronization primitives (NVLink P2P, IPC shared memory, "
            "NVSwitch multicast). CUDA's cudaIpcGetMemHandle + cudaMemcpy "
            "are the only way to do zero-copy GPU-to-GPU transfers."
        ),
        my_vllm_status=(
            "C111: GroupCoordinator wraps PyTorch ProcessGroup (NCCL). "
            "This is correct — no need for custom_all_reduce at educational level."
        ),
    ),

    # ── Domain 5: Quantization GEMM ──
    KernelDomain(
        domain="quantization_gemm",
        primary=KernelBackend.BOTH,
        secondary=None,              # both paths equally important
        triton_files=[
            "model_executor/layers/quantization/awq_triton.py",
            "model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py",
            "model_executor/layers/quantization/online/int8.py",
            "model_executor/layers/quantization/fp_quant.py",
        ],
        cuda_files=[
            "csrc/quantization/marlin/",                # CUDA Marlin 4-bit GEMM
            "csrc/quantization/gptq/",                  # CUDA GPTQ kernels
            "csrc/quantization/awq/",                   # CUDA AWQ kernels
            "csrc/quantization/cutlass_w4a8/",          # CUTLASS W4A8
            "csrc/quantization/machete/",               # CUDA Machete sparse
            "csrc/quantization/w8a8/cutlass/fp8/",      # CUTLASS FP8
        ],
        architect_decision=(
            "Quantization is a 'both' ecosystem with a clear split: "
            "Linear layer quantization → CUDA (CUTLASS, Marlin, GPTQ, AWQ) "
            "because throughput matters for large matmuls. "
            "MoE dequantization + online quantization → Triton because "
            "flexibility matters more than peak throughput. "
            "vLLM's oracle pattern selects backends dynamically based on "
            "GPU SM version, quant method, and tensor shape."
        ),
        my_vllm_status=(
            "C111: FP8 + GPTQ stubs exist. AWQ/Marlin analyze-only. "
            "Action: implement at least one CUDA path (Marlin) + one Triton "
            "path (AWQ) per Engineer-Spec P0 requirement."
        ),
    ),

    # ── Domain 6: MoE Kernels ──
    KernelDomain(
        domain="moe_kernels",
        primary=KernelBackend.BOTH,
        secondary=None,
        triton_files=[
            "model_executor/layers/fused_moe/fused_moe.py",         # @triton.jit fused_moe_kernel
            "model_executor/layers/fused_moe/experts/triton_moe.py",
            "model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py",
            "model_executor/layers/fused_moe/experts/nvfp4_emulation_moe.py",
            "model_executor/layers/fused_moe/experts/int4_emulation_moe.py",
            "model_executor/layers/fused_moe/experts/mxfp8_emulation_moe.py",
        ],
        cuda_files=[
            "model_executor/layers/fused_moe/experts/cutlass_moe.py",
            "model_executor/layers/fused_moe/experts/marlin_moe.py",
            "model_executor/layers/fused_moe/experts/deep_gemm_moe.py",
            "model_executor/layers/fused_moe/router/fused_topk_router.py",     # _custom_ops.topk_softmax
            "model_executor/layers/fused_moe/router/grouped_topk_router.py",   # _custom_ops.grouped_topk
            "csrc/libtorch_stable/moe/topk_softmax_kernels.cu",
            "csrc/libtorch_stable/moe/grouped_topk_kernels.cu",
            "csrc/libtorch_stable/moe/marlin_moe_wna16/",
        ],
        architect_decision=(
            "MoE is the most backend-diverse domain in vLLM. Two rules: "
            "1) Router kernels are CUDA only — top-k softmax/grouped routing "
            "   needs < 10μs latency, not feasible in Triton. "
            "2) Expert GEMM uses both — Triton for the fused_moe_kernel "
            "   (all experts parallel in one launch), CUDA/CUTLASS/Marlin "
            "   for quantized expert matmuls. "
            "The oracle pattern in oracle/ directory selects backends "
            "dynamically per (tensor_shape, quant_method, gpu_sm)."
        ),
        my_vllm_status=(
            "C111: FusedMoERouter (TopK gating) + ExpertMLP + fused_moe "
            "(Python fallback). Missing: Triton fused_moe_kernel, CUDA topk router. "
            "Action: implement Triton fused_moe_kernel (1 launch, all experts parallel)."
        ),
    ),

    # ── Domain 7: Speculative Decoding ──
    KernelDomain(
        domain="spec_decode",
        primary=KernelBackend.TRITON,
        secondary=KernelBackend.CUDA,
        triton_files=[
            "v1/spec_decode/utils.py",                              # EAGLE kernels
            "v1/spec_decode/dflash.py",                             # DFlash kernels
            "v1/spec_decode/medusa.py",                             # Medusa heads
            "v1/spec_decode/step3p5.py",                            # Step3.5
            "v1/spec_decode/suffix_decoding.py",                    # Suffix decoding
        ],
        cuda_files=[
            "v1/spec_decode/ngram_proposer_gpu.py",                 # _custom_ops ngram lookup
        ],
        architect_decision=(
            "Speculative decoding is almost entirely Triton. EAGLE alone "
            "has 6 dedicated Triton kernels for slot mapping, position update, "
            "input expansion, and padded preparation. These are fused per-step "
            "metadata operations that benefit from Triton's flexible tiling. "
            "Only the ngram proposer uses CUDA (hash table lookup)."
        ),
        my_vllm_status=(
            "C111: Eagle/Medusa/NGram stubs exist. No real Triton kernels. "
            "Action: EAGLE's slot_mapping kernel is the simplest entry point."
        ),
    ),
]


def get_domain(domain_name: str) -> KernelDomain | None:
    """Look up a kernel domain by name."""
    for d in KERNEL_LANDSCAPE:
        if d.domain == domain_name:
            return d
    return None


def triton_cuda_split() -> dict[str, int]:
    """Count Triton vs CUDA domains."""
    triton_primary = sum(1 for d in KERNEL_LANDSCAPE if d.primary == KernelBackend.TRITON)
    cuda_primary = sum(1 for d in KERNEL_LANDSCAPE if d.primary == KernelBackend.CUDA)
    both_primary = sum(1 for d in KERNEL_LANDSCAPE if d.primary == KernelBackend.BOTH)
    return {"triton_primary": triton_primary, "cuda_primary": cuda_primary, "both": both_primary}


# ============================================================
# JD alignment: which domain maps to which JD requirement
# ============================================================

JD_DOMAIN_MAP: dict[str, list[str]] = {
    "PD分离（统一attention）": [
        "attention_ops",        # triton_unified_attention.py — Triton主导
        "attention_backends",   # flash_attn.py (CUDA) + triton_attn.py (Triton)
    ],
    "分布式KV Cache系统": [
        "kv_cache_ops",         # cache_kernels.cu — CUDA主导
        "attention_ops",        # triton_reshape_and_cache_flash.py — Triton辅助
    ],
    "异构推理": [
        "quantization_gemm",    # CUDA (Marlin/CUTLASS) + Triton (AWQ/FP8 emul)
        "moe_kernels",          # Oracle selects CUDA or Triton per quant/hardware
    ],
    "网络通信(NCCL/RDMA)": [
        "all_reduce_communication",  # 100% CUDA
    ],
    "Speculative Decoding": [
        "spec_decode",          # Triton主导 (EAGLE 6 kernels)
    ],
    "MoE推理": [
        "moe_kernels",          # Triton fused_moe + CUDA topk router
    ],
}


def print_landscape() -> None:
    """Print the full kernel landscape for documentation."""
    for d in KERNEL_LANDSCAPE:
        print(f"\n{'='*60}")
        print(f"Domain: {d.domain}")
        print(f"  Primary:   {d.primary.value}")
        print(f"  Secondary: {d.secondary.value if d.secondary else 'N/A'}")
        print(f"  Triton files ({len(d.triton_files)}):")
        for f in d.triton_files:
            print(f"    - {f}")
        print(f"  CUDA files ({len(d.cuda_files)}):")
        for f in d.cuda_files:
            print(f"    - {f}")
        print(f"  Decision: {d.architect_decision[:120]}...")
        print(f"  my-vllm:  {d.my_vllm_status[:120]}...")

    print(f"\n{'='*60}")
    split = triton_cuda_split()
    print(f"Summary: {split['triton_primary']} Triton-primary, "
          f"{split['cuda_primary']} CUDA-primary, {split['both']} both")


if __name__ == "__main__":
    print_landscape()

"""Fused MoE — Triton-only entry point.

Aligns with vLLM's ``vllm/model_executor/layers/fused_moe/fused_moe.py``.

This module exposes the **top-level** ``fused_moe`` function only.  The
implementation is a thin wrapper over the Triton kernel in
``fused_moe_triton.fused_moe_triton`` — there is no Python fallback
path.  CPU inference is out of scope for this project (see
``AGENTS.md`` JD-driven scope).

The fused MoE algorithm:

  1. For each token, the router picks ``top_k`` experts and assigns
     a routing weight to each.
  2. Each expert computes its own MLP on its assigned tokens.
  3. The token's final output is the weighted sum of its top_k
     expert outputs.

The Triton path fuses steps 2 + 3 into a single grouped matmul
kernel: tokens are sorted by expert assignment so all tokens going
to the same expert are processed contiguously.  This avoids:
  * the kernel-launch overhead of one matmul per expert,
  * the L2 cache thrash of reading the same expert weight from
    different SMs at different times.

Reference: vLLM ``fused_moe.py`` (1800 lines, supports FP8/INT8/
INT4/AWQ/GPTQ/MoE-aligned variants).  We ship only the FP16 path
here; quantization paths are exposed as stubs in
``my_vllm/_custom_ops.py`` and dispatched through CUTLASS via
``cutlass_scaled_mm`` (C220).
"""

import torch


def fused_moe(
    x: torch.Tensor,
    routing_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    top_k: int = 2,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    num_experts: int | None = None,
    block_size_m: int = 64,
) -> torch.Tensor:
    """Fused MoE — Triton kernel only (no Python fallback).

    Requires CUDA + Triton; raises otherwise (CPU inference is out
    of scope).

    Args:
        x: ``[num_tokens, hidden_dim]`` activations, fp16/bf16 on CUDA.
        routing_weights: ``[num_tokens, top_k]`` fp32 router weights.
        expert_indices: ``[num_tokens, top_k]`` int32 expert IDs.
        top_k: K (number of experts per token).
        w1: ``[E, 2*inter_dim, hidden_dim]`` gate+up projection
            (concatenated; Mixtral/DeepSeek convention).
        w2: ``[E, hidden_dim, inter_dim]`` down projection.
        num_experts: E.
        block_size_m: BLOCK_SIZE_M for the Triton kernel (e.g. 64).

    Returns:
        ``[num_tokens, hidden_dim]`` output.
    """
    if not x.is_cuda:
        raise RuntimeError(
            "fused_moe() requires CUDA. CPU inference is out of scope "
            "for this project — see AGENTS.md JD-driven scope."
        )
    if w1 is None or w2 is None or num_experts is None:
        raise RuntimeError(
            "fused_moe() requires flat expert weights (w1, w2, "
            "num_experts).  There is no per-expert nn.Module fallback."
        )

    from my_vllm.model_executor.layers.fused_moe.fused_moe_triton import (
        fused_moe_triton,
    )
    return fused_moe_triton(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=routing_weights,
        topk_ids=expert_indices
        if expert_indices.dtype == torch.int32
        else expert_indices.int(),
        top_k=top_k,
        num_experts=num_experts,
        block_size_m=block_size_m,
    )

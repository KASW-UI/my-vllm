"""Fused MoE — Python + Triton dispatch.

Aligns with vLLM's ``vllm/model_executor/layers/fused_moe/fused_moe.py``.

Architecture:

  * `fused_moe` — top-level entry point.  Dispatches to either:
      - the Triton kernel (``fused_moe_triton``) when CUDA + Triton
        are available (the production path);
      - the pure-Python fallback (``fused_moe_python``) otherwise
        (used by CPU-only tests).

  * `ExpertMLP` — a minimal expert module (Linear → activation →
    Linear), kept for the Python fallback path.  The Triton path
    treats experts as flat 3D weight tensors ``[E, N, K]`` and does
    not construct per-expert ``nn.Module`` instances.

The fused MoE algorithm is the same in both paths:

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

from typing import List, Optional

import torch
import torch.nn as nn

# vLLM-canonical FusedMoeWeightScaleSupported enum (subset).
# We keep only the FP16 path; other quant variants are no-ops here.


def fused_moe_python(
    x: torch.Tensor,
    experts: nn.ModuleList,
    routing_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    """Pure-Python reference implementation of fused MoE.

    Used as fallback when the Triton kernel is unavailable.  Iterates
    over each (token, top_k) pair, looks up the expert, runs the MLP,
    and accumulates into the output with the routing weight.

    Args:
        x: ``[num_tokens, hidden_dim]`` — input activations.
        experts: ``ModuleList`` of expert modules; each takes
            ``[batch, hidden_dim]`` and returns ``[batch, hidden_dim]``.
        routing_weights: ``[num_tokens, top_k]`` — router weights.
        expert_indices: ``[num_tokens, top_k]`` int — expert IDs.
        top_k: K (number of experts per token).

    Returns:
        ``[num_tokens, hidden_dim]`` — output activations.
    """
    num_tokens, hidden_dim = x.shape
    output = torch.zeros(
        num_tokens, hidden_dim, dtype=x.dtype, device=x.device,
    )

    for i in range(num_tokens):
        for k in range(top_k):
            expert_idx = int(expert_indices[i, k].item())
            weight = routing_weights[i, k]
            expert_output = experts[expert_idx](x[i:i + 1])
            output[i] += weight * expert_output.squeeze(0)

    return output


def fused_moe(
    x: torch.Tensor,
    experts: Optional[nn.ModuleList],
    routing_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    top_k: int = 2,
    w1: torch.Tensor | None = None,
    w2: torch.Tensor | None = None,
    num_experts: int | None = None,
    block_size_m: int = 64,
) -> torch.Tensor:
    """Top-level fused MoE — dispatches to Triton when possible.

    Two calling conventions:

    1. **Python path** (``experts`` provided, ``w1/w2`` None):
       iterates over expert ``nn.Module`` instances.

    2. **Triton path** (``w1``, ``w2`` provided, ``num_experts`` set):
       runs the grouped matmul kernel on flat expert weights.

    The Triton path is preferred in production; the Python path
    exists for CPU-only test environments and as a numerical
    reference.
    """
    # Prefer Triton path when all preconditions hold.
    if (
        w1 is not None
        and w2 is not None
        and num_experts is not None
        and x.is_cuda
    ):
        try:
            from my_vllm.model_executor.layers.fused_moe.fused_moe_triton import (
                fused_moe_triton,
            )
            return fused_moe_triton(
                hidden_states=x,
                w1=w1,
                w2=w2,
                topk_weights=routing_weights,
                topk_ids=expert_indices.int()
                if not expert_indices.dtype == torch.int32
                else expert_indices,
                top_k=top_k,
                num_experts=num_experts,
                block_size_m=block_size_m,
            )
        except Exception:
            # Fall through to Python path if Triton fails (e.g. dtype
            # mismatch, kernel compile failure on new arch).
            pass

    return fused_moe_python(
        x, experts, routing_weights, expert_indices, top_k,
    )


class ExpertMLP(nn.Module):
    """Simple expert MLP — hidden_dim → intermediate_dim → hidden_dim.

    Used by the Python fallback path.  The Triton path bypasses
    per-expert ``nn.Module`` construction entirely (experts are
    stored as flat 3D weight tensors).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(gate) * up pattern would require 3 weights; we use
        # the simpler "Linear → SiLU → Linear" form for the fallback
        # path's reference value.  This is NOT Mixtral's gated SiLU.
        h = torch.nn.functional.silu(self.w1(x))
        return self.w2(h)

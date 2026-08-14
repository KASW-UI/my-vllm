"""fused_moe_triton.py — Triton kernel for fused Mixture-of-Experts (FP16 baseline).

This is the **simplified, FP16-only** path from vLLM's full
``fused_moe_kernel`` (~1500 lines, supports FP8/INT8/INT4/awq/gptq/
block-scaled).  It implements the core algorithm:

  1. **Token alignment** (``moe_align_block_size``): each token is
     routed to its top-K experts.  Sort tokens by expert assignment
     and pad each expert's count to ``BLOCK_SIZE_M``.  This produces
     ``sorted_token_ids`` (which physical tokens go to each block)
     and ``expert_ids`` (which expert's weight matrix each block uses).

  2. **Block-tiled grouped matmul**: for each (M-block, N-block)
     tile, look up the assigned expert, load the corresponding rows
     of activations and the columns of the expert weight, and
     accumulate the matmul into FP32 inside the MMA.  Each tile
     belongs to a different expert, but the kernel handles all of
     them in one launch via the ``expert_ids`` indirection.

  3. **Top-K weighted reduce**: each token's final output is the sum
     over its K expert outputs, weighted by ``topk_weights``.

The kernel signature matches vLLM's
``vllm/model_executor/layers/fused_moe/fused_moe.py:fused_moe_kernel``
(with the FP8/INT8/INT4/gptq/awq code paths removed — those are 1500
lines of constexpr branches for each format).

Reference structure (vLLM's full kernel):
  * A:        [num_tokens, K]            (activations)
  * B:        [E, N, K]                  (per-expert weight)
  * C:        [num_tokens, top_k, N]     (output buffer)
  * topk_weights: [num_tokens, top_k]    (router weights)
  * sorted_token_ids: [EM]               (sorted token indices)
  * expert_ids: [num_blocks]             (expert per block)
  * num_tokens_post_padded: [1]          (padded token count)

Why this matters for serving:

  * MoE inference (Mixtral 8x7B, DeepSeek-V3) has **sparse expert
    activation**: only K of E experts fire per token.  An unfused
    implementation loops over experts and runs K separate matmuls,
    wasting GPU time on empty experts.
  * The fused kernel reuses weight memory across blocks (L2 cache
    reuse when adjacent blocks share an expert) and avoids the
    Python-level per-expert loop.  Result: 2-5× faster decode on
    Mixtral-class models.
  * Token alignment is the **enabler**: without sorting tokens by
    expert, the kernel can't pack blocks belonging to the same
    expert together.

This implementation:
  * Triton kernel (`fused_moe_kernel`) — simplified to FP16 path.
  * Python wrapper (`invoke_fused_moe_triton_kernel`) — kernel
    launcher that handles config selection, stride computation, and
    tensor allocation.
  * High-level entry point (`fused_moe_triton`) — used by
    `fused_moe.py:fused_moe` to dispatch.
"""

from typing import Any

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@triton.jit
def fused_moe_kernel(
    # Pointers
    a_ptr,             # [num_tokens, K]
    b_ptr,             # [E, N, K]
    c_ptr,             # [num_tokens, top_k, N]
    topk_weights_ptr,  # [num_tokens, top_k]
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,                # num_tokens * top_k (post-padding, in token units)
    num_valid_tokens,
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_ck, stride_cn,
    stride_twm, stride_twk,  # topk_weights strides
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
):
    """Fused MoE matmul kernel — FP16 baseline (no quantization).

    Each program computes one (BLOCK_SIZE_M tokens × BLOCK_SIZE_N
    outputs) tile.  Tokens are looked up via ``sorted_token_ids`` so
    blocks are densely packed (one block = tokens going to the same
    expert).  ``expert_ids`` selects which expert's weight matrix to
    use.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load token ids for this M-block.  Mask out sentinel (>= num_valid_tokens).
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_token = tl.load(
        sorted_token_ids_ptr + offs_token_id,
        mask=offs_token_id < EM,
        other=0,
    )
    token_mask = offs_token < num_valid_tokens

    # Load expert id for this block.
    off_expert = tl.load(expert_ids_ptr + pid_m)

    # Pointers into A: each row corresponds to one token × top_k slot.
    # Decode token id into (token_idx, k_idx) pair by dividing by top_k.
    # (vLLM's flat layout: A rows are sorted_token_ids; k_idx is implicit
    # in (offs_token % top_k).)
    a_row = a_ptr + (offs_token // top_k)[:, None] * stride_am
    # For top_k-weighted output, we accumulate into c[:, k_idx, :].
    k_idx = offs_token % top_k

    # Pointers into B: this expert's weight matrix, B[expert, :, :].
    b_base = b_ptr + off_expert * stride_be

    # Accumulator.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # K-loop: BLOCK_SIZE_K chunks along the K dimension.
    for k_off in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_mask = (k_off * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K
        a_ptrs = a_row + (
            k_off * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
        ) * stride_ak
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b_ptrs = b_base + (
            tl.arange(0, BLOCK_SIZE_N)[:, None] * stride_bn +
            (k_off * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))[None, :] * stride_bk
        )
        b = tl.load(b_ptrs, mask=k_mask[None, :], other=0.0)
        accumulator += tl.dot(a, b.trans())

    # Apply top-k routing weight if requested.
    if MUL_ROUTED_WEIGHT:
        # topk_weights can be 1D ([num_tokens * top_k]) or 2D
        # ([num_tokens, top_k]).  For 1D, stride_twm is the per-element
        # stride (typically 1) and stride_twk is 0.  For 2D,
        # stride_twm is the per-row stride and stride_twk is the
        # per-k stride.
        if stride_twk == 0:
            # 1D layout: index directly by offs_token.
            w = tl.load(
                topk_weights_ptr + offs_token * stride_twm,
                mask=token_mask,
                other=0,
            )  # [BLOCK_SIZE_M]
        else:
            # 2D layout: index by (offs_token // top_k, offs_token % top_k).
            w = tl.load(
                topk_weights_ptr +
                (offs_token // top_k) * stride_twm +
                (offs_token % top_k) * stride_twk,
                mask=token_mask,
                other=0,
            )  # [BLOCK_SIZE_M]
        accumulator *= w[:, None]

    # Write to C.
    c_ptrs = (
        c_ptr + (offs_token // top_k)[:, None] * stride_cm +
        k_idx[:, None] * stride_ck +
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))[None, :] * stride_cn
    )
    c_mask = token_mask[:, None] & (
        (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < N
    )[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, Any] | None = None,
    mul_routed_weight: bool = True,
) -> None:
    """Launch the fused MoE Triton kernel (FP16 path).

    Args:
        A: ``[num_tokens, K]`` activations, fp16/bf16.
        B: ``[E, N, K]`` per-expert weights, fp16/bf16.
        C: ``[num_tokens, top_k, N]`` output buffer (allocated by caller).
        topk_weights: ``[num_tokens, top_k]`` routing weights, fp32.
        sorted_token_ids: ``[EM]`` int32 from ``moe_align_block_size``.
        expert_ids: ``[num_blocks]`` int32 from ``moe_align_block_size``.
        num_tokens_post_padded: ``[1]`` int32 from ``moe_align_block_size``.
        top_k: number of experts per token (K in MoE parlance).
        config: optional dict with BLOCK_SIZE_M/N/K and GROUP_SIZE_M.
            Defaults: ``{BLOCK_SIZE_M: 64, BLOCK_SIZE_N: 64,
            BLOCK_SIZE_K: 32, GROUP_SIZE_M: 8}``.
        mul_routed_weight: whether to apply topk_weights to the
            accumulator inside the kernel (False = caller reduces).
    """
    if config is None:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
    block_size_m = config["BLOCK_SIZE_M"]
    block_size_n = config["BLOCK_SIZE_N"]
    block_size_k = config["BLOCK_SIZE_K"]
    group_size_m = config["GROUP_SIZE_M"]

    assert A.dim() == 2 and B.dim() == 3
    num_tokens, K = A.shape
    E, N, _ = B.shape

    # EM = sorted_token_ids.size(0); the kernel uses this as the
    # upper bound of (pid_m * BLOCK_SIZE_M) accesses.
    EM = sorted_token_ids.size(0)
    num_valid_tokens = num_tokens * top_k

    # Grid: (M-block, N-block).
    grid = (
        triton.cdiv(EM, block_size_m) * triton.cdiv(N, block_size_n),
    )

    fused_moe_kernel[grid](
        A, B, C,
        topk_weights if topk_weights is not None else A,  # placeholder
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        # topk_weights may be 1D (flat) or 2D ([num_tokens, top_k]).
        # Use stride(0) for the per-row stride, stride(1) for the
        # per-k stride; default to 0 for missing dims (1D tensors).
        (topk_weights.stride(0) if topk_weights is not None and
         topk_weights.dim() >= 1 else 0),
        (topk_weights.stride(1) if topk_weights is not None and
         topk_weights.dim() >= 2 else 0),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# High-level entry point — Python wrapper used by ``fused_moe.py``
# ---------------------------------------------------------------------------


def fused_moe_triton(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    block_size_m: int = 64,
) -> torch.Tensor:
    """End-to-end fused MoE: gate (w1) → SiLU → down (w2) → reduce.

    Mirrors the vLLM ``fused_experts`` flow at a high level.  Steps:
      1. ``moe_align_block_size`` to sort tokens by expert.
      2. ``invoke_fused_moe_triton_kernel`` for the gate matmul.
      3. SiLU activation (in-place on the output buffer's expert slot).
      4. Second ``invoke_fused_moe_triton_kernel`` for the down matmul.
      5. Weighted sum across top_k experts → final output.

    Args:
        hidden_states: ``[num_tokens, hidden_dim]`` fp16/bf16.
        w1: ``[E, 2 * inter_dim, hidden_dim]`` gate+up projection
            (concatenated; this is the Mixtral/DeepSeek convention).
            In this implementation we use only the first half as the
            gate (inter_dim = w1.shape[1] // 2).
        w2: ``[E, hidden_dim, inter_dim]`` down projection.
        topk_weights: ``[num_tokens, top_k]`` fp32.
        topk_ids: ``[num_tokens, top_k]`` int32.
        top_k: K (number of experts per token).
        num_experts: E.
        block_size_m: BLOCK_SIZE_M for the Triton kernel.

    Returns:
        ``[num_tokens, hidden_dim]`` output.
    """
    from my_vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    num_tokens, hidden_dim = hidden_states.shape
    inter_dim = w1.shape[1] // 2

    # Step 1: align.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size=block_size_m, num_experts=num_experts,
    )

    # Step 2: gate matmul (using first half of w1 as the gate).
    intermediate = torch.empty(
        (num_tokens, top_k, inter_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invoke_fused_moe_triton_kernel(
        A=hidden_states,
        B=w1[:, :inter_dim, :].contiguous(),
        C=intermediate,
        topk_weights=topk_weights,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=top_k,
        config={"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        # NOTE: don't apply routing weight yet — we want the raw expert
        # output so we can SiLU it before the down-projection.
        mul_routed_weight=False,
    )

    # Step 3: SiLU in place.  vLLM fuses SiLU + down-projection in
    # production (``silu_and_mul`` + MoE down); here we apply SiLU
    # per-element for simplicity.
    intermediate = torch.nn.functional.silu(intermediate)

    # Step 4: down matmul → [num_tokens, top_k, hidden_dim].
    # The intermediate is 3D [num_tokens, top_k, inter_dim]; reshape to
    # 2D [num_tokens * top_k, inter_dim] for the launcher (each
    # token × top_k slot is treated as an independent "token" by the
    # kernel since we set top_k=1 in the down-proj call below).
    intermediate_2d = intermediate.reshape(num_tokens * top_k, inter_dim)
    expert_out = torch.empty(
        (num_tokens, top_k, hidden_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # For the down projection, we treat each (token, k) as a separate
    # row in the flat layout [num_tokens * top_k, ...].  The kernel
    # expects top_k as the divisor for routing weight / output index
    # decomposition: ``offs_token // top_k`` gives the token index,
    # ``offs_token % top_k`` gives the k index.  We pass ``top_k``
    # unchanged so the kernel writes to expert_out[token, k, ...].
    from my_vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size as _align,
    )
    flat_expert_ids = topk_ids.reshape(num_tokens * top_k, 1)
    sorted_ids2, expert_ids2, num_padded2 = _align(
        flat_expert_ids, block_size=block_size_m, num_experts=num_experts,
    )
    invoke_fused_moe_triton_kernel(
        A=intermediate_2d,
        B=w2,
        C=expert_out,
        topk_weights=topk_weights,  # 2D [num_tokens, top_k]
        sorted_token_ids=sorted_ids2,
        expert_ids=expert_ids2,
        num_tokens_post_padded=num_padded2,
        top_k=top_k,  # kernel uses offs_token // top_k, offs_token % top_k
        config={"BLOCK_SIZE_M": block_size_m, "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
        # Apply routing weight here (post-SiLU, post-down).
        mul_routed_weight=True,
    )

    # Step 5: reduce over top_k.
    output = expert_out.sum(dim=1)  # [num_tokens, hidden_dim]
    return output

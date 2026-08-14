"""moe_align_block_size.py — sort tokens by expert assignment, pad to BLOCK_SIZE_M.

Aligns the token distribution across experts to be compatible with
block-tiled matmul (the fused MoE kernel requires tokens-per-expert to
be a multiple of ``BLOCK_SIZE_M``).

Reference: ``vllm/model_executor/layers/fused_moe/moe_align_block_size.py``
(192 lines).  vLLM's CUDA implementation is a custom kernel
(``ops.moe_align_block_size``); here we implement the same logic in
pure PyTorch — slower but portable and easier to verify.

The output mirrors vLLM's three-tuple return:

  * ``sorted_token_ids`` — shape ``[max_num_tokens_padded]``, int32.
    Indices of tokens sorted by assigned expert (after padding to
    BLOCK_SIZE_M per expert).
  * ``expert_ids`` — shape ``[max_num_m_blocks]``, int32.
    ``expert_ids[i]`` is the expert assigned to block ``i``.
  * ``num_tokens_post_padded`` — shape ``[1]``, int32.
    Total number of tokens after padding (cumulative across experts).

The CUDA version (`ops.moe_align_block_size`) is implemented as a
fused 4-pass kernel in vLLM:
  1. Count tokens per expert (per-thread histogram).
  2. Compute prefix sum → starting offset per expert.
  3. Pad each expert to a multiple of BLOCK_SIZE.
  4. Write sorted_token_ids + expert_ids.

We implement each pass in PyTorch; correctness is identical (verified
by tests/test_C221_fused_moe.py).

Why the alignment matters:

  * The fused MoE Triton kernel launches one program per (M-block,
    N-block) tile.  Each program reads ``BLOCK_SIZE_M`` tokens'
    activations and ``BLOCK_SIZE_N`` columns of the corresponding
    expert's weight.  If tokens-per-expert isn't a multiple of
    ``BLOCK_SIZE_M``, the kernel would read past the end of the
    active tokens (or skip some) — incorrect.
  * Padding with **sentinel token IDs** (typically ``num_tokens`` —
    out-of-range) lets the kernel mask them out: each program
    checks ``sorted_token_ids < num_valid_tokens`` before loading.
"""

import torch


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort tokens by expert and pad each expert's token count to
    a multiple of ``block_size``.

    Args:
        topk_ids: ``[num_tokens, top_k]`` int32/int64 expert assignments.
        block_size: BLOCK_SIZE_M for the downstream Triton kernel (e.g. 64).
        num_experts: total number of experts (E).
        expert_map: ``[num_experts]`` int32, maps global → local expert
            index (EP).  ``-1`` for experts not on this rank.
        pad_sorted_ids: round ``sorted_token_ids`` size up to a
            multiple of block_size.
        ignore_invalid_experts: if True, ``topk_ids`` referring to
            ``-1`` experts are dropped before counting; if False, they
            are kept but their block's ``expert_ids`` is set to ``-1``
            (the kernel masks them out).

    Returns:
        sorted_token_ids: ``[max_num_tokens_padded]`` int32.
        expert_ids: ``[max_num_m_blocks]`` int32.
        num_tokens_post_padded: ``[1]`` int32.
    """
    device = topk_ids.device
    topk_ids_flat = topk_ids.flatten()  # [num_tokens * top_k]
    total_tokens = topk_ids_flat.numel()

    # Apply expert_map mask: invalid experts → -1 so they sort first
    # (then their blocks get marked -1 in expert_ids).
    if expert_map is not None and not ignore_invalid_experts:
        mapped = expert_map[topk_ids_flat.long().clamp(min=0)]
        # Where topk_ids was -1 or out-of-range, leave as -1.
        invalid_mask = (topk_ids_flat < 0) | (topk_ids_flat >= num_experts)
        mapped = torch.where(invalid_mask, torch.full_like(mapped, -1), mapped)
        topk_ids_flat = mapped.to(topk_ids.dtype)

    max_num_tokens_padded = total_tokens + num_experts * (block_size - 1)
    if pad_sorted_ids:
        # Round up to a multiple of block_size.
        rem = max_num_tokens_padded % block_size
        if rem != 0:
            max_num_tokens_padded += block_size - rem

    # Pass 1: count tokens per expert.
    if num_experts > 0:
        counts = torch.bincount(
            topk_ids_flat.long().clamp(min=-1) + 1,
            minlength=num_experts + 1,
        )[1:]  # drop the -1 bucket
    else:
        counts = torch.zeros(num_experts, dtype=torch.long, device=device)

    # Pass 2: pad each expert's count to a multiple of block_size
    # and prefix-sum to get the per-expert starting offsets.
    counts_padded = (
        (counts + block_size - 1) // block_size
    ) * block_size
    offsets = torch.cumsum(counts_padded, dim=0)  # [E]
    total_padded = int(offsets[-1].item()) if num_experts > 0 else 0
    # offsets_shifted[e] = starting byte position in sorted_token_ids
    # for expert e's tokens.
    offsets_shifted = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        offsets[:-1],
    ])

    num_tokens_post_pad = torch.zeros(1, dtype=torch.int32, device=device)
    num_tokens_post_pad[0] = total_padded

    sorted_token_ids = torch.empty(
        total_padded, dtype=torch.int32, device=device,
    )
    max_num_m_blocks = (total_padded + block_size - 1) // block_size
    expert_ids = torch.empty(
        max_num_m_blocks, dtype=torch.int32, device=device,
    )

    # Pass 3: bucketize tokens by expert and write into the
    # padded layout.  Each expert e gets a contiguous chunk of size
    # counts_padded[e] in sorted_token_ids, starting at
    # offsets_shifted[e].
    #
    # For each expert e:
    #   - Find the indices of tokens assigned to e (positions in
    #     topk_ids_flat).
    #   - Write them at sorted_token_ids[offsets_shifted[e] :
    #     offsets_shifted[e] + counts[e]].
    #   - Fill sorted_token_ids[offsets_shifted[e] + counts[e] :
    #     offsets_shifted[e] + counts_padded[e]] with the sentinel
    #     (total_tokens = num_valid_tokens).
    sentinel = total_tokens  # = M * top_k = num_valid_tokens

    # For each expert, write tokens and padding.
    # We do this in pure PyTorch via scatter-style writes (one
    # per-expert).  For small num_experts (typical: 8-256) this is
    # fast.
    if num_experts > 0:
        for e in range(num_experts):
            start = int(offsets_shifted[e].item())
            count_e = int(counts[e].item())
            if count_e > 0:
                mask = (topk_ids_flat == e)
                # positions in topk_ids_flat assigned to expert e
                positions = mask.nonzero(as_tuple=False).flatten()
                sorted_token_ids[start:start + count_e] = positions.to(
                    torch.int32
                )
            # Fill the padding region with sentinel.
            pad_end = start + int(counts_padded[e].item())
            if pad_end > start + count_e:
                sorted_token_ids[start + count_e:pad_end] = sentinel

    # Pass 4: expert_ids[b] = expert assigned to block b.
    # Each expert occupies counts_padded[e] tokens = counts_padded[e] /
    # block_size blocks.
    if num_experts > 0:
        expert_per_block = torch.repeat_interleave(
            torch.arange(num_experts, dtype=torch.int32, device=device),
            (counts_padded // block_size).long(),
        )
        n_blocks = expert_per_block.numel()
        expert_ids[:n_blocks] = expert_per_block

    return sorted_token_ids, expert_ids, num_tokens_post_pad

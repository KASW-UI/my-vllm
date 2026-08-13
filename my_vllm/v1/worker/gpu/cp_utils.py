"""GPU-side CP (Context-Parallel) helpers.

Mirrors vLLM V1 v1/worker/gpu/cp_utils.py.

CP splits a single request's KV cache across multiple ranks
(round-robin by token, with `cp_interleave` controlling the
stride). Each rank owns a contiguous chunk of the sequence's
KV blocks; the attention kernel on each rank attends only to
the local chunk and exchanges K/V with neighbors via
all-gather / all-to-all.

vLLM's `prepare_dcp_local_seq_lens` writes the per-rank local
sequence lengths into a persistent CUDA-graph-safe buffer so
the CG replay path doesn't need a CPU sync.

C142: minimal subset — pure-PyTorch equivalent of the vLLM
Triton kernel. The Triton kernel in vLLM uses tl.constexpr and
mask-based loads to handle variable request counts under CG
capture; my-vllm's version uses a straightforward torch loop
that is CG-incompatible but matches the API surface.
"""

from __future__ import annotations

import torch


def prepare_dcp_local_seq_lens(
    dcp_local_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    dcp_size: int,
    dcp_rank: int,
    cp_interleave: int,
) -> None:
    """Populate the persistent DCP local seq_lens buffer.

    Mirrors vLLM V1 v1/worker/gpu/cp_utils.py:prepare_dcp_local_seq_lens.

    Each rank owns a slice of length `cp_interleave` within every
    `dcp_size * cp_interleave` block of the sequence (round-robin
    distribution). For a sequence of length L on rank r:
        rounds = L // (dcp_size * cp_interleave)
        remainder = L % (dcp_size * cp_interleave)
        local = rounds * cp_interleave + max(0, min(cp_interleave,
                                                    remainder - r * cp_interleave))

    C142: torch-loop equivalent (CUDA-graph-incompatible). The
    Triton kernel in vLLM is faster but requires CG capture
    infrastructure that my-vllm doesn't yet have for this helper.
    """
    if dcp_size == 1:
        return

    max_num_reqs = dcp_local_seq_lens.shape[0]
    block = dcp_size * cp_interleave

    for i in range(max_num_reqs):
        if i < num_reqs:
            seq_len = int(seq_lens[i].item())
            rounds = seq_len // block
            remainder = seq_len % block
            remainder = max(0, min(cp_interleave, remainder - dcp_rank * cp_interleave))
            local = rounds * cp_interleave + remainder
        else:
            local = 0
        dcp_local_seq_lens[i] = local


def check_attention_cp_compatibility(
    cp_size: int,
    model_architecture: str,
) -> None:
    """Validate that the model supports the configured CP size.

    Mirrors vLLM V1 v1/worker/cp_utils.py:check_attention_cp_compatibility.

    C142: stub — only checks that cp_size is a power of two, which
    is the minimum requirement for vLLM's RingAttention path.
    Model-specific checks (e.g. GQA groups must divide cp_size)
    are out of scope.
    """
    if cp_size <= 1:
        return
    if cp_size & (cp_size - 1) != 0:
        raise ValueError(
            f"cp_size must be a power of 2, got {cp_size}"
        )


__all__ = [
    "check_attention_cp_compatibility",
    "prepare_dcp_local_seq_lens",
]
"""Spec-decode utilities — pure helpers + Triton kernel stubs.

Mirrors vLLM V1 v1/spec_decode/utils.py.

C138: minimal subset. The pure helpers (`next_power_of_2`,
`PADDING_SLOT_ID`) are implemented as-is. The Triton kernels
(`eagle_step_slot_mapping_metadata_kernel`,
`eagle_step_update_slot_mapping_and_metadata`,
`eagle_prepare_inputs_padded_kernel`,
`eagle_prepare_next_token_padded_kernel`, etc.) are out of scope
— they are EAGLE-specific CUDA kernels for tree-mask attention
prep that depend on vLLM's full attention backend. Adding them
requires porting ~500 lines of `@triton.jit` code; defer to
阶段 3 spec_decode follow-up.
"""

from __future__ import annotations

# Sentinel used in EAGLE padded slot mappings (matches vLLM).
PADDING_SLOT_ID = -1


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def extend_all_queries_by_N(
    query_start_loc: list[int],
    n: int,
) -> list[int]:
    """Append N tokens to each request's query length (EAGLE helper).

    Mirrors vLLM V1 v1/spec_decode/utils.py:extend_all_queries_by_N (L273).
    Used by EAGLE to extend the per-request query lengths for the
    draft tokens. For my-vllm's subset this helper exists for shape
    alignment; the EAGLE runner that consumes it lands in a follow-up.
    """
    return [q + n for q in query_start_loc]


__all__ = [
    "PADDING_SLOT_ID",
    "extend_all_queries_by_N",
    "next_power_of_2",
]
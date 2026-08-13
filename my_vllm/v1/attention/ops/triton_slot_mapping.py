"""Triton slot_mapping kernel — batched compute_slot_mappings.

Mirrors the slot-mapping computation in
vllm/vllm/v1/worker/gpu/block_table.py:_compute_slot_mappings_kernel
(single KV cache group, no CP, no padding). Used by
GPUModelRunner via BlockTables.compute_slot_mappings() to replace the
per-req / per-token Python loop with a single Triton kernel.

Layout assumptions:
  query_start_loc: [num_reqs + 1] int32
  positions:       [num_tokens]   int64
  block_table:     [num_reqs, max_num_blocks] int32
  block_size:      constexpr (compile-time)
  slot_mapping:    [num_tokens] int64 (out)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _slot_mapping_kernel(
    query_start_loc_ptr,   # [num_reqs + 1] int32
    positions_ptr,          # [num_tokens]   int64
    block_table_ptr,        # [num_reqs, max_blocks] int32
    block_table_stride,     # int32
    slot_mapping_ptr,       # [num_tokens]   int64 (out)
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    start = tl.load(query_start_loc_ptr + batch_idx)
    end = tl.load(query_start_loc_ptr + batch_idx + 1)

    # Per-token slot mapping:
    #   block_idx = pos // block_size
    #   offset    = pos % block_size
    #   slot_id   = block_table[batch_idx, block_idx] * block_size + offset
    row_base = block_table_ptr + batch_idx * block_table_stride
    for i in range(start, end):
        pos = tl.load(positions_ptr + i).to(tl.int64)
        block_idx = pos // BLOCK_SIZE
        offset = pos % BLOCK_SIZE
        block_id = tl.load(row_base + block_idx)
        slot_id = block_id * BLOCK_SIZE + offset
        tl.store(slot_mapping_ptr + i, slot_id)


def compute_slot_mapping(
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    slot_mapping: torch.Tensor,
    num_reqs: int,
) -> torch.Tensor:
    """Compute slot_mapping[num_tokens] from per-token positions + block_table.

    Args:
        query_start_loc: [num_reqs + 1] int32 cumulative token offsets.
        positions: [num_tokens] int64 absolute per-token positions.
        block_table: [num_reqs, max_blocks] int32 logical → physical block ids.
        block_size: KV cache block size.
        slot_mapping: [num_tokens] int64 pre-allocated output (filled by kernel).
        num_reqs: number of requests in the batch.

    Returns:
        slot_mapping (first num_tokens entries).
    """
    slot_mapping.fill_(-1)
    _slot_mapping_kernel[(num_reqs,)](
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        slot_mapping,
        BLOCK_SIZE=block_size,
    )
    return slot_mapping
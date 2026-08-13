"""BlockTables — GPU tensor block table for paged KV cache.

Mirrors vLLM V1 v1/worker/gpu/block_table.py:17.

C019: replaces the old Python-list BlockTable with GPU tensor
block tables and slot_mappings.
C118: compute_slot_mappings dispatches to a Triton kernel
(inlined below as `_compute_slot_mappings_kernel`) instead of a
per-req Python loop with .item() syncs. Single KV cache group, no CP.

C131: kernel + wrapper inlined from the former
`v1/attention/ops/triton_slot_mapping.py` into this file, matching
vLLM's layout (vllm/v1/worker/gpu/block_table.py:_compute_slot_mappings_kernel
at L250).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_slot_mappings_kernel(
    query_start_loc_ptr,   # [num_reqs + 1] int32
    positions_ptr,          # [num_tokens]   int64
    block_table_ptr,        # [num_reqs, max_blocks] int32
    block_table_stride,     # int32
    slot_mapping_ptr,       # [num_tokens]   int64 (out)
    BLOCK_SIZE: tl.constexpr,
):
    """Per-token slot_mapping kernel — slot_id = block_id * BLOCK_SIZE + offset.

    Layout assumptions:
      query_start_loc: [num_reqs + 1] int32
      positions:       [num_tokens]   int64
      block_table:     [num_reqs, max_num_blocks] int32
      block_size:      constexpr (compile-time)
      slot_mapping:    [num_tokens] int64 (out)
    """
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


class BlockTables:
    """GPU tensor block tables mapping logical → physical blocks.

    Mirrors vLLM V1 BlockTables (simplified, single KV cache group,
    no CP, no staged writes).
    """

    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks: int,
        max_num_batched_tokens: int,
        device: torch.device,
    ):
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks = max_num_blocks
        self.device = device

        # [max_num_reqs, max_num_blocks] int32
        self.block_table = torch.zeros(
            max_num_reqs, max_num_blocks,
            dtype=torch.int32, device=device,
        )

        # [max_num_reqs] int32 — current block count per request slot
        self.num_blocks = torch.zeros(
            max_num_reqs, dtype=torch.int32,
        )

        # [max_num_batched_tokens] int64 — physical KV slot per token
        self.slot_mappings = torch.full(
            (max_num_batched_tokens,), -1,
            dtype=torch.int64, device=device,
        )

    def append_block_ids(
        self, req_index: int, block_ids: list[int]
    ) -> None:
        """Append block IDs to a request's block table."""
        start = int(self.num_blocks[req_index].item())
        for i, bid in enumerate(block_ids):
            self.block_table[req_index, start + i] = bid
        self.num_blocks[req_index] = start + len(block_ids)

    def compute_slot_mappings(
        self,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Fill slot_mappings for num_tokens tokens via Triton kernel.

        Mirrors vllm BlockTables.compute_slot_mappings — single KV group,
        no CP, no padding. Replaces the prior per-req Python loop with
        a single batched Triton kernel launch.

        slot_id = block_table[req][pos // block_size] * block_size
                  + (pos % block_size)
        """
        num_reqs = query_start_loc.shape[0] - 1
        self.slot_mappings.fill_(-1)
        _compute_slot_mappings_kernel[(num_reqs,)](
            query_start_loc,
            positions,
            self.block_table,
            self.block_table.stride(0),
            self.slot_mappings,
            BLOCK_SIZE=self.block_size,
        )
        return self.slot_mappings[:num_tokens]

    def clear(self):
        self.block_table.zero_()
        self.num_blocks.zero_()
        self.slot_mappings.fill_(-1)

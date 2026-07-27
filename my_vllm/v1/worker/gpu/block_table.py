"""BlockTables — GPU tensor block table for paged KV cache.

Mirrors vLLM V1 v1/worker/gpu/block_table.py:17.

C019: replaces the old Python-list BlockTable with GPU tensor
block tables and slot_mappings.  Pure PyTorch implementation
(no Triton kernels), designed for single KV cache group.
"""

import torch


class BlockTables:
    """GPU tensor block tables mapping logical → physical blocks.

    Mirrors vLLM V1 BlockTables (simplified, single KV cache group,
    no CP, no Triton, no staged writes).
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
        """Fill slot_mappings for num_tokens tokens.

        slot_id = block_table[req][pos // block_size] * block_size
                  + (pos % block_size)
        """
        self.slot_mappings.fill_(-1)
        num_reqs = query_start_loc.shape[0] - 1

        for req_idx in range(num_reqs):
            start = int(query_start_loc[req_idx].item())
            end = int(query_start_loc[req_idx + 1].item())
            for tok_idx in range(start, end):
                pos = int(positions[tok_idx].item())
                block_idx = pos // self.block_size
                if block_idx < self.max_num_blocks:
                    block_id = int(
                        self.block_table[req_idx, block_idx].item()
                    )
                    offset = pos % self.block_size
                    self.slot_mappings[tok_idx] = (
                        block_id * self.block_size + offset
                    )

        return self.slot_mappings[:num_tokens]

    def clear(self):
        self.block_table.zero_()
        self.num_blocks.zero_()
        self.slot_mappings.fill_(-1)

"""C019: tests for BlockTables GPU tensor block table."""

import torch

from my_vllm.v1.worker.gpu.block_table import BlockTables


class TestBlockTables:
    @staticmethod
    def _make_block_tables(block_size=16):
        return BlockTables(
            block_size=block_size,
            max_num_reqs=4,
            max_num_blocks=64,
            max_num_batched_tokens=256,
            device=torch.device("cpu"),
        )

    def test_init(self):
        bt = self._make_block_tables()
        assert bt.block_table.shape == (4, 64)
        assert bt.num_blocks.shape == (4,)
        assert bt.slot_mappings.shape == (256,)
        assert bt.block_size == 16

    def test_append_block_ids(self):
        bt = self._make_block_tables()
        bt.append_block_ids(0, [3, 7, 12])
        assert int(bt.num_blocks[0].item()) == 3
        assert int(bt.block_table[0, 0].item()) == 3
        assert int(bt.block_table[0, 1].item()) == 7
        assert int(bt.block_table[0, 2].item()) == 12

    def test_append_multiple_requests(self):
        bt = self._make_block_tables()
        bt.append_block_ids(0, [1, 2])
        bt.append_block_ids(1, [5, 6, 7])
        assert int(bt.num_blocks[0].item()) == 2
        assert int(bt.num_blocks[1].item()) == 3
        assert int(bt.block_table[0, 1].item()) == 2
        assert int(bt.block_table[1, 2].item()) == 7

    def test_append_second_batch_same_request(self):
        bt = self._make_block_tables()
        bt.append_block_ids(0, [10, 11])
        bt.append_block_ids(0, [12, 13])
        assert int(bt.num_blocks[0].item()) == 4
        assert int(bt.block_table[0, 3].item()) == 13

    def test_compute_slot_mappings_single_request(self):
        bt = self._make_block_tables(block_size=4)
        bt.append_block_ids(0, [3])
        # Request 0: 6 tokens → needs 2 blocks
        # block 0 [0..3] → block_id 3, slots 12..15
        # block 1 [4..5] → would need a second block_id, but we only gave 1
        query_start = torch.tensor([0, 6], dtype=torch.long)
        positions = torch.tensor(
            [0, 1, 2, 3, 4, 5], dtype=torch.long
        )
        slots = bt.compute_slot_mappings(query_start, positions, 6)
        # Token 0: pos=0 → block=0, offset=0 → slot=3*4+0=12
        assert int(slots[0].item()) == 12
        assert int(slots[3].item()) == 15

    def test_compute_slot_mappings_multi_request(self):
        bt = self._make_block_tables(block_size=4)
        bt.append_block_ids(0, [1])  # req 0: block 1
        bt.append_block_ids(1, [5])  # req 1: block 5
        # req 0: 3 tokens
        # req 1: 2 tokens
        query_start = torch.tensor([0, 3, 5], dtype=torch.long)
        positions = torch.tensor(
            [0, 1, 2, 0, 1], dtype=torch.long
        )
        slots = bt.compute_slot_mappings(query_start, positions, 5)
        # req 0, token 0: pos=0 → block=0, offset=0 → 1*4+0=4
        assert int(slots[0].item()) == 4
        # req 1, token 0: pos=0 → block=0, offset=0 → 5*4+0=20
        assert int(slots[3].item()) == 20

    def test_clear(self):
        bt = self._make_block_tables()
        bt.append_block_ids(0, [1, 2, 3])
        bt.clear()
        assert int(bt.num_blocks[0].item()) == 0
        assert int(bt.block_table[0, 0].item()) == 0
        assert int(bt.slot_mappings[0].item()) == -1

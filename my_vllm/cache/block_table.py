from my_vllm.cache.block import KVCacheBlock


class BlockTable:
    """Per-request logical-position → physical-block mapping.

    Owns the list of blocks assigned to a single request.
    num_tokens = len(blocks) * block_size tracks total capacity.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.blocks: list[KVCacheBlock] = []

    def append(self, blocks: list[KVCacheBlock]):
        self.blocks.extend(blocks)

    @property
    def block_ids(self) -> list[int]:
        return [b.block_id for b in self.blocks]

    @property
    def num_tokens(self) -> int:
        return len(self.blocks) * self.block_size

    def __len__(self) -> int:
        return len(self.blocks)

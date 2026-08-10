"""PrefixCache — Copy-on-Write protection for shared KV cache blocks.

Mirrors vLLM V1 v1/core/prefix_cache.py.
C036: ensures shared blocks are cloned when a request needs to write,
preventing KV cache pollution across requests.
"""

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy


class PrefixCache:
    """Copy-on-Write protection layer for prefix-cached blocks.

    When two requests share a prefix block (ref_cnt > 1), any write
    by one must not pollute the other's KV cache.  PrefixCache detects
    shared blocks and clones them via CoW: allocate a new block, record
    the copy, decrement the original ref_cnt.

    The GPU-side KV data copy is recorded as KVCacheBlockCopy for later
    execution by the physical cache layer (C054c).
    """

    def __init__(self, block_pool: BlockPool):
        self.block_pool = block_pool
        self._copies: list[KVCacheBlockCopy] = []

    def get_block_for_write(self, block: KVCacheBlock) -> KVCacheBlock:
        """Return a private block safe for writing.

        If the block is shared (ref_cnt > 1), allocate a new block,
        record the copy, and decrement the original's reference count.
        Otherwise return the block as-is.
        """
        if block.ref_cnt <= 1:
            return block
        new_blocks = self.block_pool.get_new_blocks(1)
        if new_blocks is None:
            return block
        new_block = new_blocks[0]
        self._copies.append(
            KVCacheBlockCopy(src_block_id=block.block_id,
                             dst_block_id=new_block.block_id)
        )
        block.ref_cnt -= 1
        return new_block

    def get_copies(self) -> list[KVCacheBlockCopy]:
        return self._copies

    def clear_copies(self) -> None:
        self._copies.clear()

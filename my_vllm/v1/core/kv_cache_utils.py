"""KV cache block and free-block queue data structures.

Mirrors vLLM V1 v1/core/kv_cache_utils.py:118-357.

C018: KVCacheBlock gains ref_cnt, block_hash, doubly-linked list
pointers; FreeKVCacheBlockQueue replaces the old _free_ids list
with O(1) LRU-aware allocation and eviction.

C120: _gen_lora_extra_hash_keys — LoRA adapter name as extra hash key
for KV cache block prefix matching, preventing cross-adapter cache sharing.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from my_vllm.v1.request import Request


class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


@dataclass
class KVCacheBlock:
    """KV-cache block metadata — mirrors vLLM V1 kv_cache_utils.py:118."""

    block_id: int
    ref_cnt: int = 0
    _block_hash: bytes | None = None
    _block_hash_num_tokens: int | None = None
    is_null: bool = False
    _is_free: bool = True

    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    def set_block_hash(
        self, block_hash: bytes, num_tokens: int | None = None
    ) -> None:
        self._block_hash = block_hash
        self._block_hash_num_tokens = num_tokens

    def reset_hash(self) -> bytes | None:
        old = self._block_hash
        self._block_hash = None
        self._block_hash_num_tokens = None
        return old

    @property
    def is_free(self) -> bool:
        """Backward compat: a block is free if _is_free flag is set."""
        return self._is_free

    def __repr__(self) -> str:
        prev_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, ref_cnt={self.ref_cnt}, "
            f"prev_free={prev_id}, next_free={next_id})"
        )


class FreeKVCacheBlockQueue:
    """Doubly-linked list of free blocks with LRU eviction order.

    Mirrors vLLM V1 kv_cache_utils.py:184.
    Uses fake head/tail sentinel blocks to eliminate branching.
    O(1) popleft, append, remove.
    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)

        if self.num_free_blocks > 0:
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block (O(1))."""
        if self.fake_free_list_head.next_free_block is self.fake_free_list_tail:
            raise ValueError("No free blocks available")

        first: KVCacheBlock = self.fake_free_list_head.next_free_block
        self.fake_free_list_head.next_free_block = first.next_free_block
        first.next_free_block.prev_free_block = self.fake_free_list_head
        first.prev_free_block = first.next_free_block = None
        self.num_free_blocks -= 1
        return first

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks (O(n))."""
        if n == 0:
            return []
        assert self.num_free_blocks >= n, (
            f"Not enough free blocks: {self.num_free_blocks} < {n}"
        )
        self.num_free_blocks -= n

        ret = []
        curr = self.fake_free_list_head.next_free_block
        for _ in range(n):
            ret.append(curr)
            last = curr
            curr = curr.next_free_block
            last.prev_free_block = last.next_free_block = None

        if curr is not None:
            self.fake_free_list_head.next_free_block = curr
            curr.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block from the free list (O(1))."""
        if block.prev_free_block is None or block.next_free_block is None:
            raise RuntimeError(f"remove() on invalid block: {block}")
        block.prev_free_block.next_free_block = block.next_free_block
        block.next_free_block.prev_free_block = block.prev_free_block
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list at the tail (O(1))."""
        last = self.fake_free_list_tail.prev_free_block
        last.next_free_block = block
        block.prev_free_block = last
        block.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = block
        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Prepend blocks to the head (eviction front, O(n))."""
        if not blocks:
            return
        for i, blk in enumerate(blocks):
            if i > 0:
                blk.prev_free_block = blocks[i - 1]
                blocks[i - 1].next_free_block = blk
            else:
                blk.prev_free_block = None
            if i < len(blocks) - 1:
                blk.next_free_block = blocks[i + 1]
            else:
                blk.next_free_block = None
        old_first = self.fake_free_list_head.next_free_block
        self.fake_free_list_head.next_free_block = blocks[0]
        blocks[0].prev_free_block = self.fake_free_list_head
        blocks[-1].next_free_block = old_first
        if old_first is not None:
            old_first.prev_free_block = blocks[-1]
        self.num_free_blocks += len(blocks)

    def clear(self) -> None:
        """Clear the free list."""
        self.fake_free_list_head.next_free_block = self.fake_free_list_tail
        self.fake_free_list_tail.prev_free_block = self.fake_free_list_head
        self.num_free_blocks = 0

    @property
    def is_empty(self) -> bool:
        return self.num_free_blocks == 0


# ---------------------------------------------------------------------------
# Extra hash keys for KV cache block prefix matching
# ---------------------------------------------------------------------------


def _gen_lora_extra_hash_keys(request: "Request") -> list[str]:
    """Generate extra keys related to LoRA for block hash computation.

    Mirrors vllm/v1/core/kv_cache_utils.py:517-529.

    Returns LoRA name of the request if it is a LoRA request, otherwise
    returns an empty list. This ensures that requests with different
    LoRA adapters do not share KV cache blocks even when their token
    prefixes match.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_name]

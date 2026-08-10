"""KVCacheManager — KV cache lifecycle manager with allocate_slots interface.

Mirrors vLLM V1 v1/core/kv_cache_manager.py.

C018: allocate_slots() replaces allocate().  Manages per-request block
tables and provides block ID lookups.  No longer creates KVCache objects
(HF past_key_values) — the GPUModelRunner handles actual KV storage.

C036: CoW protection on shared blocks — clones shared blocks (ref_cnt>1)
and records KVCacheBlockCopy for the Worker to execute.
"""

from dataclasses import dataclass

from my_vllm.v1.core.block_pool import BlockPool
from my_vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from my_vllm.v1.request import Request


@dataclass
class KVCacheBlocks:
    """Allocation result bridging Scheduler ↔ KVCacheManager.

    Mirrors vLLM V1 kv_cache_manager.py:33.
    Hides internal block representation from the Scheduler.
    """

    block_ids: list[int]

    def __len__(self) -> int:
        return len(self.block_ids)

    @classmethod
    def empty(cls) -> "KVCacheBlocks":
        return cls(block_ids=[])


class KVCacheManager:
    """KV cache lifecycle manager with allocate_slots interface.

    C018: tracks per-request block ownership via request_blocks dict.
    C036: CoW for shared blocks — allocates new block, records copy.
    """

    def __init__(
        self,
        block_pool: BlockPool | None = None,
        block_size: int = 16,
        num_blocks: int = 512,
    ):
        self.block_pool = block_pool or BlockPool(num_blocks, block_size)
        self.block_size = self.block_pool.block_size
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}
        self._pending_cow_copies: list[KVCacheBlockCopy] = []

    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> KVCacheBlocks | None:
        """Allocate blocks for num_new_tokens. Returns block IDs or None."""
        total_blocks_needed = max(
            1, (num_new_tokens + self.block_size - 1) // self.block_size
        )

        existing = self.request_blocks.get(request.request_id, [])
        current_blocks = len(existing)
        new_blocks_needed = max(0, total_blocks_needed - current_blocks)

        new_blocks: list[KVCacheBlock] = []
        if new_blocks_needed > 0:
            new_blocks = self.block_pool.get_new_blocks(new_blocks_needed)
            if new_blocks is None:
                return None

        all_blocks = existing + new_blocks
        self.request_blocks[request.request_id] = all_blocks

        return KVCacheBlocks(
            block_ids=[b.block_id for b in all_blocks]
        )

    def get_computed_blocks(
        self, request: Request
    ) -> tuple[KVCacheBlocks, int]:
        """Find consecutive prefix cache hits for a request.

        C029: walks request.block_hashes from the start. C036: CoW —
        bumps ref_cnt on reuse, clones shared blocks (ref_cnt>1) to
        prevent cross-request KV cache pollution, and records KVCacheBlockCopy.
        """
        if not self.block_pool.enable_caching:
            return KVCacheBlocks.empty(), 0

        block_size = self.block_pool.block_size
        request.compute_block_hashes(block_size)
        cached_blocks: list[KVCacheBlock] = []
        num_computed = 0

        for block_hash in request.block_hashes:
            if not block_hash:
                break
            cached = self.block_pool.get_cached_block(block_hash)
            if cached is None:
                break
            cached.ref_cnt += 1
            private = self._apply_cow(cached)
            cached_blocks.append(private)
            num_computed += min(
                block_size,
                len(request.prompt_token_ids) - num_computed,
            )

        if cached_blocks:
            self.request_blocks[request.request_id] = cached_blocks
            return KVCacheBlocks(
                block_ids=[b.block_id for b in cached_blocks]
            ), num_computed

        return KVCacheBlocks.empty(), 0

    def _apply_cow(self, block: KVCacheBlock) -> KVCacheBlock:
        """Copy-on-write: clone shared block so writes don't pollute.

        Aligned with vLLM SingleTypeKVCacheManager._apply_cow pattern.
        """
        if block.ref_cnt <= 1:
            return block
        new_blocks = self.block_pool.get_new_blocks(1)
        if new_blocks is None:
            return block
        cow_block = new_blocks[0]
        self._pending_cow_copies.append(
            KVCacheBlockCopy(
                src_block_id=block.block_id,
                dst_block_id=cow_block.block_id,
            )
        )
        block.ref_cnt -= 1
        return cow_block

    def take_kv_cache_block_copies(
        self,
    ) -> tuple[list[KVCacheBlockCopy], list[KVCacheBlock]]:
        """Drain pending CoW copies for the Worker to execute.

        Aligned with vLLM KVCacheManager.take_kv_cache_block_copies.
        Returns copies and the list of blocks to retain.
        """
        copies = self._pending_cow_copies
        self._pending_cow_copies = []
        retained = []
        for c in copies:
            retained.append(self.block_pool.blocks[c.src_block_id])
            retained.append(self.block_pool.blocks[c.dst_block_id])
        return copies, retained

    def get_block_ids(self, request_id: str) -> list[int]:
        bt = self.request_blocks.get(request_id, [])
        return [b.block_id for b in bt]

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.request_blocks.get(request_id, [])

    def free(self, request_id: str):
        bt = self.request_blocks.pop(request_id, None)
        if bt:
            self.block_pool.free_blocks(bt)

    @property
    def num_free_blocks(self) -> int:
        return self.block_pool.num_free

    # --- backward-compat methods (C014 tests) ---

    def allocate(self, request: Request) -> bool:
        blocks = self.allocate_slots(request, len(request.prompt_token_ids))
        return blocks is not None

    def get_cache(self, request_id: str):
        blocks = self.request_blocks.get(request_id)
        return blocks if blocks else None

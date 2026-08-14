"""KV offload CPU backend — cache policies.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/policies/{base,lru,arc}.py.

C256 ships:
- `BlockStatus` (ctypes.Structure) — per-block ref_cnt +
  block_id. ref_cnt = -1 marks a "not yet ready" block
  (write-pending).
- `CachePolicy` ABC — encapsulates both block organization +
  replacement decisions. LRU + ARC are the two implementations.
- `LRUCachePolicy` — real impl using `OrderedDict` for O(1)
  LRU eviction.
- `ARCCachePolicy` — shell (declared for registry completeness).
  ARC's ghost-list + adaptive `target_t1_size` math is out of
  scope for my-vllm's JD-aligned subset.

The base abstraction is faithful to vLLM V1's interface
(`get` / `insert` / `remove` / `touch` / `evict` / `clear` /
`mark_evictable` / `mark_non_evictable`).
"""

from my_vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from my_vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from my_vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy


__all__ = [
    "ARCCachePolicy",
    "BlockStatus",
    "CachePolicy",
    "LRUCachePolicy",
]
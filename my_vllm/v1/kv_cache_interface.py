"""KVCacheSpec — architecture boundary between Attention and KVCacheManager.

Mirrors vLLM V1 v1/kv_cache_interface.py.

C018: minimal definition establishing the interface contract.
Full KV cache config resolution added in C022 (Attention Backend).
"""

from dataclasses import dataclass


@dataclass
class KVCacheSpec:
    """Specification for a single KV cache layer group.

    Mirrors vLLM V1 kv_cache_interface.py FullAttentionSpec.
    C018 minimal: only the essential fields needed to configure
    the BlockPool.
    """

    block_size: int
    num_kv_heads: int
    head_size: int
    dtype: str = "float16"

    @property
    def page_size_bytes(self) -> int:
        """Total bytes per block for this spec (K + V)."""
        element_size = 2  # float16/bfloat16 = 2 bytes
        return (
            2  # K + V
            * self.num_kv_heads
            * self.block_size
            * self.head_size
            * element_size
        )

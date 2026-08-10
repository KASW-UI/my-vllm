"""CacheConfig — aligns with vLLM vllm/config/cache.py (simplified)."""

from dataclasses import dataclass


@dataclass
class CacheConfig:
    block_size: int = 16
    num_kv_blocks: int = 512
    enable_prefix_caching: bool = True

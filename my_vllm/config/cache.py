"""CacheConfig — aligns with vLLM vllm/config/cache.py (simplified)."""

from dataclasses import dataclass


@dataclass
class CacheConfig:
    block_size: int = 16
    num_kv_blocks: int = 512
    enable_prefix_caching: bool = True
    # C149: KV cache storage dtype. One of
    #   "auto" (default; mirrors model dtype)
    #   "fp8_e4m3" / "fp8_e5m2" / "fp8" (per-tensor FP8 quantization)
    # When set to an FP8 string, the engine allocates the KV
    # cache in `torch.float8_e4m3fn` (half the memory of BF16)
    # and the reshape_and_cache kernel applies the per-tensor
    # K/V scale before the FP8 store.
    kv_cache_dtype: str = "auto"

"""KVQuantMode — stub mirroring vllm/v1/kv_cache_interface.py:KVQuantMode."""

from enum import IntEnum


class KVQuantMode(IntEnum):
    """KV cache quantization mode enum.

    Mirrors vllm/v1/kv_cache_interface.py.
    """
    NONE = 0
    FP8_TENSOR = 1
    INT8_PER_TOKEN_HEAD = 2
    FP8_PER_TOKEN_HEAD = 3
    INT4_PER_TOKEN_HEAD = 4

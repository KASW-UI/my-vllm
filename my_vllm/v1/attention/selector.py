"""Attention backend selector — mirrors vLLM V1 v1/attention/selector.py.

Default backend is TritonAttentionBackend (real vLLM Triton kernels:
triton_reshape_and_cache_flash + unified_attention). FlashAttentionBackend
(SDPA-based) is kept as a fallback; the selector returns Triton on
non-CPU platforms.
"""

from my_vllm.v1.attention.backend import AttentionBackend
from my_vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


def get_attn_backend() -> type[AttentionBackend]:
    return TritonAttentionBackend

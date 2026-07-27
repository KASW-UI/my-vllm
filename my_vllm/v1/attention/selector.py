"""Attention backend selector — mirrors vLLM V1 v1/attention/selector.py.

C022: simplified selector that returns TorchAttentionBackend.
"""

from my_vllm.v1.attention.backend import AttentionBackend
from my_vllm.v1.attention.backends.torch_attn import TorchAttentionBackend


def get_attn_backend() -> type[AttentionBackend]:
    """Select the attention backend.

    C022: always returns TorchAttentionBackend (no FlashAttention dependency).
    """
    return TorchAttentionBackend

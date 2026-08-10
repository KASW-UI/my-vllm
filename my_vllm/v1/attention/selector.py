"""Attention backend selector — mirrors vLLM V1 v1/attention/selector.py."""

from my_vllm.v1.attention.backend import AttentionBackend
from my_vllm.v1.attention.backends.flash_attn import FlashAttentionBackend


def get_attn_backend() -> type[AttentionBackend]:
    return FlashAttentionBackend

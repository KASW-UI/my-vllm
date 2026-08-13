"""FlashAttention backend — mirrors vLLM V1 v1/attention/backends/flash_attn.py.

Uses PyTorch SDPA as the attention kernel with paged KV cache
interface (block_table + slot_mapping). Both prefill and decode
paths read/write KV cache through the backend.
"""

import torch
import torch.nn.functional as F

from my_vllm.v1.attention.backend import AttentionBackend, AttentionImpl
from my_vllm.v1.attention.metadata import CommonAttentionMetadata
from my_vllm.v1.attention.backends.ops.paged_attn import (
    gather_cached_kv_pytorch,
    reshape_and_cache_pytorch,
)


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, num_kv_heads, block_size, head_size)


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        block_table = attn_metadata.block_table_tensor
        seq_lens = attn_metadata.seq_lens
        slot_mapping = attn_metadata.slot_mapping
        max_query_len = attn_metadata.max_query_len

        block_size = _get_block_size(kv_cache)

        # Step 1: Write new K/V to paged cache
        if isinstance(kv_cache, tuple):
            k_cache, v_cache = kv_cache
        elif kv_cache.numel() > 0:
            k_cache, v_cache = _split_combined_kv_cache(kv_cache, self.head_size)
        else:
            k_cache, v_cache = None, None

        if k_cache is not None and v_cache is not None and slot_mapping is not None:
            reshape_and_cache_pytorch(
                key, value, k_cache, v_cache, slot_mapping[:num_tokens],
            )

        # Step 2: Build full K/V context and compute attention
        is_decode = max_query_len == 1

        if is_decode and seq_lens.numel() > 0 and k_cache is not None:
            history_k, history_v = gather_cached_kv_pytorch(
                k_cache, v_cache,
                block_table, seq_lens, block_size,
            )
            key_full, value_full = _concat_history_and_new(
                history_k, history_v, key, value,
            )
            output_tensor = _compute_decode_attention(
                query, key_full, value_full, seq_lens,
                self.num_heads, self.num_kv_heads, self.scale,
            )
        else:
            # Prefill: use vanilla SDPA with causal mask
            q = query.transpose(0, 1).unsqueeze(0)
            k = key.transpose(0, 1).unsqueeze(0)
            v = value.transpose(0, 1).unsqueeze(0)

            if self.num_kv_heads != self.num_heads:
                ratio = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(ratio, dim=1)
                v = v.repeat_interleave(ratio, dim=1)

            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                scale=self.scale,
            )
            output_tensor = out.squeeze(0).transpose(0, 1)

        if output.shape != output_tensor.shape:
            output_tensor = output_tensor.reshape(output.shape)
        output.copy_(output_tensor)
        return output


def _get_block_size(kv_cache) -> int:
    if isinstance(kv_cache, tuple):
        return kv_cache[0].shape[2]
    if kv_cache.numel() > 0:
        return kv_cache.shape[2]
    return 16


def _split_combined_kv_cache(
    kv_cache: torch.Tensor, head_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    last_dim = kv_cache.shape[-1]
    if last_dim == head_size:
        return kv_cache, None
    k = kv_cache[..., :head_size]
    v = kv_cache[..., head_size:]
    return k, v


def _concat_history_and_new(
    history_k: torch.Tensor,
    history_v: torch.Tensor,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate cached history KV with new KV for decode.
    
    history_k: [num_reqs, num_kv_heads, max_seq_len, head_dim]
    new_k: [num_reqs, num_kv_heads, head_dim]
    Returns: [num_reqs*2, num_heads, max_seq_len+1, head_dim] — per-request KV pairs
    """
    # Concatenate new token
    # new_k: [num_reqs, num_kv_heads, head_dim] → [num_reqs, num_kv_heads, 1, head_dim]
    new_k_expanded = new_k.unsqueeze(2)
    new_v_expanded = new_v.unsqueeze(2)
    full_k = torch.cat([history_k, new_k_expanded], dim=2)
    full_v = torch.cat([history_v, new_v_expanded], dim=2)
    return full_k, full_v


def _compute_decode_attention(
    query: torch.Tensor,
    key_full: torch.Tensor,
    value_full: torch.Tensor,
    seq_lens: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    """Compute attention for decode step with per-request KV isolation.
    
    query: [num_reqs, num_heads, head_dim]
    key_full: [num_reqs, num_kv_heads, seq_len+1, head_dim]
    Returns: [num_reqs, num_heads, head_dim]
    """
    num_reqs = query.shape[0]
    ratio = num_heads // num_kv_heads if num_kv_heads > 0 else 1

    if num_kv_heads != num_heads:
        key_full = key_full.repeat_interleave(ratio, dim=1)
        value_full = value_full.repeat_interleave(ratio, dim=1)

    outputs = []
    for r in range(num_reqs):
        slen = int(seq_lens[r].item())
        q_r = query[r:r+1]  # [1, num_heads, head_dim]
        k_r = key_full[r:r+1, :, :slen, :]  # [1, num_heads, slen, head_dim]
        v_r = value_full[r:r+1, :, :slen, :]

        out_r = F.scaled_dot_product_attention(
            q_r.unsqueeze(2),  # [1, num_heads, 1, head_dim]
            k_r,               # [1, num_heads, slen, head_dim]
            v_r,               # [1, num_heads, slen, head_dim]
            scale=scale,
        )
        outputs.append(out_r.squeeze(2))  # [1, num_heads, head_dim]

    return torch.cat(outputs, dim=0)

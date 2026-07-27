"""PagedAttention — pure PyTorch implementation.

Mirrors vLLM V1 v1/attention/ops/paged_attn.py but uses
PyTorch operations instead of CUDA custom kernels.

C023: replaces HF past_key_values with physical KV cache pool
using block table-driven gather/scatter.

Key operations:
  reshape_and_cache_pytorch — write K,V into physical KV cache
  gather_cached_kv_pytorch   — gather K,V from physical KV cache
"""

import torch


def reshape_and_cache_pytorch(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Write key/value into physical KV cache at slot positions.

    Args:
        key:   [num_tokens, num_kv_heads, head_dim]
        value: [num_tokens, num_kv_heads, head_dim]
        key_cache:   [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        slot_mapping: [num_tokens] int — slot_id = block_id*block_size + offset
    """
    block_size = key_cache.shape[2]
    valid = slot_mapping >= 0
    if not valid.any():
        return

    valid_slots = slot_mapping[valid]
    valid_keys = key[valid]
    valid_values = value[valid]

    block_ids = valid_slots // block_size
    offsets = valid_slots % block_size

    key_cache[block_ids, :, offsets, :] = valid_keys
    value_cache[block_ids, :, offsets, :] = valid_values


def gather_cached_kv_pytorch(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V for each request from physical KV cache.

    Args:
        key_cache:   [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        block_table: [num_reqs, max_num_blocks] int32
        seq_lens:    [num_reqs] int32 — actual sequence length per request
        block_size:  tokens per block

    Returns:
        keys:   [num_reqs, num_kv_heads, max_seq_len, head_dim] padded
        values: [num_reqs, num_kv_heads, max_seq_len, head_dim] padded
    """
    num_reqs = block_table.shape[0]
    num_kv_heads = key_cache.shape[1]
    head_dim = key_cache.shape[3]

    max_seq_len = int(seq_lens.max().item()) if num_reqs > 0 else 0
    if max_seq_len == 0:
        empty = torch.zeros(num_reqs, num_kv_heads, 0, head_dim,
                            dtype=key_cache.dtype, device=key_cache.device)
        return empty, empty

    keys = torch.zeros(num_reqs, num_kv_heads, max_seq_len, head_dim,
                       dtype=key_cache.dtype, device=key_cache.device)
    values = torch.zeros(num_reqs, num_kv_heads, max_seq_len, head_dim,
                         dtype=key_cache.dtype, device=key_cache.device)

    for req_idx in range(num_reqs):
        slen = int(seq_lens[req_idx].item())
        if slen == 0:
            continue
        num_required_blocks = (slen + block_size - 1) // block_size
        block_ids = block_table[req_idx, :num_required_blocks].long()

        block_k = key_cache[block_ids]    # [num_blocks, num_kv_heads, block_size, head_dim]
        block_v = value_cache[block_ids]

        k_flat = block_k.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
        v_flat = block_v.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)

        keys[req_idx, :, :slen, :] = k_flat[:, :slen, :]
        values[req_idx, :, :slen, :] = v_flat[:, :slen, :]

    return keys, values


def paged_attention_forward_pytorch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
) -> torch.Tensor:
    """Full paged attention forward: write K/V, gather history, attend.

    Args:
        query: [num_tokens, num_heads, head_dim] — current step Q
        key:   [num_tokens, num_kv_heads, head_dim] — current step K
        value: [num_tokens, num_kv_heads, head_dim] — current step V
        key_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        slot_mapping: [num_tokens] int64
        block_table: [num_reqs, max_num_blocks]
        seq_lens: [num_reqs] int32
        scale: attention scale factor
        num_heads: number of query heads
        num_kv_heads: number of KV heads
        block_size: block size

    Returns:
        output: [num_tokens, num_heads, head_dim]
    """
    # 1. Write current K/V into physical cache
    reshape_and_cache_pytorch(key, value, key_cache, value_cache, slot_mapping)

    # 2. Gather full K/V history from physical cache
    hist_keys, hist_values = gather_cached_kv_pytorch(
        key_cache, value_cache, block_table, seq_lens, block_size,
    )
    # hist_keys: [num_reqs, num_kv_heads, max_seq_len, head_dim]

    # 3. Build per-token K/V context
    # For ragged batch, each token belongs to a request at a certain position.
    # We build view of [num_tokens, num_heads/kv_heads, max_seq_len, head_dim]
    # by repeating the per-request K/V for each token in that request.

    num_reqs = block_table.shape[0]
    max_seq_len = hist_keys.shape[2]

    # Compute which request each token belongs to
    req_indices = torch.zeros(query.shape[0], dtype=torch.long,
                              device=query.device)
    cumsum = torch.cat([
        torch.zeros(1, dtype=torch.long, device=seq_lens.device),
        torch.cumsum(seq_lens, dim=0),
    ])
    for i in range(num_reqs):
        req_indices[cumsum[i]:cumsum[i + 1]] = i

    # Gather per-token K/V
    k_per_token = hist_keys[req_indices]  # [num_tokens, num_kv_heads, max_seq_len, head_dim]
    v_per_token = hist_values[req_indices]

    # GQA support: expand KV to match Q heads
    if num_kv_heads != num_heads:
        ratio = num_heads // num_kv_heads
        k_per_token = k_per_token.repeat_interleave(ratio, dim=1)
        v_per_token = v_per_token.repeat_interleave(ratio, dim=1)

    # 4. SDPA attention
    # query: [num_tokens, num_heads, head_dim] → [1, num_tokens, num_heads, head_dim]
    q = query.unsqueeze(0)  # [1, num_tokens, num_heads, head_dim]
    k = k_per_token          # [num_tokens, num_heads, max_seq_len, head_dim]
    v = v_per_token          # [num_tokens, num_heads, max_seq_len, head_dim]

    # SDPA expects: [batch, heads, seq_q, dim] for q, [batch, heads, seq_kv, dim] for k,v
    # We have: [num_tokens, heads, seq_kv, dim] which is already batch-expanded.
    # For SDPA batch mode, we need [N, H, Lq, D] and [N, H, Lkv, D]
    q_sdpa = q.permute(0, 2, 1, 3)  # [1, num_heads, num_tokens, head_dim]
    k_sdpa = k.unsqueeze(0)          # [1, num_tokens, num_heads, max_seq_len, head_dim]
    v_sdpa = v.unsqueeze(0)
    q_sdpa = q_sdpa.reshape(-1, num_heads, 1, head_dim)  # [num_tokens, heads, 1, dim]
    k_sdpa = k_sdpa.reshape(-1, num_heads, max_seq_len, head_dim)
    v_sdpa = v_sdpa.reshape(-1, num_heads, max_seq_len, head_dim)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )
    # [num_tokens, num_heads, 1, head_dim] → [num_tokens, num_heads, head_dim]
    result = out.squeeze(2)
    return result

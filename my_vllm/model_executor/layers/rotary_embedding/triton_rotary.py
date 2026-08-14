"""Triton fused rotary embedding kernel.

Mirrors the vLLM CUDA kernel in
``vllm/csrc/libtorch_stable/pos_encoding_kernels.cu``
``vllm::rotary_embedding_kernel``.

Algorithm: Apply per-token rotary position embedding to Q and K
tensors in-place.

  * ``positions`` [num_tokens] int64 — the absolute position of each
    token (the unique sequence position, not the in-batch slot).
  * ``query`` [num_tokens, num_heads, head_size] (or
    [num_tokens, num_heads * head_size]) — in-place rotated.
  * ``key`` [num_tokens, num_kv_heads, head_size] (or flat) —
    in-place rotated (or None for non-cross-layer setups).
  * ``cos_sin_cache`` [max_position, rot_dim] float — precomputed
    ``[cos, sin]`` cached per absolute position; the kernel
    indexes this by position.
  * ``rot_dim`` ≤ head_size — only the first ``rot_dim`` elements
    are rotated; the trailing ``head_size - rot_dim`` elements
    pass through unchanged.
  * ``is_neox`` controls the half-split convention:
      * Neox (default for Llama / Mistral / Qwen): x = lower half,
        y = upper half. ``[..., :embed_dim]`` & ``[..., embed_dim:rot_dim]``.
      * GPT-J: x and y interleave in pairs.

Each program (one program = one CTA) processes one (token, head)
pair, holding the cos/sin pair in registers and applying the
rotation in place.  ``BLOCK_R`` (default = rot_dim / 2) threads
cooperatively load ``rot_dim / 2`` half-elements and the same
number of cos / sin values.  Grid = ``(num_tokens, num_heads)``.

For ``key`` rotation we use a separate ``(num_tokens, num_kv_heads)``
grid and the same kernel.

Constraints / simplifications vs vLLM:

  * HEAD_SIZE / ROT_DIM are constexpr per launch (kernel
    autotuned downstream by the wrapper if needed).
  * ``inverse=False`` only (no inverse-rotation kernel — that's
    a separate compute pass).
  * CPU fallback is provided by the PyTorch path in
    ``base.py:RotaryEmbedding.forward_native``; the
    ``rotary_embedding()`` dispatch in ``_custom_ops.py`` routes
    to Triton only when ``query.is_cuda``.
  * No partial-rotary-factor slicing (rot_dim == head_size for
    all currently-supported configurations).  Partial slicing
    would require a second pass over the cache.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_embedding_token_head_kernel(
    positions_ptr,
    query_ptr,
    key_ptr,  # 0 if key is None
    cos_sin_cache_ptr,
    num_tokens,
    HEAD_SIZE: tl.constexpr,
    ROT_DIM: tl.constexpr,
    EMBED_DIM: tl.constexpr,  # ROT_DIM // 2
    QUERY_STRIDE_TOKEN: tl.constexpr,
    QUERY_STRIDE_HEAD: tl.constexpr,
    KEY_STRIDE_TOKEN: tl.constexpr,
    KEY_STRIDE_HEAD: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HAS_KEY: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    """One CTA per (token, head) — rotate ``rot_dim`` of the head.

    Layout supported:

      Q: [num_tokens, num_heads, head_size]  (per-token, per-head)
      K: [num_tokens, num_kv_heads, head_size]

    with ``query_stride_head == head_size`` (contiguous heads) and
    ``query_stride_token == num_heads * head_size`` (contiguous
    tokens).  Strided layouts are not supported by the kernel —
    callers must reshape.

    Grid: (num_tokens, num_heads)
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    # Load position once and broadcast to all lanes (no perf win
    # to gate on position; the load is one int64 per CTA).
    pos = tl.load(positions_ptr + token_idx)

    # Cache layout: [max_position, rot_dim]; cos at [pos, :embed_dim],
    # sin at [pos, embed_dim:rot_dim].
    cos_base = pos * (2 * EMBED_DIM) + tl.arange(0, EMBED_DIM)
    cos = tl.load(cos_sin_cache_ptr + cos_base).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + cos_base + EMBED_DIM).to(tl.float32)

    # Address of this head's data:
    #     base = token * query_stride_token + head * query_stride_head
    q_base = (
        token_idx * QUERY_STRIDE_TOKEN
        + head_idx * QUERY_STRIDE_HEAD
    )

    # Offset within the head: lower half (offsets 0..embed_dim) and
    # upper half (offsets embed_dim..rot_dim) for Neox; interleaved
    # for GPT-J.
    if IS_NEOX:
        # Neox: x = lower half, y = upper half.
        x_off = q_base + tl.arange(0, EMBED_DIM)
        y_off = q_base + EMBED_DIM + tl.arange(0, EMBED_DIM)

        x = tl.load(query_ptr + x_off).to(tl.float32)
        y = tl.load(query_ptr + y_off).to(tl.float32)

        new_x = x * cos - y * sin
        new_y = y * cos + x * sin

        tl.store(query_ptr + x_off, new_x.to(tl.float32))
        tl.store(query_ptr + y_off, new_y.to(tl.float32))
    else:
        # GPT-J: x = even indices, y = odd indices of the first
        # rot_dim elements.  Each thread-pair (i, i+embed_dim) handles
        # one rotation pair.
        offs = tl.arange(0, EMBED_DIM)
        x_off = q_base + 2 * offs
        y_off = q_base + 2 * offs + 1

        x = tl.load(query_ptr + x_off).to(tl.float32)
        y = tl.load(query_ptr + y_off).to(tl.float32)

        # Single cos/sin shared across both indices in the pair.
        new_x = x * cos - y * sin
        new_y = y * cos + x * sin

        tl.store(query_ptr + x_off, new_x.to(tl.float32))
        tl.store(query_ptr + y_off, new_y.to(tl.float32))

    if HAS_KEY:
        # Same operation on the key head.  For GQA/MQA, multiple
        # query heads share the same K head — num_kv_heads may
        # be < num_heads.  We re-launch this kernel over
        # (num_tokens, num_kv_heads).
        if head_idx < NUM_KV_HEADS:
            k_base = (
                token_idx * KEY_STRIDE_TOKEN
                + head_idx * KEY_STRIDE_HEAD
            )
            if IS_NEOX:
                x_off = k_base + tl.arange(0, EMBED_DIM)
                y_off = k_base + EMBED_DIM + tl.arange(0, EMBED_DIM)
            else:
                offs = tl.arange(0, EMBED_DIM)
                x_off = k_base + 2 * offs
                y_off = k_base + 2 * offs + 1

            x = tl.load(key_ptr + x_off).to(tl.float32)
            y = tl.load(key_ptr + y_off).to(tl.float32)
            new_x = x * cos - y * sin
            new_y = y * cos + x * sin
            tl.store(key_ptr + x_off, new_x.to(tl.float32))
            tl.store(key_ptr + y_off, new_y.to(tl.float32))


def _next_power_of_2(n: int) -> int:
    """Smallest power of 2 >= n."""
    if n < 1:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def rotary_embedding_triton(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int | None = None,
) -> None:
    """In-place fused rotary embedding on CUDA via Triton.

    Args mirror the vLLM ``ops.rotary_embedding`` signature
    (``vllm/_custom_ops.py``) so the dispatch in ``base.py`` is a
    thin wrapper.

    Args:
        positions: [num_tokens] int64 token positions.
        query: [num_tokens, num_heads, head_size] (rotated in-place)
            OR [num_tokens, num_heads * head_size] (must be
            contiguous; we reshape internally).
        key: same shape as query but with num_kv_heads heads (or
            None).
        head_size: per-head dimension.
        cos_sin_cache: [max_position, rot_dim] precomputed
            ``[cos, sin]`` table.
        is_neox: True for Neox-style split (Llama/Mistral/Qwen),
            False for GPT-J-style.
        rot_dim: rotation dimension (default = head_size).  If
            ``rot_dim < head_size``, only the first ``rot_dim``
            elements are rotated (the rest pass through).
    """
    if not query.is_cuda:
        raise RuntimeError(
            "rotary_embedding_triton requires a CUDA query tensor; "
            "use the PyTorch fallback (RotaryEmbedding.forward_native) "
            "on CPU."
        )
    if rot_dim is None:
        rot_dim = head_size
    if rot_dim > head_size:
        raise ValueError(
            f"rot_dim ({rot_dim}) must be <= head_size ({head_size})"
        )

    num_tokens = positions.shape[0]
    # Support both flat (num_tokens, hidden) and 3D
    # (num_tokens, num_heads, head_size).  We require the last dim
    # to be head_size contiguous; we reshape accordingly.
    query_3d = query.view(num_tokens, -1, head_size)
    num_heads = query_3d.shape[1]
    if key is not None:
        key_3d = key.view(num_tokens, -1, head_size)
        num_kv_heads = key_3d.shape[1]
    else:
        key_3d = None
        num_kv_heads = num_heads

    embed_dim = rot_dim // 2
    grid = (num_tokens, num_heads)
    _rotary_embedding_token_head_kernel[grid](
        positions,
        query_3d,
        key_3d if key is not None else query_3d,  # placeholder
        cos_sin_cache,
        num_tokens,
        HEAD_SIZE=head_size,
        ROT_DIM=rot_dim,
        EMBED_DIM=embed_dim,
        QUERY_STRIDE_TOKEN=num_heads * head_size,
        QUERY_STRIDE_HEAD=head_size,
        KEY_STRIDE_TOKEN=num_kv_heads * head_size,
        KEY_STRIDE_HEAD=head_size,
        NUM_KV_HEADS=num_kv_heads,
        HAS_KEY=key is not None,
        IS_NEOX=is_neox,
    )


__all__ = ["rotary_embedding_triton"]
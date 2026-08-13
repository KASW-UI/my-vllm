"""LoRA ops — pure-torch forward (Dense Linear path).

Mirrors vLLM V1 lora/ops/torch_ops/lora_ops.py.

vLLM's `lora_ops.py` provides SGMV (segmented matrix-vector) and
BGMV (batched matrix-vector) operations for batched LoRA forward
with per-request adapter routing. The SGMV path uses
`torch.repeat_interleave` + scatter to route each request to its
adapter; BGMV does the same without segmentation.

C144: minimal subset — `lora_forward_dense(...)` covers the
Dense Linear path: `y = x @ W^T + scale * (x @ A^T) @ B^T`.
The vLLM SGMV/BGMV ops are out of scope (they require per-
request adapter dispatch via `lora_indices_tensor`, which
my-vllm's scheduler doesn't yet emit).
"""

from __future__ import annotations

import torch


def lora_forward_dense(
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    base_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute `y = x @ W^T + b + scale * (x @ A^T) @ B^T` for one adapter.

    Args:
        x: input tensor of shape `[num_tokens, in_features]`.
        base_weight: dense linear weight, `[out_features, in_features]`.
        lora_a: LoRA A matrix, `[rank, in_features]`.
        lora_b: LoRA B matrix, `[out_features, rank]`.
        scaling: `lora_alpha / lora_rank` (the standard LoRA scale).
        base_bias: optional dense bias, `[out_features]`.

    Returns:
        `y` of shape `[num_tokens, out_features]`.
    """
    # Base linear forward.
    out = torch.nn.functional.linear(x, base_weight, base_bias)

    # LoRA delta: x @ A^T → [num_tokens, rank]; @ B^T → [num_tokens, out].
    lora_delta = (x @ lora_a.T) @ lora_b.T
    return out + scaling * lora_delta


def lora_forward_dense_inplace(
    out: torch.Tensor,
    x: torch.Tensor,
    base_weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
    base_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the LoRA-forward result in-place on a pre-allocated `out`.

    Mirrors vLLM's `bgmv_expand` with `add_inputs=True` — the LoRA
    delta is *added* to a buffer that already holds the base linear's
    output. Avoids the extra allocation of `lora_forward_dense`.

    Args:
        out: pre-allocated output buffer of shape
            `[num_tokens, out_features]` (typically holding the
            result of `torch.nn.functional.linear(...)`).
        other args: same as `lora_forward_dense`.

    Returns:
        The same `out` tensor, modified in place.
    """
    if base_bias is not None:
        out = torch.nn.functional.linear(x, base_weight, base_bias)
    lora_delta = (x @ lora_a.T) @ lora_b.T
    out.add_(scaling * lora_delta)
    return out


__all__ = [
    "lora_forward_dense",
    "lora_forward_dense_inplace",
]
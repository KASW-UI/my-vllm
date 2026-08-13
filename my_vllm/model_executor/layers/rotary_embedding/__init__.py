"""Rotary Positional Embeddings — aligns with vLLM vllm/model_executor/layers/rotary_embedding/."""

from typing import Any

import torch

from .base import RotaryEmbedding

_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """Get a RotaryEmbedding instance — aligns with vLLM get_rope()."""
    if dtype is None:
        dtype = torch.get_default_dtype()

    if rope_parameters is not None:
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    dual_chunk_attention_args = None
    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")

    if rotary_dim := rope_parameters.get("rope_dim", None):
        pass
    else:
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
            raise ValueError(
                f"{partial_rotary_factor=} must be between 0.0 and 1.0"
            )
        rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dual_chunk_attention_args,
    )

    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if scaling_type == "default":
        rotary_emb = RotaryEmbedding(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unsupported rope_type: {scaling_type!r}. "
            f"Only 'default' is supported in my-vllm."
        )

    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

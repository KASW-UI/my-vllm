"""Forward context — per-forward-pass metadata injection.

Aligns with vLLM vllm/forward_context.py core pattern. A global ForwardContext
is set before model forward via set_forward_context(), and layers read
attention metadata from it via get_forward_context().

The hot path passes through this single object: attn_metadata per layer,
slot_mapping per layer, optional cudagraph_runtime_mode + batch_descriptor
for the CUDA-graph path (C119+).

Engineering details skipped (principled scope reduction):
- DPMetadata, UBatchSlices (no DP/Microbatching in my-vllm basic runtime)
- batchsize_forward_time tracking (no logging infrastructure)
- coordinate_batch_across_dp (no DP)
- static_forward_context / no_compile_layers (no torch.compile)
- all_moe_layers / moe_layer_index (no MoE custom ops in compiled graph)
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class CUDAGraphMode(Enum):
    """CUDA graph runtime mode — subset of vLLM CUDAGraphMode (NONE/PIECEWISE/FULL).

    Mirrors vLLM vllm/config/compilation.py:CUDAGraphMode runtime subset.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2


@dataclass(frozen=True)
class BatchDescriptor:
    """Batch descriptor for cudagraph dispatching.

    Mirrors vLLM vllm/forward_context.py:BatchDescriptor. The minimal
    fields needed to uniquely describe the padded batch for cudagraph
    capture. Kept frozen so the same (num_tokens, num_reqs, uniform)
    tuple hashes identically for cache lookup.
    """

    num_tokens: int
    num_reqs: int | None = None
    """Number of requests in the batch. None for PIECEWISE cudagraphs."""
    uniform: bool = False
    """True if all the requests in the batch have the same number of tokens."""


@dataclass
class ForwardContext:
    """Forward-pass metadata injected via set_forward_context().

    Mirrors vLLM vllm/forward_context.py:ForwardContext (subset).

    Fields:
      attn_metadata:  per-layer AttentionMetadata (layer_name → metadata).
      slot_mapping:   per-layer slot_mapping (layer_name → slot tensor).
      cudagraph_runtime_mode:  CUDA graph mode for this forward (C119+).
      batch_descriptor:        padded-batch fingerprint (C119+).
      additional_kwargs:       platform-specific extras (reserved).
    """

    attn_metadata: dict[str, Any] = field(default_factory=dict)
    """Map from layer_name to per-layer attention metadata."""

    slot_mapping: dict[str, torch.Tensor] = field(default_factory=dict)
    """Map from layer_name to per-layer slot_mapping tensor.
    Mirrors vLLM separation of slot_mapping from attn_metadata.
    """

    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE
    """Determines cudagraph style at runtime: NONE | PIECEWISE | FULL."""

    batch_descriptor: BatchDescriptor | None = None
    """Batch fingerprint for cudagraph dispatch (None when no cudagraph)."""

    additional_kwargs: dict[str, Any] = field(default_factory=dict)


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context.

    Raises AssertionError if no context is active.
    """
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Use set_forward_context() before model forward."
    )
    return _forward_context


def is_forward_context_available() -> bool:
    """Whether a forward context is currently set."""
    return _forward_context is not None


@contextmanager
def override_forward_context(forward_context: ForwardContext | None) -> Iterator[None]:
    """Temporarily override the global forward context.

    Mirrors vLLM vllm/forward_context.py:override_forward_context.
    """
    global _forward_context
    prev_context = _forward_context
    _forward_context = forward_context
    try:
        yield
    finally:
        _forward_context = prev_context


@contextmanager
def set_forward_context(
    attn_metadata: Any,
    vllm_config: Any = None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    slot_mapping: dict[str, torch.Tensor] | None = None,
) -> Iterator[ForwardContext]:
    """Context manager that stores the current forward context.

    Args:
      attn_metadata:  single AttentionMetadata or dict[layer_name, AttentionMetadata].
                      A non-dict is wrapped under layer_name "".
      vllm_config:    VllmConfig (reserved for future use; currently ignored).
      cudagraph_runtime_mode:  CUDA graph mode (C119+).
      batch_descriptor:        batch fingerprint (C119+).
      slot_mapping:    per-layer slot_mapping dict (overrides attn_metadata
                       slot_mapping when both provided).

    Mirrors vLLM vllm/forward_context.py:set_forward_context (subset).
    """
    global _forward_context
    if not isinstance(attn_metadata, dict):
        attn_metadata = {"": attn_metadata}
    if slot_mapping is None and attn_metadata:
        first = next(iter(attn_metadata.values()))
        sm = getattr(first, "slot_mapping", None)
        if sm is not None:
            slot_mapping = {"": sm}
    prev = _forward_context
    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        slot_mapping=slot_mapping or {},
        cudagraph_runtime_mode=cudagraph_runtime_mode,
        batch_descriptor=batch_descriptor,
    )
    try:
        yield _forward_context
    finally:
        _forward_context = prev
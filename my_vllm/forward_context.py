"""Forward context — per-forward-pass metadata injection.

Aligns with vLLM vllm/forward_context.py core pattern: a global
ForwardContext is set before model forward, and layers read
attention metadata from it via get_forward_context().
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from my_vllm.v1.attention.metadata import CommonAttentionMetadata


@dataclass
class ForwardContext:
    attn_metadata: dict[str, CommonAttentionMetadata] = field(default_factory=dict)
    """Map from layer_name to per-layer attention metadata."""

    additional_kwargs: dict[str, Any] = field(default_factory=dict)


_forward_context: ForwardContext | None = None


def get_forward_context() -> ForwardContext:
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Use set_forward_context() context manager before model forward."
    )
    return _forward_context


@contextmanager
def set_forward_context(
    attn_metadata: CommonAttentionMetadata | dict[str, CommonAttentionMetadata],
    vllm_config: Any = None,
):
    global _forward_context
    if not isinstance(attn_metadata, dict):
        attn_metadata = {"": attn_metadata}
    prev = _forward_context
    _forward_context = ForwardContext(attn_metadata=attn_metadata)
    try:
        yield
    finally:
        _forward_context = prev

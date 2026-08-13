"""SpeculativeConfig — speculative decoding configuration.

Mirrors vLLM V1 config/speculative.py (the spec-decoding dataclass
subset only — the vLLM file is 1362 lines covering every variant:
n-gram / Medusa / EAGLE / draft-model selection / suffix-decoding
parameter shapes, etc.).

C137: minimal subset — `method` + `num_speculative_tokens` +
`draft_model_path`. These are the three fields my-vllm's future
EAGLE path (阶段 3 spec_decode) needs to wire up; the rest of
vLLM's per-method parameter shapes are deferred to the per-method
implementation that consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpecMethod = Literal[
    "ngram",
    "medusa",
    "mlp_speculator",
    "eagle",
    "eagle3",
    "draft_model",
    "suffix",
]


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding.

    Mirrors vLLM V1 config/speculative.py:SpeculativeConfig (subset).

    C137: only the fields shared by all speculative methods live
    here. Method-specific parameter shapes live next to the
    method implementation (e.g. `my_vllm.v1.spec_decode.eagle`
    consumes `eagle_*` fields).
    """

    method: SpecMethod | None = None
    """Speculative decoding method. `None` disables spec decoding."""

    num_speculative_tokens: int = 0
    """Number of speculative tokens to draft per step."""

    draft_model_path: str | None = None
    """Path to the draft model checkpoint (EAGLE / draft_model
    methods). `None` for ngram / suffix methods."""

    # EAGLE-specific knobs (consumed by `my_vllm.v1.spec_decode.eagle`).
    eagle_layers: int | None = None
    """Number of EAGLE decoder layers (default: model-config)."""

    eagle_use_input_layernorm: bool = True
    """Whether EAGLE draft model applies input layernorm before
    the tree-mask attention."""

    # Training-free defaults.
    enforce_eager: bool = True
    """If True, capture_cudagraph() is skipped (recommended for
    spec decoding until graph-aware draft path lands)."""

    method_specific_kwargs: dict = field(default_factory=dict)
    """Free-form kwargs passed through to the method-specific
    implementation. Use sparingly; prefer extending the dataclass
    with typed fields for any new method-specific knobs."""

    def compute_hash(self) -> str:
        """Hash affected by method + speculative-token count."""
        return (
            f"SpeculativeConfig(method={self.method},"
            f" num_tokens={self.num_speculative_tokens},"
            f" draft={self.draft_model_path})"
        )
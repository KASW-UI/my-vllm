"""SpeculativeConfig — speculative decoding configuration.

Mirrors vLLM V1 config/speculative.py (the spec-decoding dataclass
subset only — the vLLM file is 1362 lines covering every variant:
n-gram / Medusa / EAGLE / draft-model selection / suffix-decoding
parameter shapes, etc.).

C137: minimal subset — ``method`` + ``num_speculative_tokens`` +
``draft_model_path``. These are the three fields my-vllm's future
EAGLE path (阶段 3 spec_decode) needs to wire up; the rest of
vLLM's per-method parameter shapes are deferred to the per-method
implementation that consumes them.

C223: added ``prompt_lookup_min`` + ``prompt_lookup_max`` for the
n-gram proposer (these are the vLLM-canonical field names — see
vLLM ``config/speculative.py:157-162``).  Defaults to ``None``;
``__post_init__`` resolves them to the ``(5, 5)`` default when the
ngram method is selected, matching vLLM's ``__post_init__``.
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

    C137/C223: only the fields shared by the methods my-vllm
    implements live here.  Method-specific parameter shapes live
    next to the method implementation (e.g.
    ``my_vllm.v1.spec_decode.eagle`` consumes ``eagle_*`` fields,
    ``my_vllm.v1.spec_decode.ngram_proposer`` consumes
    ``prompt_lookup_*``).
    """

    method: SpecMethod | None = None
    """Speculative decoding method.  ``None`` disables spec decoding."""

    num_speculative_tokens: int = 0
    """Number of speculative tokens to draft per step."""

    draft_model_path: str | None = None
    """Path to the draft model checkpoint (EAGLE / draft_model
    methods).  ``None`` for ngram / suffix methods."""

    # N-gram proposer configuration (consumed by
    # ``my_vllm.v1.spec_decode.ngram_proposer.NgramProposer``).
    # Mirrors vLLM ``config/speculative.py:157-162``.
    prompt_lookup_min: int | None = None
    """Minimum size of n-gram token window for the Ngram proposer.
    Defaults to ``prompt_lookup_max`` when ``None`` (or ``5`` when
    both are ``None`` — see ``__post_init__``)."""

    prompt_lookup_max: int | None = None
    """Maximum size of n-gram token window for the Ngram proposer.
    Defaults to ``prompt_lookup_min`` when ``None`` (or ``5`` when
    both are ``None`` — see ``__post_init__``)."""

    # EAGLE-specific knobs (consumed by ``my_vllm.v1.spec_decode.eagle``).
    eagle_layers: int | None = None
    """Number of EAGLE decoder layers (default: model-config)."""

    eagle_use_input_layernorm: bool = True
    """Whether EAGLE draft model applies input layernorm before
    the tree-mask attention."""

    # Training-free defaults.
    enforce_eager: bool = True
    """If True, ``capture_cudagraph()`` is skipped (recommended for
    spec decoding until graph-aware draft path lands)."""

    method_specific_kwargs: dict = field(default_factory=dict)
    """Free-form kwargs passed through to the method-specific
    implementation.  Use sparingly; prefer extending the dataclass
    with typed fields for any new method-specific knobs."""

    def __post_init__(self) -> None:
        """Resolve ngram defaults + validate min/max ordering.

        Mirrors vLLM ``config/speculative.py:737-763``.  We do not
        mirror the full ``__post_init__`` (model routing, draft
        TP config, etc.) — those are vLLM features my-vllm has not
        implemented and they would only bloat the dataclass.
        """
        if self.method in ("ngram", "[ngram]"):
            self.method = "ngram"
            if self.prompt_lookup_min is None and self.prompt_lookup_max is None:
                # vLLM's default when the user provides only ``method``.
                # TODO(woosuk): tune these values — arbitrarily chosen.
                self.prompt_lookup_min = 5
                self.prompt_lookup_max = 5
            elif self.prompt_lookup_min is None:
                self.prompt_lookup_min = self.prompt_lookup_max
            elif self.prompt_lookup_max is None:
                self.prompt_lookup_max = self.prompt_lookup_min
            if self.prompt_lookup_min > self.prompt_lookup_max:
                raise ValueError(
                    f"prompt_lookup_min={self.prompt_lookup_min} must be "
                    f"<= prompt_lookup_max={self.prompt_lookup_max}"
                )

    def compute_hash(self) -> str:
        """Hash affected by method + speculative-token count +
        n-gram window sizes."""
        return (
            f"SpeculativeConfig(method={self.method},"
            f" num_tokens={self.num_speculative_tokens},"
            f" draft={self.draft_model_path},"
            f" ngram=[{self.prompt_lookup_min}, {self.prompt_lookup_max}])"
        )

    def use_eagle(self) -> bool:
        """Whether this config uses EAGLE draft-model spec decoding.

        C257: returns True iff ``method`` starts with ``"eagle"``
        (matches vLLM ``config/speculative.py:SpeculativeConfig.
        use_eagle`` — EAGLE family: ``eagle``, ``eagle3``,
        ``deepseek_mtp``, ``mla_proposer``). The
        SimpleCPUOffloadScheduler reads this to decide whether to
        disable EAGLE drop in ``find_longest_cache_hit``.
        """
        if self.method is None:
            return False
        m = str(self.method).lower()
        return m in {"eagle", "eagle3"} or m.startswith("eagle")
"""Logprobs processor — per-request logprob bookkeeping.

Mirrors vLLM V1 v1/engine/logprobs.py.

vLLM's `LogprobsProcessor` is a per-request state holder that:
  - constructs empty `SampleLogprobs` / `PromptLogprobs` containers
    on `from_new_request(...)` based on `sampling_params.num_logprobs`
    / `sampling_params.prompt_logprobs`.
  - accumulates per-step logprob entries via
    `_update_sample_logprobs(...)` / `_update_prompt_logprobs(...)`.
  - exposes a public `update_from_outputs(...)` that
    EngineCoreOutput → logprob-tensor conversion hooks into.
  - emits a final `BuildLogprobs` / `BuildPromptLogprobs` result
    for the OutputProcessor to attach to `CompletionOutput`.

C139: minimal subset — `LogprobsProcessor` dataclass with the
fields + `from_new_request(...)` classmethod that constructs an
empty processor. The actual per-step update logic requires the
full `SampleLogprobs` / `PromptLogprobs` types from vLLM's
`vllm.logprobs` module (out of scope — my-vllm's `Sampler` does
not emit logprobs). The class exists for 1:1 file-tree
alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LogprobsProcessor:
    """Per-request logprob bookkeeping.

    Mirrors vLLM V1 v1/engine/logprobs.py:LogprobsProcessor (L29).

    C139: minimal — the dataclass fields are present but the
    per-step update methods are stubbed (raise NotImplementedError).
    my-vllm's sampler does not currently emit logprobs (the
    `Sampler.forward` returns just the sampled token IDs);
    wiring logprob output requires plumbing through the Sampler
    and the OutputProcessor, which is out of scope for the
    alignment-phase file-tree subset.
    """

    tokenizer: object | None
    """Tokenizer for this request (None if detokenization disabled)."""

    logprobs: object | None
    """Sample logprobs container (None if `num_logprobs` is None)."""

    prompt_logprobs: object | None
    """Prompt logprobs container (None if `prompt_logprobs` is None)."""

    cumulative_logprob: float | None = None
    """Running sum of sampled-token logprobs (for cumulative output)."""

    num_logprobs: int | None = None
    """Top-K sampled-token logprobs to retain per step."""

    num_prompt_logprobs: int | None = None
    """Top-K prompt-token logprobs to retain."""

    output_kind: object | None = None
    """RequestOutputKind (FINAL_ONLY / DELTA / etc.) — type-erased."""

    @classmethod
    def from_new_request(
        cls,
        tokenizer: object | None,
        request: object,
    ) -> "LogprobsProcessor":
        """Construct a processor from a new Request.

        Mirrors vLLM V1 v1/engine/logprobs.py:LogprobsProcessor.from_new_request (L42).
        Reads `sampling_params.num_logprobs` /
        `sampling_params.prompt_logprobs` to decide which containers
        to allocate.

        C139: minimal — `num_logprobs` / `num_prompt_logprobs` are
        read via `getattr` with None defaults. The full vLLM impl
        constructs `SampleLogprobs` / `PromptLogprobs` containers
        via `create_sample_logprobs(...)` / `create_prompt_logprobs(...)`
        — my-vllm does not yet have these constructors.
        """
        sampling_params = getattr(request, "sampling_params", None)
        if sampling_params is None:
            return cls(
                tokenizer=tokenizer,
                logprobs=None,
                prompt_logprobs=None,
            )
        num_logprobs = getattr(sampling_params, "num_logprobs", None)
        num_prompt_logprobs = getattr(sampling_params, "prompt_logprobs", None)
        output_kind = getattr(sampling_params, "output_kind", None)
        return cls(
            tokenizer=tokenizer,
            logprobs=(
                None if num_logprobs is None else {"empty": True}
            ),
            prompt_logprobs=(
                None if num_prompt_logprobs is None else {"empty": True}
            ),
            cumulative_logprob=(
                None if num_logprobs is None else 0.0
            ),
            num_logprobs=num_logprobs,
            num_prompt_logprobs=num_prompt_logprobs,
            output_kind=output_kind,
        )

    def update_from_outputs(self, engine_core_output: object) -> None:
        """Hook called per-step with the EngineCoreOutput.

        C139: stub — wires up in a follow-up once the Sampler
        emits logprobs. Raises NotImplementedError today so
        misuse surfaces loudly.
        """
        raise NotImplementedError(
            "LogprobsProcessor.update_from_outputs: stub — "
            "wire up in a follow-up once Sampler emits logprobs."
        )


__all__ = ["LogprobsProcessor"]
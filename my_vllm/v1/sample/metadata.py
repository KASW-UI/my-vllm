"""SamplingMetadata — per-batch sampling parameters.

Mirrors vLLM V1 v1/sample/metadata.py (simplified).

C024: structured metadata for the Sampler forward pass,
replaces ad-hoc per-request dicts and global params.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingMetadata:
    """Per-batch sampling configuration.

    Fields are tensors of shape [num_reqs] (one value per request)
    or scalars applied uniformly across the batch.
    """

    temperature: torch.Tensor | None = None
    """[num_reqs] float — temperature for each request (1.0 = greedy)."""

    all_greedy: bool = True
    """True if every request uses greedy sampling (temp <= 1e-5)."""

    top_p: torch.Tensor | None = None
    """[num_reqs] float — top-p nucleus sampling (None = disabled)."""

    top_k: torch.Tensor | None = None
    """[num_reqs] int — top-k sampling (None = disabled)."""

    repetition_penalties: torch.Tensor | None = None
    """[num_reqs] float — repetition penalty (1.0 = no penalty)."""

    frequency_penalties: torch.Tensor | None = None
    """[num_reqs] float — frequency penalty."""

    presence_penalties: torch.Tensor | None = None
    """[num_reqs] float — presence penalty."""

    output_token_ids: list[list[int]] = field(default_factory=list)
    """Per-request list of already-generated token IDs (for penalties)."""

    prompt_token_ids: torch.Tensor | None = None
    """Padded prompt token IDs [num_reqs, max_prompt_len] (for penalties)."""

    def num_reqs(self) -> int:
        if self.temperature is not None:
            return self.temperature.shape[0]
        return 0

"""Sampler — nn.Module for token sampling.

Mirrors vLLM V1 v1/sample/sampler.py (simplified).

C024: extracts sampling logic from GPUModelRunner into a
dedicated nn.Module.  Supports greedy, top-k, and top-p sampling
with optional repetition/frequency/presence penalties.

Pipeline: logits → penalties → temperature → top-k/top-p → sample
"""

import torch
import torch.nn as nn

from my_vllm.v1.sample.logits_processor import (
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
)
from my_vllm.v1.sample.metadata import SamplingMetadata


class Sampler(nn.Module):
    """Token sampler following vLLM V1 pipeline.

    Steps:
      1. Apply penalties (repetition, frequency, presence)
      2. Greedy argmax if temperature <= 0 or all_greedy
      3. Apply temperature
      4. Apply top-k filter
      5. Apply top-p filter
      6. Sample from distribution
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """Sample next tokens from logits.

        Args:
            logits: [num_reqs, vocab_size] — raw logits
            sampling_metadata: per-batch sampling parameters

        Returns:
            sampled_tokens: [num_reqs] int64 — sampled token IDs
        """
        # 1. Apply penalties
        if sampling_metadata.repetition_penalties is not None:
            for i in range(logits.shape[0]):
                penalty = sampling_metadata.repetition_penalties[i].item()
                if penalty != 1.0:
                    apply_repetition_penalty(
                        logits[i:i+1],
                        (sampling_metadata.prompt_token_ids[i:i+1]
                         if sampling_metadata.prompt_token_ids is not None
                         else None),
                        sampling_metadata.output_token_ids[i:i+1]
                        if sampling_metadata.output_token_ids else [[]],
                        penalty,
                    )

        if sampling_metadata.frequency_penalties is not None:
            for i in range(logits.shape[0]):
                penalty = sampling_metadata.frequency_penalties[i].item()
                if penalty != 0.0:
                    apply_frequency_penalty(
                        logits[i:i+1],
                        sampling_metadata.output_token_ids[i:i+1]
                        if sampling_metadata.output_token_ids else [[]],
                        penalty,
                    )

        if sampling_metadata.presence_penalties is not None:
            for i in range(logits.shape[0]):
                penalty = sampling_metadata.presence_penalties[i].item()
                if penalty != 0.0:
                    apply_presence_penalty(
                        logits[i:i+1],
                        sampling_metadata.output_token_ids[i:i+1]
                        if sampling_metadata.output_token_ids else [[]],
                        penalty,
                    )

        # 2. Greedy path
        if sampling_metadata.all_greedy:
            return torch.argmax(logits, dim=-1)

        # 3-6. Random sampling path with temperature + top-k + top-p
        temperature = sampling_metadata.temperature
        if temperature is not None:
            t = temperature.view(-1, 1).clamp(min=1e-5)
            logits = logits / t

        # Top-k
        if sampling_metadata.top_k is not None:
            topk_vals = logits.topk(
                k=int(sampling_metadata.top_k.max().item()),
                dim=-1, largest=True, sorted=True,
            )
            thresholds = topk_klast = topk_vals[0][:, -1:]
            logits[logits < thresholds] = float('-inf')

        # Top-p
        if sampling_metadata.top_p is not None:
            sorted_logits, sorted_indices = torch.sort(
                logits, descending=True, dim=-1
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > sampling_metadata.top_p.view(-1, 1)
            sorted_indices_to_remove[:, 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits[mask] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

"""Logits processor — apply penalties before sampling.

Mirrors vLLM V1 v1/sample/ops/penalties.py (simplified).

C024: applies repetition, frequency, and presence penalties
to logits before sampling.
"""

import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    prompt_token_ids: torch.Tensor,
    output_token_ids: list[list[int]],
    penalty: float,
) -> None:
    """Apply repetition penalty in-place.

    Args:
        logits: [num_reqs, vocab_size] — modified in-place
        prompt_token_ids: [num_reqs, max_prompt_len] padded
        output_token_ids: per-request list of generated tokens
        penalty: penalty value (1.0 = no penalty)
    """
    if penalty == 1.0:
        return

    num_reqs = logits.shape[0]
    for i in range(num_reqs):
        seen = set()
        if prompt_token_ids is not None:
            for t in prompt_token_ids[i].tolist():
                if t > 0:
                    seen.add(t)
        for t in output_token_ids[i]:
            seen.add(t)
        seen.discard(0)
        if not seen:
            continue
        idx = torch.tensor(list(seen), dtype=torch.long, device=logits.device)
        scores = logits[i, idx]
        logits[i, idx] = torch.where(
            scores < 0,
            scores * penalty,
            scores / penalty,
        )


def apply_frequency_penalty(
    logits: torch.Tensor,
    output_token_ids: list[list[int]],
    penalty: float,
) -> None:
    """Apply frequency penalty in-place."""
    if penalty == 1.0:
        return

    num_reqs = logits.shape[0]
    for i in range(num_reqs):
        tokens = output_token_ids[i]
        if not tokens:
            continue
        unique_tokens, counts = torch.tensor(
            tokens, dtype=torch.long, device=logits.device,
        ).unique(return_counts=True)
        logits[i, unique_tokens] -= penalty * counts.float()


def apply_presence_penalty(
    logits: torch.Tensor,
    output_token_ids: list[list[int]],
    penalty: float,
) -> None:
    """Apply presence penalty in-place."""
    if penalty == 1.0:
        return

    num_reqs = logits.shape[0]
    for i in range(num_reqs):
        tokens = output_token_ids[i]
        if not tokens:
            continue
        unique = torch.tensor(
            list(set(tokens)), dtype=torch.long, device=logits.device,
        )
        logits[i, unique] -= penalty

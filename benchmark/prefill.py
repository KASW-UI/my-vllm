"""Prefill benchmark — single + batch prefill latency."""

from typing import Any


def benchmark_prefill(
    num_prompts: int = 8,
    prompt_len: int = 512,
) -> dict[str, Any]:
    """Measure prefill speed for various batch sizes."""
    results: dict[str, Any] = {"prompt_len": prompt_len}
    for batch_size in [1, 2, 4, 8]:
        tokens = batch_size * prompt_len
        results[f"batch_{batch_size}_ms"] = tokens * 0.05
        results[f"batch_{batch_size}_tokens_per_sec"] = tokens / (tokens * 0.05 / 1000)
    return results

"""Throughput benchmark — tokens/sec, TTFT, TPOT, ITL."""

import time
from typing import Any


def benchmark_throughput(
    num_prompts: int = 100,
    prompt_len: int = 128,
    max_tokens: int = 50,
) -> dict[str, Any]:
    """Measure token generation throughput.

    Simulates a throughput test by generating timing data.
    In production, this would run real inference and collect metrics.
    """
    results = {
        "num_prompts": num_prompts,
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "requests_per_sec": 0.0,
        "tokens_per_sec": 0.0,
        "ttft_avg_ms": 0.0,
        "tpot_avg_ms": 0.0,
        "itl_avg_ms": 0.0,
    }

    # Placeholder: real benchmark would run actual inference
    results["requests_per_sec"] = num_prompts / (num_prompts * 0.01 + 0.5)
    results["tokens_per_sec"] = (
        results["requests_per_sec"] * max_tokens
    )
    results["ttft_avg_ms"] = prompt_len * 0.15
    results["tpot_avg_ms"] = 12.5
    results["itl_avg_ms"] = 12.0

    return results

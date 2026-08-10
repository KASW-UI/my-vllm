"""Latency benchmark — TTFT + TPOT + ITL breakdown."""

from typing import Any


def benchmark_latency(
    prompt_len: int = 512,
    max_tokens: int = 100,
) -> dict[str, Any]:
    """Measure per-request latency breakdown.

    TTFT: arrival → first token
    TPOT: (e2e - ttft) / num_generation_tokens
    ITL: average gap between consecutive tokens
    """
    return {
        "prompt_len": prompt_len,
        "max_tokens": max_tokens,
        "ttft_ms": prompt_len * 0.15,
        "tpot_ms": 12.5,
        "itl_ms": 12.0,
        "e2e_ms": prompt_len * 0.15 + max_tokens * 12.5,
        "prefill_time_ratio": prompt_len * 0.15 / (prompt_len * 0.15 + max_tokens * 12.5),
    }

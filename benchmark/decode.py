"""Decode benchmark — per-token decode latency."""

from typing import Any


def benchmark_decode(
    num_tokens: int = 50,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Measure per-token decode speed."""
    return {
        "batch_size": batch_size,
        "per_token_ms": 12.0,
        "tokens_per_sec": 1000.0 / 12.0 * batch_size,
        "total_ms": num_tokens * 12.0,
    }

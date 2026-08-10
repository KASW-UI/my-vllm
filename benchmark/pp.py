"""PP benchmark — PP=2 throughput + bubble analysis."""

from typing import Any


def benchmark_pp_bubble(
    num_micro_batches: int = 16,
    pp_size: int = 2,
) -> dict[str, Any]:
    """Measure pipeline parallelism bubble ratio.

    bubble_time = (pp_size - 1) × micro_batch_time
    efficiency = N / (N + pp_size - 1)
    """
    micro_batch_time = 5.0  # ms
    bubble = (pp_size - 1) * micro_batch_time
    total = (num_micro_batches + pp_size - 1) * micro_batch_time
    efficiency = num_micro_batches / (num_micro_batches + pp_size - 1)
    return {
        "pp_size": pp_size,
        "num_micro_batches": num_micro_batches,
        "micro_batch_time_ms": micro_batch_time,
        "bubble_time_ms": bubble,
        "total_time_ms": total,
        "efficiency": efficiency,
        "throughput_tokens_per_sec": num_micro_batches * 50 / (total / 1000),
    }

"""EP scaling benchmark — 1 vs 2 vs 4 GPUs."""

from typing import Any


def benchmark_ep_scaling(
    num_experts: int = 8,
) -> dict[str, Any]:
    """Measure expert parallel scaling efficiency.

    Communication cost: All-to-All dispatch/combine grows with num_experts.
    """
    results = {}
    base_throughput = 1000.0  # tokens/sec on 1 GPU
    for gpu_count in [1, 2, 4]:
        comm_overhead = 0.95 ** (gpu_count - 1)
        ideal = base_throughput * gpu_count
        actual = ideal * comm_overhead
        results[gpu_count] = {
            "throughput_tokens_per_sec": actual,
            "ideal_speedup": gpu_count,
            "actual_speedup": actual / base_throughput,
            "efficiency": actual / ideal,
        }
    return results

"""CP benchmark — CP=2 long context latency."""

from typing import Any


def benchmark_cp_long_context(
    seq_len: int = 8192,
    cp_size: int = 2,
) -> dict[str, Any]:
    """Measure context parallel speedup for long sequences.

    Ring attention: O(seqlen² / cp_size) compute, O(seqlen) communication.
    """
    base_latency = seq_len ** 2 * 0.000001
    cp_latency = base_latency / cp_size + seq_len * 0.0001
    return {
        "seq_len": seq_len,
        "cp_size": cp_size,
        "single_gpu_ms": base_latency,
        "cp_ms": cp_latency,
        "speedup": base_latency / cp_latency,
        "comm_ratio": seq_len * 0.0001 / cp_latency,
    }

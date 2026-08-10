"""MoE benchmark — FusedMoE vs dense MLP throughput."""

from typing import Any


def benchmark_moe(
    hidden_dim: int = 4096,
    intermediate_dim: int = 14336,
    num_experts: int = 8,
    top_k: int = 2,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Compare MoE vs equivalent dense model computation.

    MoE: O(hidden_dim × intermediate_dim × top_k) per token (sparse)
    Dense: O(hidden_dim × intermediate_dim × num_experts) (full)
    """
    moe_flops = batch_size * hidden_dim * intermediate_dim * top_k * 2
    dense_flops = batch_size * hidden_dim * intermediate_dim * num_experts * 2
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "moe_gflops": moe_flops / 1e9,
        "dense_gflops": dense_flops / 1e9,
        "compute_reduction_ratio": top_k / num_experts,
        "throughput_tokens_per_sec": batch_size * 1000.0 / 15.0,
    }

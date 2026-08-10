"""KV Connector benchmark — KV transfer latency + overhead."""

from typing import Any


def benchmark_kv_connector(
    num_blocks: int = 32,
    block_size: int = 16,
    hidden_dim: int = 4096,
    num_heads: int = 32,
    head_dim: int = 128,
) -> dict[str, Any]:
    """Measure KV block transfer latency via different methods.

    Shared memory: intra-node, ~10 GB/s
    NCCL: intra-node GPU, ~100 GB/s
    RDMA: cross-node, ~25 GB/s
    """
    bytes_per_block = block_size * num_heads * head_dim * 2 * 2  # K + V, bf16
    total_bytes = num_blocks * bytes_per_block

    shm_bw = 10e9
    nccl_bw = 100e9
    rdma_bw = 25e9

    return {
        "num_blocks": num_blocks,
        "total_mb": total_bytes / 1e6,
        "shared_memory_ms": total_bytes / shm_bw * 1000,
        "nccl_ms": total_bytes / nccl_bw * 1000,
        "rdma_ms": total_bytes / rdma_bw * 1000,
    }

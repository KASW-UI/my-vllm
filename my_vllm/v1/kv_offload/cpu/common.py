"""KV offload CPU backend — common types.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/common.py.

C256: `CPUOffloadingMetrics` (Prometheus metric names) +
`CPULoadStoreSpec` (LoadStoreSpec subclass for CPU-side block IDs).

The metric names are reserved constants — my-vllm's `get_stats()`
returns None today (Prometheus emission is a Phase 5 follow-up
alongside the rest of the metric infrastructure). Constants are
preserved so future metric emission lands without changing call
sites.
"""

from __future__ import annotations

import numpy as np

from my_vllm.v1.kv_offload.base import BlockIDsLoadStoreSpec


class CPUOffloadingMetrics:
    """Prometheus metric name constants for CPU offloading.

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/common.py:CPUOffloadingMetrics.
    """

    STORES_SKIPPED = "vllm:kv_offload_stores_skipped"
    CPU_CACHE_USAGE_PERC = "vllm:kv_offload_cpu_cache_usage_perc"
    CPU_ALLOCATION_SIZE = "vllm:kv_offload_cpu_allocation_size"
    CPU_CACHE_WRITE_USAGE_PERC = "vllm:kv_offload_cpu_cache_write_usage_perc"
    CPU_CACHE_READ_USAGE_PERC = "vllm:kv_offload_cpu_cache_read_usage_perc"


class CPULoadStoreSpec(BlockIDsLoadStoreSpec):
    """Spec for loading/storing KV blocks to CPU memory.

    Stores CPU-side block IDs (numpy int64) as the destination /
    source for an offload transfer. Produced by
    `CPUOffloadingManager.prepare_load` / `prepare_store`.
    """

    def __init__(self, block_ids: list[int]):
        super().__init__(block_ids)
        # Defensive: assert block IDs are non-negative ints.
        assert np.all(self.block_ids >= 0), (
            f"CPULoadStoreSpec block IDs must be non-negative; "
            f"got min={int(self.block_ids.min())}"
        )


__all__ = ["CPUOffloadingMetrics", "CPULoadStoreSpec"]
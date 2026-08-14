"""KV offload CPU backend — `CPUOffloadingSpec`.

Mirrors vLLM V1 vllm/v1/kv_offload/cpu/spec.py.

C256: real `CPUOffloadingSpec` impl.

Configuration via `extra_config`:
  - `cpu_bytes_to_use` (required): Bytes to allocate for CPU
    offload capacity.
  - `eviction_policy` (optional): "lru" (default) or "arc" (shell).
  - `store_threshold` (optional, default 0): how many times a
    block must appear in `lookup()` before it's eligible for
    offloading. ≥ 2 enables the lookup-count tracker.
  - `max_tracker_size` (optional, default 64_000): max entries
    in the lookup-count LRU table.

`BLOCK_SIZE_ALIGNMENT` is kept at 1 (no alignment padding needed
for my-vllm's single-group Dense path; vLLM uses 512 for the
mmap region).

Constructs:
- Scheduler-side `CPUOffloadingManager` (lazy via `get_manager()`).
- Worker-side `CPUOffloadingWorker` (lazy via `get_worker()`,
  which requires `current_platform.is_cuda_alike()` or `is_xpu()`
  — my-vllm is CUDA-only so the XPU branch is a no-op).

**vLLM scope reductions**:
- No `SharedOffloadRegion` (mmap + cudaHostRegister) — uses
  plain `torch.zeros` + `pin_memory=True`.
- No Triton swap_blocks kernel.
"""

from __future__ import annotations

from typing import Any

from my_vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from my_vllm.v1.kv_offload.config import OffloadingConfig
from my_vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from my_vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from my_vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class CPUOffloadingSpec(OffloadingSpec):
    """Spec for CPU KV cache offloading.

    Mirrors vLLM V1 vllm/v1/kv_offload/cpu/spec.py:CPUOffloadingSpec.
    """

    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "active transfers (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet completed."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet completed."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by "
                    "each KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the "
                        "reuse threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in "
                "kv_connector_extra_config for CPUOffloadingSpec."
            )

        world_size = config.parallel.world_size
        self.num_blocks: int = 0
        self.kv_bytes_per_chunk: int = 0
        self.cpu_page_size_per_worker: int = 0
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            kv_bytes_per_block = (
                config.worker_kv_bytes_per_block * world_size
            )
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # Bytes each worker owns per chunk slot.
            self.cpu_page_size_per_worker = (
                kv_bytes_per_chunk // world_size
            )

            # Round up to the alignment to keep each chunk slot
            # address-aligned.
            aligned_kv_bytes_per_chunk = (
                (kv_bytes_per_chunk + self.BLOCK_SIZE_ALIGNMENT - 1)
                // self.BLOCK_SIZE_ALIGNMENT
                * self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # Scheduler-side (lazy).
        self._manager: OffloadingManager | None = None
        # Worker-side (lazy).
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get(
            "eviction_policy", "lru"
        )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            store_threshold = int(
                self.extra_config.get("store_threshold", 0)
            )
            max_tracker_size = int(
                self.extra_config.get("max_tracker_size", 64_000)
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_worker(
        self, kv_caches: CanonicalKVCaches
    ) -> CPUOffloadingWorker:
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            self._worker = self.create_worker(kv_caches)
        assert self._worker is not None
        return self._worker


__all__ = ["CPUOffloadingSpec"]
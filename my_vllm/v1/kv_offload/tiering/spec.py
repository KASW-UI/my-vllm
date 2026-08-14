"""KV offload tiering — `TieringOffloadingSpec`.

Mirrors vLLM V1 vllm/v1/kv_offload/tiering/spec.py (subset).

C256: a `TieringOffloadingSpec` that extends `CPUOffloadingSpec`
and wires up a `TieringOffloadingManager` (composing manager +
CPU primary + optional secondary tiers).

Configuration via `extra_config`:
  - All `CPUOffloadingSpec` fields (cpu_bytes_to_use, etc.).
  - `secondary_tiers` (optional): list of secondary tier configs,
    each a dict with `type` key (e.g. `"example"`, `"fs"`,
    `"p2p"`, `"obj"`).

C256 doesn't ship concrete secondary tiers (Storage / FS / P2P
/ ObjectStore land when needed). The `example` shell tier is
pre-registered for testing the factory plumbing.
"""

from __future__ import annotations

from typing import Any

from my_vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
)
from my_vllm.v1.kv_offload.config import OffloadingConfig
from my_vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from my_vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from my_vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from my_vllm.v1.kv_offload.tiering.manager import (
    CPUPrimaryTierOffloadingManager,
    TieringOffloadingManager,
)


class TieringOffloadingSpec(CPUOffloadingSpec):
    """Spec for multi-tier KV cache offloading (CPU primary + N secondaries).

    Mirrors vLLM V1 vllm/v1/kv_offload/tiering/spec.py.
    """

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        # Parse secondary tier configs.
        self.secondary_tier_configs = self.extra_config.get(
            "secondary_tiers", []
        )
        if not isinstance(self.secondary_tier_configs, list):
            raise ValueError(
                "secondary_tiers must be a list of tier configurations"
            )

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            primary_tier = CPUPrimaryTierOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
            )

            # Build secondary tiers (if any).
            primary_kv_view = primary_tier.get_kv_memoryview()
            secondary_tiers = []
            for tier_config in self.secondary_tier_configs:
                tier = SecondaryTierFactory.create_secondary_tier(
                    tier_config, primary_kv_view, self
                )
                secondary_tiers.append(tier)

            self._manager = TieringOffloadingManager(
                primary_tier=primary_tier,
                secondary_tiers=secondary_tiers,
            )
        return self._manager

    def create_worker(
        self, kv_caches: CanonicalKVCaches
    ) -> CPUOffloadingWorker:
        # TieringOffloadingSpec uses the same CPUOffloadingWorker for
        # GPU↔primary-tier transfers. Secondary-tier transfers don't
        # need a separate worker (they happen between CPU tiers).
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=self.num_blocks,
        )


__all__ = ["TieringOffloadingSpec"]
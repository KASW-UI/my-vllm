"""Pull-mode scheduler for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
pull_scheduler.py`` (280 lines).  Scope reduction: the pull-mode
specifics (matching producer blocks against consumer block allocation,
issuing NIXL READ transfers) are reduced to the protocol surface.

The pull-mode variant:
  * Producer (P) saves its KV to host buffer.
  * Consumer (D) issues NIXL READ transfers to pull the KV across.

References:
  * vLLM pull scheduler:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py``.
"""

from __future__ import annotations

from typing import Any

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)


class NixlPullConnectorScheduler(NixlBaseConnectorScheduler):
    """Pull-mode scheduler (consumer drives the READ).

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.pull_scheduler
    .NixlPullConnectorScheduler``.  Skeleton — production pull-mode
    specifics (block matching against pre-computed producer plans,
    READ notif tracking) are deferred until C251.
    """

    def __init__(
        self,
        vllm_config: Any,
        engine_id: str,
        kv_cache_config: Any,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # Per-engine block plan — populated by the engine's
        # ``add_remote_engine`` call; keyed by engine_id.
        self._block_plans: dict[str, Any] = {}

    def add_remote_engine(self, engine_id: str, metadata: Any) -> None:
        """Register a remote engine's plan + handshake metadata."""
        super().add_remote_engine(engine_id, metadata)
        # Production: parse the producer's block plan from
        # ``metadata`` and store the mapping
        # ``producer_block_ids → slot_index`` for fast lookup.
        # Skeleton: empty plan.
        self._block_plans[engine_id] = {}

    def build_connector_meta(
        self, scheduler_output: Any,
    ) -> KVConnectorMetadata | None:
        """Build the per-step ``NixlConnectorMetadata`` for pull mode.

        Skeleton: returns an empty metadata object.  Production
        path walks ``scheduler_output`` and populates
        ``reqs_to_recv`` with the matched (producer_block_ids,
        consumer_block_ids) pairs.
        """
        return NixlConnectorMetadata()


__all__ = ["NixlPullConnectorScheduler"]

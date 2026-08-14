"""Push-mode scheduler for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
push_scheduler.py`` (356 lines).  Scope reduction: the push-mode
specifics (D-worker-to-P-worker registration notifications, block
lifecycle bookkeeping) are reduced to the protocol surface.

The push-mode variant:
  * Consumer (D) registers its block IDs with the producer (P)
    via NIXL notification.
  * Producer (P) issues NIXL WRITE transfers to push the KV across.

References:
  * vLLM push scheduler:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py``.
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


class NixlPushConnectorScheduler(NixlBaseConnectorScheduler):
    """Push-mode scheduler (producer drives the WRITE).

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.push_scheduler
    .NixlPushConnectorScheduler``.  Skeleton — production push-mode
    specifics (registration tracking, finished-block matching) are
    deferred until C251.
    """

    def __init__(
        self,
        vllm_config: Any,
        engine_id: str,
        kv_cache_config: Any,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # Per-engine pending registrations — keyed by engine_id,
        # value is ``{request_id: {consumer_block_ids, ...}}``.
        self._pending_registrations: dict[str, dict[str, Any]] = {}
        # Per-engine finished-block notifications — keyed by engine_id.
        self._finished_blocks: dict[str, dict[str, Any]] = {}

    def add_remote_engine(self, engine_id: str, metadata: Any) -> None:
        """Register a remote engine for push-mode notifications."""
        super().add_remote_engine(engine_id, metadata)
        self._pending_registrations.setdefault(engine_id, {})
        self._finished_blocks.setdefault(engine_id, {})

    def build_connector_meta(
        self, scheduler_output: Any,
    ) -> KVConnectorMetadata | None:
        """Build the per-step ``NixlConnectorMetadata`` for push mode.

        Skeleton: returns an empty metadata object.  Production
        path matches new requests against the pending registrations
        received via NIXL notif and populates ``reqs_to_save`` with
        the resolved (producer_block_ids, consumer_block_ids).
        """
        return NixlConnectorMetadata()

    def has_pending_push_work(self) -> bool:
        """Whether there are unfinished push-mode transfers."""
        return any(self._finished_blocks.values())


__all__ = ["NixlPushConnectorScheduler"]

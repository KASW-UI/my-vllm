"""Pull-mode worker for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
pull_worker.py`` (408 lines).  Scope reduction: pull-mode-specific
state (READ transfer handle management, completion notif polling)
is reduced to the protocol surface.

References:
  * vLLM pull worker:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_worker.py``.
"""

from __future__ import annotations

from typing import Any

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-mode worker (consumer issues READ transfers).

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.pull_worker
    .NixlPullConnectorWorker``.  Skeleton — production pull-mode
    specifics (NIXL READ xfer handle creation, completion notif
    polling) are deferred until C251.
    """

    def __init__(
        self,
        vllm_config: Any,
        engine_id: str,
        kv_cache_config: Any,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # In-flight READ transfer handles — ``{req_id: xfer_handle}``.
        self._inflight_reads: dict[str, Any] = {}

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Issue READ transfers for the consumer (skeleton).

        Production path: for each ``req_id`` in ``metadata.reqs_to_recv``,
        builds a NIXL READ descriptor list and posts the transfer.
        Skeleton: no-op (the protocol layer doesn't have NIXL).
        """
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until ``layer_name`` is loaded (skeleton: immediate)."""
        return None


__all__ = ["NixlPullConnectorWorker"]

"""Push-mode worker for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
push_worker.py`` (775 lines).  Scope reduction: push-mode-specific
state (WRITE transfer handle management, background ``nixl-push-writer``
thread) is reduced to the protocol surface.

References:
  * vLLM push worker:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_worker.py``.
"""

from __future__ import annotations

import threading
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


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-mode worker (producer issues WRITE transfers).

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.push_worker
    .NixlPushConnectorWorker``.  Skeleton — production push-mode
    specifics (NIXL WRITE xfer handle creation, background
    ``nixl-push-writer`` thread, heartbeat loop) are deferred until
    C251.
    """

    def __init__(
        self,
        vllm_config: Any,
        engine_id: str,
        kv_cache_config: Any,
    ) -> None:
        super().__init__(vllm_config, engine_id, kv_cache_config)
        # In-flight WRITE transfer handles — ``{req_id: xfer_handle}``.
        self._inflight_writes: dict[str, Any] = {}
        # Producer-side KV cache mirror (host buffer) — populated by
        # ``save_kv_layer`` calls during prefill.
        self._host_kv_buffers: dict[str, Any] = {}

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: Any,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Stash a layer's KV for the next WRITE transfer (skeleton).

        Production path: copies the layer tensor into the host
        buffer + records the descriptor for the WRITE xfer.  Skeleton:
        just records the layer name.
        """
        self._host_kv_buffers[layer_name] = kv_layer

    def wait_for_save(self) -> None:
        """Block until all pending WRITE transfers complete (skeleton: immediate)."""
        return None

    def start_load_kv(self, metadata: NixlConnectorMetadata) -> None:
        """Drive push-mode processing on the worker (skeleton).

        Production path: enqueues the metadata's registrations onto
        the background ``nixl-push-writer`` thread which polls NIXL
        notifs without engine-thread involvement.  Skeleton: no-op.
        """
        return None


__all__ = ["NixlPushConnectorWorker"]

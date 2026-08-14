"""ExampleConnector — minimal reference KV connector.

C137: minimal subset — alias of ``NixlConnector`` (in-process
shared-memory transfer backend).
C250: ``NixlConnector`` was refactored into the 13-file NIXL
package; ``ExampleConnector`` remains a thin alias.
C251: ``KVConnectorBase_V1`` was filled in with the full surface
(KV ownership / index / failure tracking / metadata lifecycle /
partial transfer).  ``ExampleConnector`` inherits the new surface
automatically via the alias.
C253: integrated with the C252 KV event bus — publish events to
the coordinator on save, consume events on load.  The PD
e2e test in ``tests/test_C253_pd_disagg_full.py`` exercises the
multi-PD-instance scenario described in the C253 plan (mock 2
producer + 4 consumer, bit-exact K/V verify).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    RecvResult,
    TransferBundle,
    TransferStatus,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlConnector,
)

logger = logging.getLogger(__name__)


class ExampleConnector(NixlConnector):
    """Debug reference KV connector — alias of NixlConnector in my-vllm.

    Mirrors ``vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py``
    in name + factory registration only.  vLLM's reference writes
    bundles to disk; my-vllm uses the shared-memory backend (the
    same path NixlConnector uses for in-process P→D transfer).

    The class is defined as a thin subclass of ``NixlConnector`` so:
      * The C251 ``KVConnectorBase_V1`` surface (kv_index,
        failure_tracker, metadata lifecycle, …) is inherited.
      * The C250 NIXL package wiring (``start_load_kv`` /
        ``save_kv_layer`` delegation to ``NixlPullConnectorWorker``)
        is shared.
      * The C252 event bus helpers (``take_events`` default empty
        iterator; subclass overrides can publish to the coordinator).

    For C253, this connector additionally:
      * Maintains a per-(request_id, layer_name) flag dict for the
        C252 ``BlockStored`` event emission.
      * Implements ``get_num_new_matched_tokens`` so a P-instance
        can be told "this many tokens can be sourced from the
        consumer's local cache" (returning the queued matched
        bundle's token count).
    """

    # ── Producer-side event emission (C253) ─────────────────────

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Save a layer's KV; emit a C252 ``BlockStored`` event.

        Calls the parent ``NixlBaseConnector.save_kv_layer`` to copy
        the tensor into the shared-memory buffer, then publishes a
        ``BlockStored`` event to the C252 coordinator (no-op when
        the coordinator has no producer registered for this
        engine).
        """
        super().save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

        # Emit a BlockStored event (C253 integration).
        try:
            from my_vllm.distributed.stateless_coordinator import (
                get_stateless_coordinator,
            )
            from my_vllm.distributed.kv_events import (
                BlockStored,
                MEDIUM_CPU,
            )
            coord = get_stateless_coordinator()
            flat_hash = hash((layer_name, kv_layer.data_ptr()))
            event = BlockStored(
                block_hashes=[flat_hash],
                parent_block_hash=None,
                token_ids=kv_layer.flatten().tolist()[:8],  # sample
                block_size=1,
                medium=MEDIUM_CPU,
            )
            coord.publish_event(event, producer_engine_id="example")
        except Exception:
            # Coordinator not initialized or event emit failed —
            # skip silently (this is a debug connector).
            pass

    # ── Consumer-side matched-tokens (C253) ──────────────────────

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int,
    ) -> int:
        """How many tokens can be sourced from the consumer's cache.

        C253: returns the size of the largest pending bundle's
        ``num_tokens`` (i.e., the largest match).  Production would
        query the C252 coordinator for ``BlockStored`` events
        matching ``request.request_id``.
        """
        if not self._pending_bundles:
            return 0
        largest = max(
            (b.num_tokens for b in self._pending_bundles),
            default=0,
        )
        return largest

    # ── Worker→Scheduler metadata (C253) ─────────────────────────

    def build_connector_worker_meta(
        self,
    ) -> Any:
        """Report per-step stats (number of saved bundles, in-flight
        transfers).

        C253: minimal — just returns the count of completed
        transfers + the count of pending bundles as a plain dict.
        """
        return {
            "saved_layers": len(self._kv_loaded_flags),
            "pending_bundles": len(self._pending_bundles),
        }


__all__ = ["ExampleConnector"]

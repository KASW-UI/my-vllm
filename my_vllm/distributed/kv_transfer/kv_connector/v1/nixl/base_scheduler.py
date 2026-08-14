"""Base scheduler for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
base_scheduler.py`` (466 lines).  Scope reductions:

  * All ``self._nixl_*`` NIXL-runtime calls — replaced with
    ``_raise_nixl_required()`` (the protocol layer doesn't
    need NIXL to instantiate, only to perform an actual RDMA
    transfer).
  * The full PD-disaggregation handshake (ZMQ listener) — kept
    as a stub (the production wire-up is in C251 / C252 / C253).
  * ``get_xfer_handshake_metadata`` + ``set_xfer_handshake_metadata``
    — kept verbatim (this is the **out-of-band** handshake that
    carries the engine ID + NIXL descriptor between P/D, not
    NIXL itself).

The scheduler-side protocol:

  1. ``on_new_request`` — register a new prefill request that may
     need KV transfer; tracks the request until it has a remote
     consumer.
  2. ``build_connector_meta`` — assemble the per-step
     ``NixlConnectorMetadata`` that the worker will act on.
  3. ``update_state_after_alloc`` — given a block allocation,
     decide how many tokens the consumer should source externally.
  4. ``request_finished`` — cleanup when a request finishes.
  5. ``get_num_new_matched_tokens`` — return 0 when no remote
     KV is available (the consumer will recompute).

References:
  * vLLM base scheduler:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorRole,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _raise_nixl_required(op: str) -> None:
    """Guard for NIXL-runtime operations that the protocol skeleton
    cannot execute without the vendored NIXL library."""
    raise RuntimeError(
        f"NIXL runtime not available: '{op}' requires the vendored "
        f"nixl library + RDMA hardware (C250 protocol-only). "
        f"Use the C242 SHM broadcast primitive for an in-process "
        f"substitute during development."
    )


class NixlBaseConnectorScheduler:
    """Base scheduler shared by pull and push variants.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
    .NixlBaseConnectorScheduler``.  Subclasses (pull / push) override
    the per-mode-specific methods; the base class holds the
    per-engine handshake state.
    """

    def __init__(
        self,
        vllm_config: Any,
        engine_id: str,
        kv_cache_config: Any,
    ) -> None:
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config

        # Out-of-band handshake state — exchanged once per remote
        # engine (host:port + NIXL descriptor).  In production this
        # is filled in by the NIXL handshake listener; in the
        # protocol skeleton it's settable via ``set_xfer_handshake_metadata``.
        self._remote_engines: dict[str, Any] = {}
        self._xfer_handshake_metadata: Any = None

        # Per-request transfer state — keyed by ``request_id``.
        self._pending_saves: dict[str, Any] = {}

        # Push-mode only — heartbeat bookkeeping.
        self._heartbeat_tasks: dict[str, Any] = {}

    # ── Handshake ────────────────────────────────────────────────

    @property
    def xfer_handshake_metadata(self) -> Any:
        """Handshake payload for this engine (set during init)."""
        return self._xfer_handshake_metadata

    def set_xfer_handshake_metadata(self, metadata: Any) -> None:
        """Set the handshake metadata (called by the worker after
        NIXL init completes)."""
        self._xfer_handshake_metadata = metadata

    def add_remote_engine(self, engine_id: str, metadata: Any) -> None:
        """Register a remote engine's handshake metadata."""
        self._remote_engines[engine_id] = metadata

    # ── Per-request lifecycle ─────────────────────────────────────

    def on_new_request(self, request: Any) -> None:
        """Called when a new request is admitted.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        .NixlBaseConnectorScheduler.on_new_request``.  Stashes the
        request for later ``build_connector_meta`` calls.
        """
        request_id = getattr(request, "request_id", str(request))
        self._pending_saves[request_id] = request

    def get_sw_clipped_blocks(self, block_ids: Any) -> Any:
        """Trim block IDs to the sliding-window length, if any.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        .NixlBaseConnectorScheduler.get_sw_clipped_blocks``.
        Production: returns the last ``sw_blocks`` blocks.  Skeleton:
        identity (no sliding window).
        """
        return block_ids

    # ── Scheduler integration ─────────────────────────────────────

    def build_connector_meta(
        self, scheduler_output: Any,
    ) -> KVConnectorMetadata | None:
        """Build the per-step connector metadata for the worker.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        .NixlBaseConnectorScheduler.build_connector_meta``.  Walks the
        scheduler output and populates ``reqs_to_save`` /
        ``reqs_to_recv`` from the pending request state.

        Skeleton: returns an empty ``NixlConnectorMetadata`` so the
        worker has a valid metadata object to dispatch on.
        """
        meta = NixlConnectorMetadata()
        return meta

    def update_connector_output(self, connector_output: Any) -> None:
        """Update scheduler state from the worker's connector output."""
        # No-op in the skeleton (the worker would populate
        # ``reqs_to_send`` with release deadlines).
        return None

    def has_pending_push_work(self) -> bool:
        """Whether the push variant has unfinished work (skeleton)."""
        return False

    def get_num_new_matched_tokens(
        self,
        request: Any,
        num_computed_tokens: int,
    ) -> int:
        """How many tokens can be sourced from external cache.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler
        .NixlBaseConnectorScheduler.get_num_new_matched_tokens``.
        Skeleton: 0 (no remote KV matched).
        """
        return 0

    def update_state_after_alloc(
        self,
        request: Any,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        """Update connector state after block allocation."""
        return None

    def request_finished(self, request: Any, block_ids: Any) -> None:
        """Cleanup when a request finishes."""
        request_id = getattr(request, "request_id", str(request))
        self._pending_saves.pop(request_id, None)
        self._heartbeat_tasks.pop(request_id, None)

    # ── Lifecycle ─────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release handshake state and stop background tasks."""
        self._remote_engines.clear()
        self._pending_saves.clear()
        self._heartbeat_tasks.clear()


__all__ = ["NixlBaseConnectorScheduler"]

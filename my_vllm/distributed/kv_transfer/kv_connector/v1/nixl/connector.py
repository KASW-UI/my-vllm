"""NIXL connector facades.

C250: refactored from the C131-era single-class ``NixlConnector`` to
mirror vLLM's three-class structure:

  * :class:`NixlBaseConnector` – common logic shared by pull and push.
  * :class:`NixlPullConnector` – pull-based (READ) KV transfer.
  * :class:`NixlPushConnector` – push-based (WRITE) KV transfer.
  * :class:`NixlConnector` – backward-compatible alias for
    :class:`NixlPullConnector`.

The local shared-memory backend (C131/C133-era) is preserved inside
``NixlBaseConnector`` so the connector is exercisable in dev /
single-process tests without RDMA hardware or the vendored NIXL
library.  In production, ``NixlPullConnectorWorker`` /
``NixlPushConnectorWorker`` would override the per-layer save /
load methods to use NIXL's GPU-direct RDMA.

References:
  * vLLM nixl connector:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from my_vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    RecvResult,
    TransferBundle,
    TransferStatus,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
    NixlPullConnectorScheduler,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
)

logger = logging.getLogger(__name__)


class NixlBaseConnector(KVConnectorBase_V1):
    """Base connector with shared local-backend logic.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.connector
    .NixlBaseConnector``.  The local shared-memory backend (kept from
    C131/C133) makes the connector exercisable on a single dev box
    without RDMA hardware.  Production deployments replace the
    ``save_kv_layer`` / ``start_load_kv`` paths with NIXL RDMA
    calls (deferred until NIXL is vendored).
    """

    # ── Local backend knobs ──────────────────────────────────────
    _DEFAULT_NUM_LAYERS = 12
    _DEFAULT_BUFFER_SIZE = 1024 * 1024 * 100  # 100MB per layer
    _DEFAULT_TRANSFER_TIMEOUT = 30.0

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
        num_layers: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._num_layers = num_layers or self._DEFAULT_NUM_LAYERS
        self._buffer_size = buffer_size or self._DEFAULT_BUFFER_SIZE
        self._transfer_timeout = self._DEFAULT_TRANSFER_TIMEOUT

        # Per-layer KV buffers: ``{layer_name: torch.Tensor (shared)}``.
        self._kv_buffers: dict[str, torch.Tensor] = {}
        self._kv_loaded_flags: dict[str, bool] = {}

        # Transfer queue — producer-side pending bundles.
        self._pending_bundles: list[TransferBundle] = []
        # Transfer queue — consumer-side ready bundles.
        self._received_bundles: list[TransferBundle] = []
        # ACK channel — consumer → producer.
        self._acks: dict[str, bool] = {}

        # Timing — per-request transfer start timestamps.
        self._transfer_start: dict[str, float] = {}

        # Scheduler + Worker — populated by the role-specific subclass.
        self.connector_scheduler: Any = None
        self.connector_worker: Any = None

    # ── Producer API (Prefill Worker) ────────────────────────────

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Save one layer's KV cache for transfer (local backend).

        Mirrors the C133-era local implementation: copies the KV
        tensor into a shared-memory buffer accessible by the
        consumer.  Production path overrides this to issue a NIXL
        WRITE transfer instead.
        """
        if layer_name not in self._kv_buffers:
            buf = torch.empty(
                kv_layer.numel(), dtype=kv_layer.dtype, device="cpu",
            )
            buf.share_memory_()
            self._kv_buffers[layer_name] = buf

        buf = self._kv_buffers[layer_name]
        flat = kv_layer.flatten().cpu()
        if flat.numel() <= buf.numel():
            buf[:flat.numel()].copy_(flat)
        else:
            logger.warning(
                "KV layer %s size %d exceeds buffer %d, truncating",
                layer_name, flat.numel(), buf.numel(),
            )
            buf.copy_(flat[:buf.numel()])

        self._kv_loaded_flags[layer_name] = True

    def wait_for_save(self) -> None:
        """Block until all pending saves complete (local: synchronous)."""
        return None

    # ── Consumer API (Decode Worker) ─────────────────────────────

    def start_load_kv(
        self, forward_context: Any = None, **kwargs: Any,
    ) -> None:
        """Begin loading KV blocks from external source.

        Skeleton: no-op (the actual implementation lives in the
        pull / push subclass ``start_load_kv`` which delegates to
        ``self.connector_worker.start_load_kv(metadata)``).
        """
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until ``layer_name`` is loaded (local: poll shared flag)."""
        max_wait = 10.0
        start = time.monotonic()
        while layer_name not in self._kv_loaded_flags:
            if time.monotonic() - start > max_wait:
                logger.warning("Timeout waiting for layer %s", layer_name)
                return
            time.sleep(0.01)

    # ── Transfer management ──────────────────────────────────────

    def enqueue_transfer(self, bundle: TransferBundle) -> None:
        """Enqueue a KV transfer request (producer-side setup)."""
        self._pending_bundles.append(bundle)
        self._transfer_start[bundle.request_id] = time.monotonic()

    def get_bundle_kv(self, layer_name: str) -> torch.Tensor | None:
        """Consumer: retrieve a saved KV layer tensor."""
        return self._kv_buffers.get(layer_name)

    # ── Failure handling ─────────────────────────────────────────

    def check_transfer_done(self, timeout: float | None = None) -> RecvResult:
        """Poll for completed transfers with timeout detection."""
        if timeout is not None:
            self._transfer_timeout = timeout

        bundles: list[TransferBundle] = list(self._received_bundles)
        self._received_bundles.clear()

        invalid_ids: set[str] = set()
        valid_prefix: dict[str, int] = {}

        still_pending: list[TransferBundle] = []
        for bundle in self._pending_bundles:
            elapsed = time.monotonic() - self._transfer_start.get(
                bundle.request_id, time.monotonic(),
            )
            if elapsed < self._transfer_timeout:
                still_pending.append(bundle)
            else:
                invalid_ids.add(bundle.request_id)
                valid_prefix[bundle.request_id] = 0

        self._pending_bundles = still_pending

        status = (
            TransferStatus.SUCCESS if not invalid_ids else TransferStatus.TIMEOUT
        )
        return RecvResult(
            status=status,
            bundles=bundles,
            invalid_req_ids=invalid_ids,
            valid_prefix_tokens=valid_prefix,
        )

    def send_ack(self, request_id: str, success: bool) -> None:
        """Consumer → Producer: acknowledge transfer result."""
        self._acks[request_id] = success

    def drain_acks(self) -> dict[str, bool]:
        """Producer: collect ACKs from consumers."""
        acks = dict(self._acks)
        self._acks.clear()
        return acks

    # ── Scheduler integration ───────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int,
    ) -> int:
        """How many tokens can be sourced from external cache."""
        return 0

    def request_finished(self, request: Any, block_ids: Any) -> None:
        """Cleanup transfer state when request finishes.

        C251: the C251 base default also clears the KV index +
        failure tracker.  We call ``super().request_finished()``
        after the local cleanup so both layers stay in sync.
        """
        rid = (
            request.request_id
            if hasattr(request, "request_id")
            else str(request)
        )
        self._kv_loaded_flags.pop(rid, None)
        self._acks.pop(rid, None)
        self._transfer_start.pop(rid, None)
        # C251 hook: also clean up the KV index + failure tracker.
        super().request_finished(request, block_ids)

    def shutdown(self) -> None:
        """Release connector state."""
        if self.connector_worker is not None:
            self.connector_worker.shutdown()
            self.connector_worker = None
        if self.connector_scheduler is not None:
            self.connector_scheduler.shutdown()
            self.connector_scheduler = None


class NixlPullConnector(NixlBaseConnector):
    """Pull-based (READ) NIXL KV transfer connector.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.connector
    .NixlPullConnector``.  The consumer drives the READ: it issues
    NIXL READ transfers against the producer's saved KV buffer.

    In production, ``start_load_kv`` delegates to
    ``NixlPullConnectorWorker.start_load_kv(metadata)`` which posts
    NIXL xfer handles and polls completion notifications.
    """

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
        num_layers: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        super().__init__(
            vllm_config, role, kv_cache_config,
            num_layers=num_layers, buffer_size=buffer_size,
        )
        self.engine_id = ""
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPullConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config,
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlPullConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config,
            )

    def start_load_kv(
        self, forward_context: Any = None, **kwargs: Any,
    ) -> None:
        """Drive pull-mode loading on the worker (delegates)."""
        if self.connector_worker is None:
            return None
        meta = self._connector_metadata or NixlConnectorMetadata()
        self.connector_worker.start_load_kv(meta)


class NixlPushConnector(NixlBaseConnector):
    """Push-based (WRITE) NIXL KV transfer connector.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.connector
    .NixlPushConnector``.  The producer drives the WRITE: it issues
    NIXL WRITE transfers to the consumer's allocated blocks.

    In production, ``start_load_kv`` enqueues registrations onto the
    background ``nixl-push-writer`` thread which polls NIXL notifs
    without engine-thread involvement.
    """

    def __init__(
        self,
        vllm_config: Any,
        role: KVConnectorRole,
        kv_cache_config: Any = None,
        num_layers: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        super().__init__(
            vllm_config, role, kv_cache_config,
            num_layers=num_layers, buffer_size=buffer_size,
        )
        self.engine_id = ""
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlPushConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config,
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NixlPushConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config,
            )
        else:
            raise ValueError(f"Unsupported KVConnectorRole: {role}")

    def start_load_kv(
        self, forward_context: Any = None, **kwargs: Any,
    ) -> None:
        """Drive push-mode processing on the worker (delegates)."""
        if self.connector_worker is None:
            return None
        meta = self._connector_metadata or NixlConnectorMetadata()
        self.connector_worker.start_load_kv(meta)


# Backward compatibility: NixlConnector is the pull-based connector.
NixlConnector = NixlPullConnector


__all__ = [
    "NixlBaseConnector",
    "NixlConnector",
    "NixlPullConnector",
    "NixlPushConnector",
]

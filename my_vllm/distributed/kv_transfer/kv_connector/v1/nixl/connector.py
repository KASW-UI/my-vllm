"""NIXL KV Connector — NVIDIA In-Xfer Library connector for KV cache transfer.

Mirrors vllm/distributed/kv_transfer/kv_connector/v1/nixl/ connector architecture.

Implements KVConnectorBase_V1 with shared-memory transfer (local backend).
In real vLLM, this uses NIXL RDMA for GPU-direct cross-node KV transfer.
The local implementation preserves the identical interface, demonstrating
the full connector lifecycle: save → transfer → load → ack.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

from my_vllm.v1.core.kv_connector import (
    KVConnectorBase_V1,
    RecvResult,
    TransferBundle,
    TransferStatus,
)

logger = logging.getLogger(__name__)


class NixlConnector(KVConnectorBase_V1):
    """NIXL-based KV cache connector (local shared-memory implementation).

    In production vLLM, NIXL provides GPU-direct RDMA for cross-node
    KV transfer in disaggregated prefill/decode serving. This local
    implementation uses torch shared memory to demonstrate the full
    interface without requiring CUDA IPC/RDMA hardware setup.

    Architecture:
      Producer (Prefill):  save_kv_layer() → buffer → wait_for_save()
      Consumer (Decode):   start_load_kv() → poll for bundles → decode
    """

    def __init__(
        self,
        num_layers: int = 12,
        buffer_size: int = 1024 * 1024 * 100,  # 100MB per layer
    ):
        self._num_layers = num_layers
        self._buffer_size = buffer_size

        # Per-layer KV buffers: {layer_name: torch.Tensor}
        self._kv_buffers: dict[str, torch.Tensor] = {}
        self._kv_loaded_flags: dict[str, bool] = {}

        # Transfer queue
        self._pending_bundles: list[TransferBundle] = []
        self._received_bundles: list[TransferBundle] = []
        self._acks: dict[str, bool] = {}

        # Timing
        self._transfer_start: dict[str, float] = {}
        self._transfer_timeout: float = 30.0

    # ------------------------------------------------------------------
    # Producer API (Prefill Worker)
    # ------------------------------------------------------------------

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Save one layer's KV cache for transfer.

        Mirrors vllm NIXL connector: copies the KV tensor for this layer
        into a shared buffer accessible by the consumer.
        """
        if layer_name not in self._kv_buffers:
            buf = torch.empty(
                kv_layer.numel(),
                dtype=kv_layer.dtype,
                device="cpu",
            )
            # Share memory for inter-process access
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
        """Block until all pending save operations complete.

        In local mode, all saves are synchronous so this is a no-op.
        In real NIXL, this waits for RDMA completion.
        """
        # Mark all layers as ready for the consumer
        pass

    # ------------------------------------------------------------------
    # Consumer API (Decode Worker)
    # ------------------------------------------------------------------

    def start_load_kv(
        self,
        forward_context: Any = None,
        **kwargs: Any,
    ) -> None:
        """Begin loading KV blocks from the producer.

        Checks which bundles are ready and moves them to _received_bundles.
        """
        ready: list[TransferBundle] = []
        still_pending: list[TransferBundle] = []

        for bundle in self._pending_bundles:
            if bundle.request_id in self._kv_loaded_flags:
                ready.append(bundle)
            else:
                elapsed = time.monotonic() - self._transfer_start.get(
                    bundle.request_id, time.monotonic()
                )
                if elapsed < self._transfer_timeout:
                    still_pending.append(bundle)
                else:
                    logger.warning(
                        "KV transfer timeout for %s (%.1fs)",
                        bundle.request_id, elapsed,
                    )

        self._pending_bundles = still_pending
        self._received_bundles = ready

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until a specific layer's KV data is available."""
        max_wait = 10.0
        start = time.monotonic()
        while layer_name not in self._kv_loaded_flags:
            if time.monotonic() - start > max_wait:
                logger.warning(
                    "Timeout waiting for layer %s", layer_name,
                )
                return
            time.sleep(0.01)

    # ------------------------------------------------------------------
    # Transfer management
    # ------------------------------------------------------------------

    def enqueue_transfer(self, bundle: TransferBundle) -> None:
        """Enqueue a KV transfer request (producer-side setup)."""
        self._pending_bundles.append(bundle)
        self._transfer_start[bundle.request_id] = time.monotonic()

    def get_bundle_kv(
        self, layer_name: str,
    ) -> torch.Tensor | None:
        """Consumer: retrieve a saved KV layer tensor."""
        return self._kv_buffers.get(layer_name)

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def check_transfer_done(self, timeout: float | None = None) -> RecvResult:
        """Poll for completed transfers with timeout detection."""
        if timeout is not None:
            self._transfer_timeout = timeout

        bundles: list[TransferBundle] = list(self._received_bundles)
        self._received_bundles.clear()

        invalid_ids: set[str] = set()
        valid_prefix: dict[str, int] = {}

        # Check pending for timeouts
        still_pending: list[TransferBundle] = []
        for bundle in self._pending_bundles:
            elapsed = time.monotonic() - self._transfer_start.get(
                bundle.request_id, time.monotonic()
            )
            if elapsed < self._transfer_timeout:
                still_pending.append(bundle)
            else:
                invalid_ids.add(bundle.request_id)
                valid_prefix[bundle.request_id] = 0

        self._pending_bundles = still_pending

        status = TransferStatus.SUCCESS if not invalid_ids else TransferStatus.TIMEOUT
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

    # ------------------------------------------------------------------
    # Scheduler integration
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> int:
        """How many tokens can be sourced from external cache."""
        return 0

    def request_finished(self, request: Any, block_ids: Any) -> None:
        """Cleanup transfer state when request finishes."""
        rid = (
            request.request_id
            if hasattr(request, "request_id")
            else str(request)
        )
        self._kv_loaded_flags.pop(rid, None)
        self._acks.pop(rid, None)
        self._transfer_start.pop(rid, None)

"""KVConnectorBase_V1 — KV cache block transfer base class.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/base.py:171.

Defines:
  - Role enum (SCHEDULER / WORKER) for connector location in the runtime.
  - Abstract metadata shapes (Handshake / Connector / WorkerMetadata)
    with `aggregate()` for cross-worker merging.
  - `KVConnectorBase_V1` — the runtime-side interface, with Worker
    primitives (start_load_kv / save_kv_layer / wait_for_layer_load /
    wait_for_save) and Scheduler primitives
    (get_num_new_matched_tokens / build_connector_meta / request_finished
    / update_state_after_alloc).
  - `NoopConnector` — concrete no-op implementation used when no
    connector is configured (the default for single-process my-vllm).

C133: migrated from the former
`my_vllm/v1/core/kv_connector.py` (172 lines) into the vLLM-aligned
file-tree location. This file was previously empty (C112/C131 left a
shell waiting for the kv_connector migration).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


# ── Cross-backend helpers ──────────────────────────────────────────────

# s_tensor_list, d_tensor_list, s_indices, d_indices, direction
CopyBlocksOp = Callable[
    [
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        list[int],
        list[int],
        Literal["h2d", "d2h"],
    ],
    None,
]


# ── HMA (Hybrid Memory Allocator) support ──────────────────────────────


class SupportsHMA(ABC):
    """Marker ABC for connectors that support hybrid memory allocator.

    Mirrors vLLM V1 v1/.../v1/base.py:85.
    Connectors that handle kv cache groups asynchronously (HMA)
    must implement `request_finished_all_groups`.
    """

    @abstractmethod
    def request_finished_all_groups(
        self,
        request: Any,
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished for all kv cache groups.

        Returns (free_now, transfer_params):
          - free_now: True if the connector wants blocks freed immediately;
            False if it will free them asynchronously.
          - transfer_params: optional metadata to attach to the request.
        """


def supports_hma(connector: Any) -> bool:
    """Return True if `connector` (class or instance) supports HMA."""
    if isinstance(connector, type):
        return issubclass(connector, SupportsHMA)
    return isinstance(connector, SupportsHMA)


# ── Role + metadata shapes ─────────────────────────────────────────────


class KVConnectorRole(enum.Enum):
    """Where the connector instance lives in the runtime.

    Mirrors vLLM V1 v1/.../v1/base.py:124.
    """

    # Connector running in the scheduler process.
    SCHEDULER = 0

    # Connector running in the worker process.
    WORKER = 1


class KVConnectorHandshakeMetadata(ABC):
    """Metadata used for out-of-band connector handshake between P/D workers.

    Mirrors vLLM V1 v1/.../v1/base.py:132. Must be serializable.
    """


class KVConnectorMetadata(ABC):
    """Abstract metadata sent Scheduler KVConnector → Worker KVConnector.

    Mirrors vLLM V1 v1/.../v1/base.py:141.
    """


class KVConnectorWorkerMetadata(ABC):
    """Abstract metadata sent Worker KVConnector → Scheduler KVConnector.

    Mirrors vLLM V1 v1/.../v1/base.py:150.

    Each worker can output its own metadata; for a single engine step,
    all metadata objects are aggregated via `aggregate()` before being
    passed to the Scheduler connector.
    """

    @abstractmethod
    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        """Aggregate this metadata with another worker's metadata."""


# ── Per-transfer bundle + result (PD transfer payloads) ───────────────


class KVTransferMethod(enum.Enum):
    """Transport method for KV block transfer."""

    SHARED_MEMORY = "shared_memory"
    NCCL = "nccl"
    RDMA = "rdma"


class TransferStatus(enum.Enum):
    """Result of a KV block transfer check."""

    SUCCESS = 0
    TIMEOUT = 1
    CORRUPTION = 2


@dataclass
class TransferBundle:
    """A completed KV transfer for one request.

    Aligns with vLLM's implicit bundle: request + block_ids + tensors.
    """

    request_id: str
    block_ids: list[int]
    num_tokens: int
    kv_tensors_k: torch.Tensor | None = None
    kv_tensors_v: torch.Tensor | None = None


@dataclass
class RecvResult:
    """Result of `check_transfer_done()` on the consumer side.

    bundles:             successfully received requests ready for decode.
    invalid_req_ids:     requests whose transfer failed.
    valid_prefix_tokens: for failed requests, the prefix length that did
                         transfer successfully (used by recompute policy
                         to resume prefill from that position).
    """

    status: TransferStatus = TransferStatus.SUCCESS
    bundles: list[TransferBundle] = field(default_factory=list)
    invalid_req_ids: set[str] = field(default_factory=set)
    valid_prefix_tokens: dict[str, int] = field(default_factory=dict)


# ── Base connector ────────────────────────────────────────────────────


class KVConnectorBase_V1(ABC):
    """Cross-process/network KV block transfer base.

    Worker-side API (called from inside the model forward):
      start_load_kv / wait_for_layer_load
      save_kv_layer   / wait_for_save

    Scheduler-side API (called from inside the Scheduler):
      get_num_new_matched_tokens / update_state_after_alloc
      build_connector_meta       / request_finished
      take_events                / update_connector_output

    Failure-handling API (consumer side):
      check_transfer_done / send_ack / drain_acks
    """

    def __init__(
        self,
        vllm_config: "VllmConfig | None" = None,
        role: KVConnectorRole | None = None,
        kv_cache_config: Any = None,
    ):
        # Defaults to None so the no-op connector and test stubs can
        # construct without ceremony; vLLM requires non-None at runtime.
        self._vllm_config = vllm_config
        self._role = role if role is not None else KVConnectorRole.SCHEDULER
        self._kv_cache_config = kv_cache_config
        self._connector_metadata: KVConnectorMetadata | None = None

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    # ── Worker-side primitives ─────────────────────────────────────────

    @abstractmethod
    def start_load_kv(
        self, forward_context: Any = None, **kwargs: Any
    ) -> None:
        """Begin loading KV blocks from external source into paged KV buffer.

        Called before the forward pass in the decode worker.
        """

    @abstractmethod
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        """Save one layer's KV cache for transfer to another worker.

        Called inside each attention layer during prefill.
        """

    @abstractmethod
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until the KV for a specific layer is loaded."""

    @abstractmethod
    def wait_for_save(self) -> None:
        """Block until all pending save operations complete."""

    # ── P/D failure handling API ───────────────────────────────────────

    def send_ack(self, request_id: str, success: bool) -> None:
        """Consumer → Producer: acknowledge transfer result.

        Producer uses this to determine when to free deferred blocks.
        """

    def check_transfer_done(self, timeout: float | None = None) -> RecvResult:
        """Poll for completed transfers with timeout detection.

        Returns `RecvResult` with successful bundles and failed request
        IDs. Subclasses override this to implement actual polling.
        """
        return RecvResult(status=TransferStatus.SUCCESS)

    def drain_acks(self) -> dict[str, bool]:
        """Producer: collect ACKs from the consumer.

        Returns {request_id: success} for all pending transfers that
        have received an ACK (success or failure) from the consumer.
        """
        return {}

    # ── Scheduler-side API ─────────────────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> int:
        """Calculate how many tokens can be sourced from external cache."""
        return 0

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Update connector state after block allocation."""
        pass

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        """Build connector metadata for the Worker."""
        return None

    def request_finished(self, request: Any, block_ids: Any) -> None:
        """Cleanup when a request finishes."""
        pass


class NoopConnector(KVConnectorBase_V1):
    """No-op connector: all operations are identity.

    Used as the default when no KV transfer is configured
    (the single-process / single-engine default).
    """

    def start_load_kv(
        self, forward_context: Any = None, **kwargs: Any
    ) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any = None,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self) -> None:
        pass
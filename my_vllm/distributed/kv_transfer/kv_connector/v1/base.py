"""KVConnectorBase_V1 — KV cache block transfer base class.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/base.py:171.

Defines:
  - Role enum (SCHEDULER / WORKER) for connector location in the runtime.
  - Abstract metadata shapes (Handshake / Connector / WorkerMetadata)
    with `aggregate()` for cross-worker merging.
  - KV ownership + index dataclasses for tracking which blocks belong
    to which request across the producer / consumer pair.
  - `KVConnectorBase_V1` — the runtime-side interface, with Worker
    primitives (start_load_kv / save_kv_layer / wait_for_layer_load /
    wait_for_save) and Scheduler primitives
    (get_num_new_matched_tokens / build_connector_meta / request_finished
    / update_state_after_alloc).
  - `NoopConnector` — concrete no-op implementation used when no
    connector is configured (the default for single-process my-vllm).

C133: migrated from the former
`my_vllm/v1/core/kv_connector.py` (172 lines) into the vLLM-aligned
file-tree location.
C251: filled in the missing KV ownership / metadata lifecycle /
failure retransmission / partial transfer / KV index surface that
was deferred at C133 time.  Aligned with vLLM's
``vllm/distributed/kv_transfer/kv_connector/v1/base.py:171-708``
(708 lines; my-vllm is now ~530 lines after the C251 additions).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
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


# ── KV ownership + index (C251 additions) ────────────────────────────


@dataclass
class KVOwnership:
    """Per-request KV ownership record.

    Tracks which local blocks are owned by a request and which engine
    (local or remote) currently holds the canonical copy.  Mirrors
    the implicit ownership state in vLLM's ``KVConnectorBase_V1``
    (decomposed into a dataclass for clarity).

    Attributes:
        request_id: Stable request identifier (base form, no random
            suffix — see ``nixl/utils.get_base_request_id``).
        local_block_ids: Local block IDs that this engine owns.
        owner_engine_id: Engine ID of the canonical owner (== self for
            local-owned blocks; == remote for foreign-owned blocks).
        state: Current transfer state.
        created_at: Wall-clock timestamp when the ownership was
            recorded (used for lease renewal / expiry in heartbeat).
    """

    request_id: str
    local_block_ids: list[int]
    owner_engine_id: str
    state: str = "owned"
    created_at: float = 0.0


class KVOwnershipState(enum.Enum):
    """KV block ownership lifecycle states.

    Mirrors the implicit state machine in vLLM's PD-disagg code paths.
    """

    # Block is allocated but not yet filled with tokens.
    ALLOCATED = "allocated"
    # Block is filled and owned by the local engine.
    OWNED = "owned"
    # Block is being transferred to a remote engine.
    TRANSFERRING = "transferring"
    # Block is leased to a remote engine (the local engine no longer
    # owns it; the remote engine has read access).
    LEASED = "leased"
    # Block is expired (the remote engine's lease ran out).
    EXPIRED = "expired"
    # Block is freed back to the free pool.
    FREED = "freed"


@dataclass
class KVIndexEntry:
    """Single (request_id, block_id) → ownership entry.

    The ``KVIndex`` maintains a 2-way lookup:
      * ``request_id → set[block_id]`` — for ``request_finished`` cleanup.
      * ``block_id → request_id`` — for fast block-level queries.
    """

    request_id: str
    block_id: int
    state: KVOwnershipState = KVOwnershipState.ALLOCATED


class KVIndex:
    """Two-way index over KV block ownership.

    C251: extracted from the implicit lookup tables that vLLM keeps
    inline in the scheduler connector.  The index supports:

      * **Add / Remove**: per-request block add/remove.
      * **Lookup by request**: ``get_blocks_for_request(request_id)``.
      * **Lookup by block**: ``get_request_for_block(block_id)``.
      * **Update state**: bulk state update for a request (used when a
        transfer completes / fails).
      * **Cleanup**: ``remove_request(request_id)`` removes all entries
        for a request.

    The index is intentionally pure-Python (no numpy / torch
    dependency) so it can be used in tests and in the scheduler
    thread without GIL contention.
    """

    def __init__(self) -> None:
        self._by_request: dict[str, dict[int, KVIndexEntry]] = {}
        self._by_block: dict[int, str] = {}

    # ── Mutations ──────────────────────────────────────────────────

    def add(self, request_id: str, block_ids: Iterable[int]) -> None:
        """Record ``request_id`` owning ``block_ids`` (in ALLOCATED state)."""
        request_entries = self._by_request.setdefault(request_id, {})
        for block_id in block_ids:
            entry = KVIndexEntry(
                request_id=request_id,
                block_id=block_id,
                state=KVOwnershipState.ALLOCATED,
            )
            request_entries[block_id] = entry
            self._by_block[block_id] = request_id

    def remove(self, request_id: str) -> list[int]:
        """Remove all entries for ``request_id``.  Returns the removed block IDs."""
        entries = self._by_request.pop(request_id, {})
        for block_id in entries:
            self._by_block.pop(block_id, None)
        return list(entries.keys())

    def remove_block(self, block_id: int) -> str | None:
        """Remove a single block.  Returns the request_id that owned it (or None)."""
        request_id = self._by_block.pop(block_id, None)
        if request_id is None:
            return None
        entries = self._by_request.get(request_id)
        if entries is not None:
            entries.pop(block_id, None)
            if not entries:
                self._by_request.pop(request_id, None)
        return request_id

    def update_state(
        self,
        request_id: str,
        new_state: KVOwnershipState,
        block_ids: Iterable[int] | None = None,
    ) -> int:
        """Update ownership state for one (or all) of a request's blocks.

        Returns the number of blocks updated.
        """
        entries = self._by_request.get(request_id)
        if entries is None:
            return 0
        if block_ids is None:
            for entry in entries.values():
                entry.state = new_state
            return len(entries)
        count = 0
        for block_id in block_ids:
            entry = entries.get(block_id)
            if entry is not None:
                entry.state = new_state
                count += 1
        return count

    # ── Queries ────────────────────────────────────────────────────

    def get_blocks_for_request(self, request_id: str) -> list[int]:
        """Return the block IDs owned by ``request_id``."""
        entries = self._by_request.get(request_id)
        if entries is None:
            return []
        return list(entries.keys())

    def get_request_for_block(self, block_id: int) -> str | None:
        """Return the request_id that owns ``block_id`` (or None)."""
        return self._by_block.get(block_id)

    def get_state(
        self,
        request_id: str,
        block_id: int | None = None,
    ) -> KVOwnershipState | None:
        """Return the ownership state for ``(request_id, block_id)``.

        When ``block_id`` is None, returns the state of the first
        block (or None if the request has no blocks)."""
        entries = self._by_request.get(request_id)
        if not entries:
            return None
        if block_id is None:
            return next(iter(entries.values())).state
        entry = entries.get(block_id)
        return entry.state if entry is not None else None

    def get_blocks_in_state(
        self,
        state: KVOwnershipState,
    ) -> list[tuple[str, int]]:
        """Return all ``(request_id, block_id)`` pairs in ``state``."""
        result: list[tuple[str, int]] = []
        for request_id, entries in self._by_request.items():
            for block_id, entry in entries.items():
                if entry.state == state:
                    result.append((request_id, block_id))
        return result

    # ── Bookkeeping ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._by_block)

    def __contains__(self, block_id: int) -> bool:
        return block_id in self._by_block

    def clear(self) -> None:
        self._by_request.clear()
        self._by_block.clear()


@dataclass
class KVTransferFailure:
    """Per-request transfer failure record.

    C251: used by ``get_block_ids_with_load_errors`` and the failure
    retransmission path.  Mirrors vLLM's implicit ``FailureRecord``
    tuple (``(req_id, failed_block_ids, num_completed_tokens)``).
    """

    request_id: str
    failed_block_ids: list[int] = field(default_factory=list)
    num_completed_tokens: int = 0
    error_message: str = ""
    timestamp: float = 0.0


class KVFailureTracker:
    """Tracks per-request transfer failures for retransmission.

    C251: my-vllm's protocol-level implementation of the failure
    retransmission loop.  When a transfer fails for some blocks
    (partial transfer), the consumer records the failed block IDs
    here.  The producer can then re-issue the failed blocks
    independently of the successful ones.

    Per the vLLM design: on failure, the producer retransmits only
    the failed blocks (not the whole request), and the consumer
    resumes from ``num_completed_tokens``.
    """

    def __init__(self, max_age_seconds: float = 60.0) -> None:
        self._failures: dict[str, KVTransferFailure] = {}
        self._max_age = max_age_seconds

    def record_failure(
        self,
        request_id: str,
        failed_block_ids: list[int],
        num_completed_tokens: int = 0,
        error_message: str = "",
    ) -> None:
        """Record a transfer failure (or extend an existing one)."""
        import time

        existing = self._failures.get(request_id)
        if existing is None:
            self._failures[request_id] = KVTransferFailure(
                request_id=request_id,
                failed_block_ids=list(failed_block_ids),
                num_completed_tokens=num_completed_tokens,
                error_message=error_message,
                timestamp=time.monotonic(),
            )
        else:
            existing.failed_block_ids.extend(failed_block_ids)
            existing.timestamp = time.monotonic()
            if num_completed_tokens > existing.num_completed_tokens:
                existing.num_completed_tokens = num_completed_tokens
            if error_message:
                existing.error_message = error_message

    def get_failure(self, request_id: str) -> KVTransferFailure | None:
        """Return the failure record for ``request_id`` (or None)."""
        return self._failures.get(request_id)

    def has_failure(self, request_id: str) -> bool:
        return request_id in self._failures

    def get_failed_block_ids(self, request_id: str) -> list[int]:
        """Return the failed block IDs for ``request_id`` (or [])."""
        rec = self._failures.get(request_id)
        return list(rec.failed_block_ids) if rec else []

    def clear(self, request_id: str) -> None:
        """Clear the failure record for ``request_id`` (after retransmit)."""
        self._failures.pop(request_id, None)

    def prune_stale(self, now: float | None = None) -> int:
        """Drop failure records older than ``max_age_seconds``.  Returns the count pruned."""
        import time

        if now is None:
            now = time.monotonic()
        stale = [
            rid for rid, rec in self._failures.items()
            if now - rec.timestamp > self._max_age
        ]
        for rid in stale:
            self._failures.pop(rid, None)
        return len(stale)

    def __len__(self) -> int:
        return len(self._failures)


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
      get_block_ids_with_load_errors (C251 — partial transfer support)

    KV ownership / index API (C251):
      register_kv_caches / handle_preemptions / bind_gpu_block_pool
      get_handshake_metadata / set_xfer_handshake_metadata
      build_connector_worker_meta / on_new_request
      get_finished / get_finished_count / take_events
      has_pending_push_work / shutdown / reset_cache
      get_kv_connector_stats / update_connector_output
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

        # C251: KV ownership + index + failure tracker.
        self._kv_index: KVIndex = KVIndex()
        self._failure_tracker: KVFailureTracker = KVFailureTracker()

        # C251: GPU block pool binding (for ``bind_gpu_block_pool``).
        self._gpu_block_pool: Any = None

        # C251: KV cache registration (set by ``register_kv_caches``).
        self._kv_caches: dict[str, torch.Tensor] = {}

        # C251: handshake metadata cache (set by ``set_xfer_handshake_metadata``).
        self._xfer_handshake_metadata: KVConnectorHandshakeMetadata | None = None

        # C251: finished / failed transfer tracking (per-step).
        self._finished_req_ids: set[str] = set()
        self._failed_req_ids: set[str] = set()

        # C251: optional stats hook (subclasses can override).
        self._stats: Any = None

    @property
    def role(self) -> KVConnectorRole:
        return self._role

    @property
    def kv_index(self) -> KVIndex:
        """The connector's KV block index (C251)."""
        return self._kv_index

    @property
    def failure_tracker(self) -> KVFailureTracker:
        """The connector's failure tracker (C251)."""
        return self._failure_tracker

    # ── Metadata lifecycle (C251) ─────────────────────────────────────

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        """Bind the per-step connector metadata (called by the engine).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.bind_connector_metadata``.
        """
        self._connector_metadata = connector_metadata

    def clear_connector_metadata(self) -> None:
        """Clear the per-step connector metadata (called after step)."""
        self._connector_metadata = None

    def _get_connector_metadata(self) -> KVConnectorMetadata | None:
        """Return the bound connector metadata (or None)."""
        return self._connector_metadata

    def has_connector_metadata(self) -> bool:
        """Return True iff connector metadata is bound."""
        return self._connector_metadata is not None

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

    # ── KV cache registration (C251) ──────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register KV cache tensors for descriptor building.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.register_kv_caches`` (skeleton: just
        stores the references; subclass overrides compute NIXL
        descriptors).
        """
        self._kv_caches = dict(kv_caches)

    def register_cross_layers_kv_cache(self, kv_cache: torch.Tensor) -> None:
        """Register a single cross-layer packed KV cache.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.register_cross_layers_kv_cache``.  Some
        backends (FA2 / FA3 / Triton-Attn with cross-layer blocks)
        store all layers' KV in a single packed tensor for cache
        reuse; the connector is told about it via this hook.
        """
        self._kv_caches["cross_layers"] = kv_cache

    def set_host_xfer_buffer_ops(self, copy_operation: CopyBlocksOp) -> None:
        """Set the host-transfer-buffer copy operation (skeleton)."""
        # Skeleton — production would dispatch this to the worker.
        self._copy_op = copy_operation

    # ── KV block pool binding (C251) ──────────────────────────────────

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        """Bind the GPU block pool (for ``handle_preemptions``).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.bind_gpu_block_pool``.  The connector
        uses this to translate block IDs between the producer and
        consumer when a request is preempted.
        """
        self._gpu_block_pool = gpu_block_pool

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Handle block preemptions (skeleton: no-op).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.handle_preemptions``.  Called by the
        scheduler when a request's blocks are preempted; the connector
        should mark the corresponding ``KVIndexEntry`` states as
        ``EXPIRED`` and re-issue the affected transfers if needed.
        """
        return None

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

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return the block IDs that failed to load (partial transfer).

        C251: enables partial-transfer recovery.  When a transfer
        partially succeeds (some blocks arrive, others don't), the
        consumer records the failed block IDs here; the scheduler
        then issues a recompute from ``valid_prefix_tokens`` while
        keeping the loaded blocks in place.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_block_ids_with_load_errors``.
        """
        failed_blocks: set[int] = set()
        for request_id in self._failed_req_ids:
            failed_blocks.update(
                self._failure_tracker.get_failed_block_ids(request_id),
            )
        return failed_blocks

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return ``(finished_req_ids, failed_req_ids)`` for the last step.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_finished``.  Clears the per-step
        state on read (consumers should poll once per step).
        """
        done, failed = self._finished_req_ids, self._failed_req_ids
        self._finished_req_ids = set()
        self._failed_req_ids = set()
        return done, failed

    def get_finished_count(self) -> int | None:
        """Return the count of finished transfers (or None).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_finished_count``.  Subclasses can
        override to return the actual count (e.g. NIXL notif polls).
        """
        return len(self._finished_req_ids) + len(self._failed_req_ids)

    # ── Handshake metadata (C251) ──────────────────────────────────────

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        """Return the handshake payload for this engine (or None).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_handshake_metadata``.  Subclasses
        override to compute the payload on demand (e.g. NIXL agent
        descriptor + compat hash).
        """
        return self._xfer_handshake_metadata

    def set_xfer_handshake_metadata(
        self, metadata: KVConnectorHandshakeMetadata,
    ) -> None:
        """Store the local handshake payload (called by the worker).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.set_xfer_handshake_metadata``.
        """
        self._xfer_handshake_metadata = metadata

    # ── Worker → Scheduler metadata (C251) ─────────────────────────────

    def build_connector_worker_meta(
        self,
    ) -> KVConnectorWorkerMetadata | None:
        """Build the worker→scheduler metadata for the next step.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.build_connector_worker_meta``.  Subclasses
        override to report per-worker stats (e.g. NIXL transfer
        telemetry aggregated by the worker).
        """
        return None

    # ── Scheduler-side API ─────────────────────────────────────────────

    def on_new_request(self, request: Any) -> None:
        """Hook called when a new request is admitted.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.on_new_request``.  Default: no-op.
        """
        return None

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> int:
        """Calculate how many tokens can be sourced from external cache."""
        return 0

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Update connector state after block allocation.

        C251: default implementation registers the newly allocated
        blocks in the KV index (so ``request_finished`` can find
        them).  Subclasses can override for richer bookkeeping.
        """
        request_id = getattr(request, "request_id", str(request))
        block_ids = list(blocks) if blocks else []
        if block_ids:
            self._kv_index.add(request_id, block_ids)
            self._kv_index.update_state(
                request_id, KVOwnershipState.ALLOCATED,
            )

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        """Build connector metadata for the Worker."""
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """Update connector state from the worker's connector output.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.update_connector_output``.  Default:
        no-op (subclasses extract per-step release deadlines / etc.).
        """
        return None

    def request_finished(self, request: Any, block_ids: Any) -> None:
        """Cleanup when a request finishes.

        C251: default implementation removes the request from the KV
        index + clears any failure record.  Subclasses can extend.
        """
        request_id = getattr(request, "request_id", str(request))
        # If the caller passes block_ids, remove them explicitly;
        # otherwise trust the KV index.
        if block_ids:
            for block_id in block_ids:
                self._kv_index.remove_block(block_id)
            # If the index has no more entries for this request, drop
            # the request_id key.
            if not self._kv_index.get_blocks_for_request(request_id):
                self._kv_index.remove(request_id)
        else:
            self._kv_index.remove(request_id)
        self._failure_tracker.clear(request_id)
        self._finished_req_ids.discard(request_id)
        self._failed_req_ids.discard(request_id)

    def take_events(self) -> Iterable[Any]:
        """Yield KV cache events queued by this connector (default: empty).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.take_events``.  Events are consumed by
        the metrics subsystem + KV event bus (C252).
        """
        return iter(())

    def has_pending_push_work(self) -> bool:
        """Whether the connector has unfinished push-mode work.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.has_pending_push_work``.
        """
        return False

    # ── Lifecycle (C251) ──────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release connector state (default: clear index + failures)."""
        self._kv_index.clear()
        self._failure_tracker._failures.clear()  # type: ignore[attr-defined]
        self._finished_req_ids.clear()
        self._failed_req_ids.clear()
        self._connector_metadata = None
        self._xfer_handshake_metadata = None
        self._kv_caches.clear()

    def reset_cache(self) -> bool | None:
        """Reset the connector's cache (default: True).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.reset_cache``.  Subclasses with remote
        state should return False to indicate the remote cache needs
        separate invalidation.
        """
        self._kv_index.clear()
        self._failure_tracker._failures.clear()  # type: ignore[attr-defined]
        return True

    # ── Stats (C251) ──────────────────────────────────────────────────

    def get_kv_connector_stats(self) -> Any:
        """Return the connector's stats object (or None).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_kv_connector_stats``.  Subclasses
        with stats (NIXL, …) override to return their stats.
        """
        return self._stats

    # ── Block layout hints (C251) ──────────────────────────────────────

    @classmethod
    def get_required_kv_cache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        """Return the KV cache layout this connector requires (or None).

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.get_required_kv_cache_layout``.  Subclasses
        override to declare "NHD" / "HND" / etc.  Default: any layout.
        """
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """Whether this connector prefers cross-layer block layout.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.base
        .KVConnectorBase_V1.prefer_cross_layer_blocks``.  Default: False.
        """
        return False


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
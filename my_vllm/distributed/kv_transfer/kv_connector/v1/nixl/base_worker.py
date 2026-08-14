"""Base worker for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
base_worker.py`` (2454 lines — the largest single file in the NIXL
connector package).  Scope reductions:

  * All ``self._nixl_*`` NIXL-runtime calls — replaced with
    ``_raise_nixl_required()``.  The actual RDMA path requires the
    vendored NIXL library + RDMA hardware, which is explicitly
    out of scope for the JD-aligned subset.
  * The huge block-descriptor computation (``_build_local_splits_from_plan``
    / ``_fa_desc_replicated`` / ``_build_fa_local`` / ``_build_fa_remote``
    / ``_stack_descs`` / ``_register_packed_kv_cache`` etc.) — replaced
    with stub methods.  The full descriptor logic is tightly coupled
    to NIXL's tensor-descriptor format and cannot be exercised
    without the runtime.
  * HMA (hybrid Mamba) split (``_build_mamba_local`` /
    ``_build_mamba_remote``) — stubbed (my-vllm has no MambaSpec
    KVCacheSpec).
  * Background ``nixl-push-writer`` thread — stubbed (push-mode
    only; needs NIXL notif polling).
  * Heartbeat listener thread — stubbed (push-mode only; uses ZMQ).

What is preserved (the protocol surface):

  * ``register_kv_caches`` — public hook called by the model runner
    after KV cache allocation.  Skeleton: records the cache
    references but doesn't register NIXL descriptors.
  * ``add_remote_agent`` — handshake completion for a remote engine.
    Skeleton: stores the remote handshake metadata.
  * ``save_kv_to_host`` / ``sync_recved_kv_to_device`` — public
    hooks called per step.  Skeleton: records the request metadata.
  * ``get_finished`` — public hook returning (success, failure) sets.
    Skeleton: empty (no transfers complete in protocol-only mode).
  * ``get_num_new_matched_tokens`` / ``update_state_after_alloc`` —
    scheduler-side hooks duplicated here for symmetry.
  * ``shutdown`` — releases handshake state.

References:
  * vLLM base worker:
    ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _raise_nixl_required(op: str) -> None:
    """Guard for NIXL-runtime operations that the protocol skeleton
    cannot execute without the vendored NIXL library."""
    raise RuntimeError(
        f"NIXL runtime not available: '{op}' requires the vendored "
        f"nixl library + RDMA hardware (C250 protocol-only)."
    )


class NixlBaseConnectorWorker:
    """Base worker shared by pull and push variants.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.base_worker
    .NixlBaseConnectorWorker``.  Subclasses (pull / push) override
    the per-mode ``start_load_kv`` / ``save_kv_layer`` /
    ``wait_for_layer_load`` / ``wait_for_save`` methods.
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

        # NIXL agent (would be ``nixl_agent`` from the vendored lib).
        # Stored as ``None`` because the agent creation requires the
        # NIXL runtime; the protocol skeleton doesn't instantiate it.
        self._nixl_agent: Any = None

        # Handshake state — exchanged once per remote engine.
        self._remote_handshakes: dict[str, Any] = {}
        self._xfer_handshake_metadata: Any = None

        # KV cache registration — filled by ``register_kv_caches``.
        self._kv_caches_base_addr: list[int] = []
        self._num_blocks: int = 0
        self._block_lens: list[int] = []
        self._block_size: int = 16

        # Per-step transfer state.
        self._done_reqs: set[str] = set()
        self._failed_reqs: set[str] = set()

        # Statistics — Prometheus deferred (C250 scope reduction);
        # only the in-memory counter is preserved.
        self._stats = NixlKVConnectorStats()

        # Background thread placeholders — push mode only.
        self._heartbeat_thread: threading.Thread | None = None
        self._push_writer_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

    # ── Handshake ────────────────────────────────────────────────

    @property
    def xfer_handshake_metadata(self) -> Any:
        """Handshake payload for this engine (computed by ``_compute_handshake``)."""
        return self._xfer_handshake_metadata

    def _compute_handshake(self, attn_backend_name: str, cross_layers_blocks: bool) -> Any:
        """Build the engine's handshake payload.

        Skeleton: returns a dict with the version + engine ID.  In
        production this would also embed the NIXL agent descriptor.
        """
        from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            NIXL_CONNECTOR_VERSION,
            NixlAgentMetadata,
            NixlHandshakePayload,
            compute_nixl_compatibility_hash,
        )

        compat_hash = compute_nixl_compatibility_hash(
            self.vllm_config, attn_backend_name, cross_layers_blocks,
        )
        # Agent metadata would be ``self._nixl_agent.get_agent_metadata()``
        # serialized via msgpack; skeleton uses a JSON placeholder
        # (msgpack isn't vendored, but JSON is sufficient for the
        # protocol skeleton — production would switch to msgpack).
        agent_meta = NixlAgentMetadata(
            engine_id=self.engine_id,
            agent_metadata=b"",
            kv_caches_base_addr=self._kv_caches_base_addr,
            device_id=0,
            num_blocks=self._num_blocks,
            block_lens=self._block_lens,
            kv_cache_layout="NHD",
            block_size=self._block_size,
            ssm_sizes=(0, 0),
            attn_backend_name=attn_backend_name,
            physical_blocks_per_logical_kv_block=1,
        )
        import json

        # Convert lists/tuples to JSON-safe lists.
        payload = {
            "engine_id": agent_meta.engine_id,
            "kv_caches_base_addr": agent_meta.kv_caches_base_addr,
            "device_id": agent_meta.device_id,
            "num_blocks": agent_meta.num_blocks,
            "block_lens": agent_meta.block_lens,
            "kv_cache_layout": agent_meta.kv_cache_layout,
            "block_size": agent_meta.block_size,
            "ssm_sizes": list(agent_meta.ssm_sizes),
            "attn_backend_name": agent_meta.attn_backend_name,
            "physical_blocks_per_logical_kv_block": (
                agent_meta.physical_blocks_per_logical_kv_block
            ),
        }
        encoded = json.dumps(payload).encode()
        self._xfer_handshake_metadata = NixlHandshakePayload(
            compatibility_hash=compat_hash,
            agent_metadata_bytes=encoded,
        )
        return self._xfer_handshake_metadata

    def add_remote_agent(self, engine_id: str, metadata: Any) -> None:
        """Store a remote engine's handshake metadata.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.base_worker
        .NixlBaseConnectorWorker.add_remote_agent``.  Skeleton: only
        records the metadata; does not register NIXL descriptors.
        """
        self._remote_handshakes[engine_id] = metadata

    # ── KV cache registration ────────────────────────────────────

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Record KV cache references for descriptor building.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.base_worker
        .NixlBaseConnectorWorker.register_kv_caches``.  Skeleton:
        stores the metadata; production would compute NIXL
        descriptors via ``_register_packed_kv_cache``.
        """
        for name, cache in kv_caches.items():
            self._kv_caches_base_addr.append(cache.data_ptr())
            # Each tensor's first dim is the block count.
            n_blocks = cache.shape[0]
            self._block_lens.append(cache.numel() * cache.element_size() // max(n_blocks, 1))
            self._num_blocks += n_blocks

    # ── Per-step operations ──────────────────────────────────────

    def save_kv_to_host(self, metadata: NixlConnectorMetadata) -> None:
        """Save the producer's KV blocks to host (skeleton).

        Production path: copies device → host buffer and posts NIXL
        WRITE transfers for the consumer to read.  Skeleton: no-op.
        """
        return None

    def sync_recved_kv_to_device(self, req_id: str, meta: Any) -> None:
        """Copy received KV from host buffer back to device (skeleton)."""
        return None

    def get_finished(self) -> tuple[set[str], set[str]]:
        """Return (done_reqs, failed_reqs) for the last step.

        Mirrors ``vllm.distributed.kv_transfer.kv_connector/v1/nixl.base_worker
        .NixlBaseConnectorWorker.get_finished``.  Skeleton: empty
        sets (no transfers complete in protocol-only mode).
        """
        done, failed = self._done_reqs, self._failed_reqs
        self._done_reqs = set()
        self._failed_reqs = set()
        return done, failed

    # ── Scheduler-side helpers ───────────────────────────────────

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int,
    ) -> int:
        """How many tokens can be sourced from remote KV (skeleton: 0)."""
        return 0

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int,
    ) -> None:
        """Update connector state after block allocation."""
        return None

    # ── Lifecycle ─────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Release handshake state + signal background threads."""
        self._shutdown.set()
        self._remote_handshakes.clear()
        self._kv_caches_base_addr.clear()
        self._block_lens.clear()


__all__ = ["NixlBaseConnectorWorker"]

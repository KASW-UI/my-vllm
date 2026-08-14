"""Metadata dataclasses and helpers for the NIXL connector.

C250: extracted from ``vllm/distributed/kv_transfer/kv_connector/v1/nixl/
metadata.py`` (233 lines).  Scope reductions:

  * ``vllm.__version__`` import — replaced with a module-level
    constant ``_MY_VLLM_VERSION``.
  * ``vllm.config.utils.hash_factors`` — replaced with a stdlib
    ``hashlib.sha256`` implementation (the factor set is the same).
  * ``VllmConfig`` typing — duck-typed; my-vllm uses
    ``my_vllm.config.VllmConfig`` but accepts any object with the
    relevant attribute accessors.
  * All dataclasses + protocol constants (``GET_META_MSG``,
    ``PUSH_REG_NOTIF_PREFIX``, ``NIXL_CONNECTOR_VERSION``) —
    **kept verbatim** to maintain wire-format compatibility with
    vLLM at the metadata-exchange level.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from my_vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)

# Wire-format constants — match vLLM byte-for-byte.
GET_META_MSG = b"get_meta_msg"
PUSH_REG_NOTIF_PREFIX = b"PUSH_REG:"


# Bump when any of:
#   * NixlAgentMetadata schema changes
#   * kv_transfer_params schema / semantics changes
#   * NIXL transfer protocol or wire format changes
#   * KV cache memory layout or block organization changes
#   * Any other change that breaks P/D interoperability
NIXL_CONNECTOR_VERSION: int = 5


# Version of my-vllm — used by ``compute_nixl_compatibility_hash``.
# In production vLLM this is the vllm package version.  Hard-coded
# here because my-vllm doesn't have a package __version__.
_MY_VLLM_VERSION = "my-vllm-0.1"


# Type aliases (matches vLLM ``kv_connector.utils``).
TransferHandle = int
ReqId = str
EngineId = str
BlockIds = Any  # list[int] | np.ndarray — duck-typed.


@dataclass
class NixlAgentMetadata:
    """Per-engine metadata exchanged during NIXL handshake.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata
    .NixlAgentMetadata``.  Sent over the wire from producer → consumer
    during the out-of-band handshake.
    """

    engine_id: str
    agent_metadata: bytes  # NIXL agent descriptor (would be nixl serialized)
    kv_caches_base_addr: list[int]  # base pointers for each KV cache
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """Two-phase handshake payload.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata
    .NixlHandshakePayload``.  The two-phase decode (hash → payload)
    allows graceful failure when schema versions are incompatible.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded (msgpack)


def compute_nixl_compatibility_hash(
    vllm_config: Any,
    attn_backend_name: str,
    cross_layers_blocks: bool,
) -> str:
    """Compute the NIXL compatibility hash.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata
    .compute_nixl_compatibility_hash``.  Hashes only the factors that
    affect whether two NIXL instances can transfer KV cache data:

    * NIXL connector version
    * Model architecture (name, dtype, KV heads, layers)
    * KV cache format (dtype)
    * Attention backend
    * HMA / cross-layer-blocks enablement

    ``tensor_parallel_size``, ``block_size``, ``kv_cache_layout`` are
    **not** included — they are validated at runtime in
    ``_validate_remote_agent_handshake`` (heterogeneous deployments
    may have different TP sizes but still interoperate).
    """
    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    is_hma_enabled = (
        not scheduler_config.disable_hybrid_kv_cache_manager
        if scheduler_config is not None
        else True
    )

    factors = {
        "my_vllm_version": _MY_VLLM_VERSION,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        "model": getattr(model_config, "model", "unknown"),
        "dtype": str(getattr(model_config, "dtype", "unknown")),
        "num_kv_heads": getattr(
            model_config, "get_total_num_kv_heads", lambda: 0
        )(),
        "head_size": getattr(model_config, "get_head_size", lambda: 0)(),
        "num_hidden_layers": getattr(
            model_config, "get_total_num_hidden_layers", lambda: 0
        )(),
        "attn_backend_name": attn_backend_name,
        "cache_dtype": str(getattr(cache_config, "cache_dtype", "auto")),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
    }
    payload = json.dumps(factors, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class HeartbeatInfo:
    """Heartbeat data for a single remote engine (D worker → P worker)."""

    req_ids: set[str]
    host: str
    port: int
    tp_size: int
    pp_size: int = 1


@dataclass
class RemoteMeta:
    """Per-request metadata about the remote producer engine."""

    block_ids: Any  # BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str
    blocks_expiry_time: float | None = None


@dataclass
class ReqMeta:
    """Per-request KV transfer state.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata
    .ReqMeta``.
    """

    local_block_ids: Any  # BlockIds
    local_physical_block_ids: Any  # BlockIds
    tp_size: int
    remote: RemoteMeta | None = None
    remote_block_size: int | None = None
    pp_size: int = 1


class NixlConnectorMetadata(KVConnectorMetadata):
    """Per-step connector metadata exchanged scheduler → worker.

    Mirrors ``vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata
    .NixlConnectorMetadata``.  Holds:
      * ``reqs_to_save`` / ``reqs_to_recv`` — requests the worker
        needs to push / pull this step.
      * ``reqs_to_send`` — request IDs the producer should release
        after the next ``wait_for_save``.
      * ``reqs_in_batch`` / ``reqs_not_processed`` — bookkeeping
        for partial-batch scheduling.
      * ``heartbeat_by_engine`` — D-worker heartbeats sent to P.
      * ``push_registrations`` / ``push_finished_blocks`` — push-mode
        worker-to-worker registration notifications.
    """

    def __init__(self) -> None:
        self.reqs_to_recv: dict[str, ReqMeta] = {}
        self.reqs_to_save: dict[str, ReqMeta] = {}
        self.reqs_to_send: dict[str, float] = {}
        self.reqs_in_batch: set[str] = set()
        self.reqs_not_processed: set[str] = set()
        self.heartbeat_by_engine: dict[str, HeartbeatInfo] = {}
        # Push mode (D side): registrations to send to P via NIXL notif.
        self.push_registrations: dict[str, dict[str, Any]] = {}
        # Push mode (P side): newly finished blocks to match against
        # pending D registrations on the P worker.
        self.push_finished_blocks: dict[str, Any] = {}

    def _add_new_req(
        self,
        local_block_ids: Any,
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        """Build a ``ReqMeta`` from a ``kv_transfer_params`` dict."""
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            pp_size=kv_transfer_params.get("pp_size", 1),
        )

    def add_new_req_to_save(
        self,
        request_id: str,
        local_block_ids: Any,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        """Mark a request as save-side (prefill will save KV)."""
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params,
        )

    def add_new_req_to_recv(
        self,
        request_id: str,
        local_block_ids: Any,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        """Mark a request as recv-side (decode will pull KV)."""
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get(
                "remote_blocks_expiry_time",
            ),
        )
        self.reqs_to_recv[request_id] = req


__all__ = [
    "BlockIds",
    "EngineId",
    "GET_META_MSG",
    "HeartbeatInfo",
    "NIXL_CONNECTOR_VERSION",
    "NixlAgentMetadata",
    "NixlConnectorMetadata",
    "NixlHandshakePayload",
    "PUSH_REG_NOTIF_PREFIX",
    "RemoteMeta",
    "ReqId",
    "ReqMeta",
    "TransferHandle",
    "compute_nixl_compatibility_hash",
]

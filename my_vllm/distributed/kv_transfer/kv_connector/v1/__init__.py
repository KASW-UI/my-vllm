"""V1 KV Connector package — cross-process KV cache transfer.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/.

Exports the connector base class + metadata shapes + role enum + HMA
support + KV ownership / index / failure-tracker helpers used by
concrete backends (NixlConnector, …) and by the EngineCore's
`_kv_connector_init` (NoopConnector default).

C133: completed the migration — the formerly-empty
`vllm/distributed/kv_transfer/kv_connector/v1/base.py` now contains
the canonical types.
C251: filled in KV ownership / index / failure tracking + extended
the ``KVConnectorBase_V1`` surface to align with
vLLM's ``vllm/distributed/kv_transfer/kv_connector/v1/base.py:171-708``
(missing methods: ``bind_connector_metadata``,
``clear_connector_metadata``, ``register_kv_caches``,
``bind_gpu_block_pool``, ``handle_preemptions``,
``get_block_ids_with_load_errors``, ``get_finished``,
``get_finished_count``, ``get_handshake_metadata``,
``set_xfer_handshake_metadata``, ``build_connector_worker_meta``,
``on_new_request``, ``update_connector_output``, ``take_events``,
``has_pending_push_work``, ``shutdown``, ``reset_cache``,
``get_kv_connector_stats``, ``prefer_cross_layer_blocks``,
``get_required_kv_cache_layout``).
"""

from .base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    KVFailureTracker,
    KVIndex,
    KVIndexEntry,
    KVOwnership,
    KVOwnershipState,
    KVTransferFailure,
    KVTransferMethod,
    NoopConnector,
    RecvResult,
    SupportsHMA,
    TransferBundle,
    TransferStatus,
    supports_hma,
)

__all__ = [
    "CopyBlocksOp",
    "KVConnectorBase_V1",
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "KVConnectorWorkerMetadata",
    "KVFailureTracker",
    "KVIndex",
    "KVIndexEntry",
    "KVOwnership",
    "KVOwnershipState",
    "KVTransferFailure",
    "KVTransferMethod",
    "NoopConnector",
    "RecvResult",
    "SupportsHMA",
    "TransferBundle",
    "TransferStatus",
    "supports_hma",
]
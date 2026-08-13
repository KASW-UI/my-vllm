"""V1 KV Connector package — cross-process KV cache transfer.

Mirrors vLLM V1 distributed/kv_transfer/kv_connector/v1/.

Exports the connector base class + metadata shapes + role enum + HMA
support used by concrete backends (NixlConnector, …) and by the
EngineCore's `_kv_connector_init` (NoopConnector default).

C133: completed the migration — the formerly-empty
`vllm/distributed/kv_transfer/kv_connector/v1/base.py` now contains
the canonical types (KVConnectorBase_V1 + KVConnectorRole + 3
metadata ABCs + SupportsHMA + the per-transfer bundle / result
dataclasses that migrated from the old
`my_vllm/v1/core/kv_connector.py`).
"""

from .base import (
    CopyBlocksOp,
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
    NoopConnector,
    RecvResult,
    SupportsHMA,
    TransferBundle,
    TransferStatus,
    KVTransferMethod,
    supports_hma,
)

__all__ = [
    "CopyBlocksOp",
    "KVConnectorBase_V1",
    "KVConnectorHandshakeMetadata",
    "KVConnectorMetadata",
    "KVConnectorRole",
    "KVConnectorWorkerMetadata",
    "KVTransferMethod",
    "NoopConnector",
    "RecvResult",
    "SupportsHMA",
    "TransferBundle",
    "TransferStatus",
    "supports_hma",
]
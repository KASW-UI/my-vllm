"""NIXL (NVIDIA In-Xfer Library) KV connector.

C250: mirrors vLLM's 13-file ``kv_transfer/kv_connector/v1/nixl/``
package.  The package implements the full protocol layer (scheduler
/ worker split, push / pull variants, TP rank mapping, metadata
exchange, stats tracking) but uses the local shared-memory
``NixlBaseConnector`` backend for actual KV transfer.  In
production, NIXL RDMA replaces the local backend once the NIXL
library is vendored.

File map:

  * ``utils``        — ``get_base_request_id`` / ``get_representative_spec_type``.
  * ``tp_mapping``   — TP rank mapping for heterogeneous P/D TP.
  * ``metadata``     — wire-format dataclasses (``NixlAgentMetadata``,
                        ``NixlHandshakePayload``, ``NixlConnectorMetadata``,
                        ``HeartbeatInfo``, ``RemoteMeta``, ``ReqMeta``) +
                        ``compute_nixl_compatibility_hash``.
  * ``stats``        — ``NixlKVConnectorStats`` + ``TransferTelemetry``.
  * ``base_scheduler`` / ``base_worker`` — shared protocol surface.
  * ``pull_scheduler`` / ``pull_worker`` — pull-mode specifics.
  * ``push_scheduler`` / ``push_worker`` — push-mode specifics.
  * ``scheduler`` / ``worker``           — default facades
                                          (pull variants, backward compat).
  * ``connector``     — ``NixlBaseConnector`` + ``NixlPullConnector`` +
                        ``NixlPushConnector`` + ``NixlConnector`` alias.
"""

from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
    NixlBaseConnector,
    NixlConnector,
    NixlPullConnector,
    NixlPushConnector,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NIXL_CONNECTOR_VERSION,
    HeartbeatInfo,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    PUSH_REG_NOTIF_PREFIX,
    RemoteMeta,
    ReqMeta,
    compute_nixl_compatibility_hash,
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
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    TransferTelemetry,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    TPMapping,
    TransferTopology,
    compute_tp_mapping,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_base_request_id,
    get_representative_spec_type,
)
from my_vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)


__all__ = [
    # Connectors
    "NixlBaseConnector",
    "NixlConnector",
    "NixlPullConnector",
    "NixlPushConnector",
    # Default facades (backward compat)
    "NixlConnectorScheduler",
    "NixlConnectorWorker",
    # Scheduler / Worker
    "NixlBaseConnectorScheduler",
    "NixlBaseConnectorWorker",
    "NixlPullConnectorScheduler",
    "NixlPullConnectorWorker",
    "NixlPushConnectorScheduler",
    "NixlPushConnectorWorker",
    # Metadata
    "GET_META_MSG",
    "NIXL_CONNECTOR_VERSION",
    "HeartbeatInfo",
    "NixlAgentMetadata",
    "NixlConnectorMetadata",
    "NixlHandshakePayload",
    "PUSH_REG_NOTIF_PREFIX",
    "RemoteMeta",
    "ReqMeta",
    "compute_nixl_compatibility_hash",
    # Stats
    "NixlKVConnectorStats",
    "TransferTelemetry",
    # TP mapping
    "ReadSpec",
    "TPMapping",
    "TransferTopology",
    "compute_tp_mapping",
    # Utils
    "get_base_request_id",
    "get_representative_spec_type",
]

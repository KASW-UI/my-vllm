"""NIXL (NVIDIA In-Xfer Library) KV connector.

Mirrors vllm/distributed/kv_transfer/kv_connector/v1/nixl/.

C131: NixlConnector implementation now lives in `.connector`
(renamed from the formerly-invented `nixl_connector.py`).
"""

from ..base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.block_descriptors: dict[int, str] = {}


class NixlConnectorHandshakeMetadata(KVConnectorHandshakeMetadata):
    def __init__(self):
        self.agent_configs: dict = {}


class NixlConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    def __init__(self):
        self.loaded_blocks: set[int] = set()
        self.saved_blocks: set[int] = set()

    def aggregate(self, other: "NixlConnectorWorkerMetadata") -> "NixlConnectorWorkerMetadata":
        merged = NixlConnectorWorkerMetadata()
        merged.loaded_blocks = self.loaded_blocks | other.loaded_blocks
        merged.saved_blocks = self.saved_blocks | other.saved_blocks
        return merged


# Re-export the canonical NixlConnector implementation.
# Two classes previously existed (one in __init__.py, one in
# nixl_connector.py); C131 collapses them — the concrete implementation
# lives in .connector.
from .connector import NixlConnector  # noqa: E402

__all__ = [
    "NixlConnector",
    "NixlConnectorMetadata",
    "NixlConnectorHandshakeMetadata",
    "NixlConnectorWorkerMetadata",
]

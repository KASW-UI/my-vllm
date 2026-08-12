"""NIXL (NVIDIA In-Xfer Library) KV connector."""

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


class NixlConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role: KVConnectorRole, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self._nixl_agent = None
        self._pending_saves: int = 0

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name: str):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        self._pending_saves += 1

    def wait_for_save(self):
        self._pending_saves = 0

    def get_num_new_matched_tokens(self, request, block_pool) -> int:
        return 0

    def request_finished(self, request, block_ids):
        return False, None

    def build_connector_worker_meta(self):
        return NixlConnectorWorkerMetadata()

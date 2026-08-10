"""KV Connector — cross-process KV cache block transfer.

Aligns with vLLM KVConnectorBase_V1 (distributed/kv_transfer/kv_connector/v1/base.py).
C102: start_load_kv / save_kv_layer / wait_for_save — PD disaggregation foundation.

Use case (Prefill/Decode disaggregation):
    Prefill Worker: prompt → prefill → save_kv_layer()
        ↓
    KV Connector:  shared memory / NCCL / RDMA
        ↓
    Decode Worker:  start_load_kv() → loaded blocks → decode
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import torch


class KVTransferMethod(Enum):
    SHARED_MEMORY = "shared_memory"
    NCCL = "nccl"
    RDMA = "rdma"


class KVConnectorBase_V1(ABC):
    """Cross-process/network KV block transfer base.

    Aligns with vLLM KVConnectorBase_V1.
    Worker-side API: start_load_kv / save_kv_layer / wait_for_save.
    Scheduler-side API: get_num_new_matched_tokens / build_connector_meta.
    """

    @abstractmethod
    def start_load_kv(self, forward_context: Any = None, **kwargs: Any) -> None:
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

    # --- Scheduler-side API ---

    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> int:
        """Calculate how many tokens can be sourced from external cache."""
        return 0

    def update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ) -> None:
        """Update connector state after block allocation."""
        pass

    def build_connector_meta(self, scheduler_output) -> Any:
        """Build connector metadata for the Worker."""
        return None

    def request_finished(self, request, block_ids) -> None:
        """Cleanup when a request finishes."""
        pass


class NoopConnector(KVConnectorBase_V1):
    """No-op connector: all operations are identity."""

    def start_load_kv(self, forward_context: Any = None, **kwargs: Any) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor,
        attn_metadata: Any = None, **kwargs: Any
    ) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

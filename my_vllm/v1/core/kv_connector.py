"""KV Connector — cross-process KV cache block transfer.

Aligns with vLLM KVConnectorBase_V1 (distributed/kv_transfer/kv_connector/v1/base.py).
C102: start_load_kv / save_kv_layer / wait_for_save — PD disaggregation foundation.
C112: failure handling — TransferStatus, RecvResult, timeout + recompute/fail policy.

Use case (Prefill/Decode disaggregation):
    Prefill Worker: prompt → prefill → save_kv_layer()
        ↓
    KV Connector:  shared memory / NCCL / RDMA
        ↓
    Decode Worker:  start_load_kv() → loaded blocks → decode

Failure modes:
    TIMEOUT: transfer did not complete within transfer_timeout.
    CORRUPTION: block hash mismatch between producer and consumer.
    Policy: "recompute" → retry transfer for failed blocks.
            "fail" → mark request as FINISHED_ERROR.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch


class KVTransferMethod(Enum):
    SHARED_MEMORY = "shared_memory"
    NCCL = "nccl"
    RDMA = "rdma"


class TransferStatus(Enum):
    """Result of a KV block transfer check."""
    SUCCESS = auto()
    TIMEOUT = auto()
    CORRUPTION = auto()


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
    """Result of check_transfer_done() on the consumer side.

    bundles: successfully received requests ready for decode.
    invalid_req_ids: requests whose transfer failed (timeout or corruption).
    valid_prefix_tokens: for failed requests, how many tokens were successfully
        transferred before the failure (used by recompute policy to resume).
    """

    status: TransferStatus = TransferStatus.SUCCESS
    bundles: list[TransferBundle] = field(default_factory=list)
    invalid_req_ids: set[str] = field(default_factory=set)
    valid_prefix_tokens: dict[str, int] = field(default_factory=dict)


class KVConnectorBase_V1(ABC):
    """Cross-process/network KV block transfer base.

    Aligns with vLLM KVConnectorBase_V1.
    Worker-side API: start_load_kv / save_kv_layer / wait_for_save.
    Scheduler-side API: get_num_new_matched_tokens / build_connector_meta.
    Failure API: check_transfer_done / send_ack / drain_acks.
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

    # --- P/D failure handling API ---

    def send_ack(self, request_id: str, success: bool) -> None:
        """Consumer → Producer: acknowledge transfer result.

        Producer uses this to determine when to free deferred blocks.
        """

    def check_transfer_done(self, timeout: float) -> RecvResult:
        """Consumer: poll for completed transfers with timeout detection.

        Returns RecvResult with successful bundles and failed request IDs.
        Subclasses override this to implement actual polling logic.
        """
        return RecvResult(status=TransferStatus.SUCCESS)

    def drain_acks(self) -> dict[str, bool]:
        """Producer: collect ACKs from the consumer.

        Returns {request_id: success} for all pending transfers that
        have received an ACK (success or failure) from the consumer.
        """
        return {}

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

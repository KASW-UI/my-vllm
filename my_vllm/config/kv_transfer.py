"""KVTransferConfig — P/D disaggregation configuration.

Aligns with vLLM vllm/config/kv_transfer.py (simplified).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class KVTransferConfig:
    """Prefill/Decode disaggregation configuration.

    kv_role=None: standalone mode (no P/D separation).
    kv_role="kv_producer": this instance does prefill, sends KV to consumer.
    kv_role="kv_consumer": this instance does decode, receives KV from producer.
    """

    kv_role: Literal["kv_producer", "kv_consumer"] | None = None
    connector_type: str = "shared_memory"
    kv_load_failure_policy: Literal["recompute", "fail"] = "recompute"
    transfer_timeout: float = 30.0

    @property
    def is_kv_producer(self) -> bool:
        return self.kv_role == "kv_producer"

    @property
    def is_kv_consumer(self) -> bool:
        return self.kv_role == "kv_consumer"

    @property
    def is_kv_transfer_instance(self) -> bool:
        return self.kv_role is not None

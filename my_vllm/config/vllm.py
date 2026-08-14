"""VllmConfig — aggregates all sub-configs. Aligns with vLLM vllm/config/vllm.py."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from my_vllm.config.cache import CacheConfig
from my_vllm.config.lora import LoRAConfig
from my_vllm.config.model import ModelConfig
from my_vllm.config.parallel import ParallelConfig
from my_vllm.config.quantization import QuantizationConfig
from my_vllm.config.scheduler import SchedulerConfig
from my_vllm.config.speculative import SpeculativeConfig

if TYPE_CHECKING:
    from my_vllm.config.kv_transfer import KVTransferConfig


@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    quantization_config: QuantizationConfig = field(
        default_factory=QuantizationConfig
    )
    lora_config: LoRAConfig | None = None
    kv_transfer_config: "KVTransferConfig | None" = None
    speculative_config: SpeculativeConfig | None = None
    """Speculative decoding configuration.  ``None`` (the default)
    disables spec decoding.  Mirrors vLLM ``VllmConfig.
    speculative_config`` — set to ``SpeculativeConfig(method="ngram",
    num_speculative_tokens=K, ...)`` to enable the n-gram proposer
    (see ``my_vllm.v1.spec_decode.ngram_proposer``)."""

    # C257: KV-cache event publisher config (placeholder). Mirrors
    # ``VllmConfig.kv_events_config``. The SimpleCPUOffloadScheduler
    # reads ``enable_kv_cache_events`` to forward to the CPU coordinator.
    kv_events_config: Any | None = None

    # C257: max in-flight token budget (mirrors ``VllmConfig.
    # max_in_flight_tokens``). Used by ``get_kv_cache_coordinator`` as
    # the long-prefill cap.
    max_in_flight_tokens: int | None = None


EngineConfig = VllmConfig

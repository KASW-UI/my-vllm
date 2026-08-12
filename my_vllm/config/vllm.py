"""VllmConfig — aggregates all sub-configs. Aligns with vLLM vllm/config/vllm.py."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from my_vllm.config.cache import CacheConfig
from my_vllm.config.lora import LoRAConfig
from my_vllm.config.model import ModelConfig
from my_vllm.config.multi_tenant import MultiTenantConfig
from my_vllm.config.parallel import ParallelConfig
from my_vllm.config.quantization import QuantizationConfig
from my_vllm.config.scheduler import SchedulerConfig

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
    multi_tenant_config: MultiTenantConfig = field(
        default_factory=MultiTenantConfig
    )
    kv_transfer_config: "KVTransferConfig | None" = None


EngineConfig = VllmConfig

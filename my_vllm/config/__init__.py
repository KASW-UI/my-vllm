"""Config package — VllmConfig + sub-configs.

Aligns with vLLM vllm/config/ directory structure.
C069: decomposes monolithic EngineConfig into ModelConfig / CacheConfig / SchedulerConfig.
C112: KVTransferConfig for P/D disaggregation.
C113: QuantizationConfig + MultiTenantConfig for heterogeneous + multi-tenant inference.
"""

from my_vllm.config.cache import CacheConfig
from my_vllm.config.kv_transfer import KVTransferConfig
from my_vllm.config.lora import LoRAConfig
from my_vllm.config.model import ModelConfig
from my_vllm.config.multi_tenant import MultiTenantConfig, TenantConfig
from my_vllm.config.quantization import QuantizationConfig, QuantizationMethod
from my_vllm.config.scheduler import SchedulerConfig
from my_vllm.config.vllm import EngineConfig, VllmConfig

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "KVTransferConfig",
    "LoRAConfig",
    "ModelConfig",
    "MultiTenantConfig",
    "QuantizationConfig",
    "QuantizationMethod",
    "SchedulerConfig",
    "TenantConfig",
    "VllmConfig",
]

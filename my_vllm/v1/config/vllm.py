"""VllmConfig — aggregates all sub-configs. Aligns with vLLM vllm/config/vllm.py."""

from dataclasses import dataclass, field

from my_vllm.v1.config.cache import CacheConfig
from my_vllm.v1.config.model import ModelConfig
from my_vllm.v1.config.scheduler import SchedulerConfig


@dataclass
class EngineConfig:
    """Backward-compat alias for VllmConfig (deprecated, use VllmConfig)."""

    model_name: str = "opt-125m"
    dtype: str = "bfloat16"
    block_size: int = 16
    num_kv_blocks: int = 512
    enable_prefix_caching: bool = True
    max_running: int = 8
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096
    max_new_tokens: int = 50
    eos_token_id: int = 2
    enforce_eager: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90


@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    tensor_parallel_size: int = 1

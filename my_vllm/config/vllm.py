"""VllmConfig — aggregates all sub-configs. Aligns with vLLM vllm/config/vllm.py."""

from dataclasses import dataclass, field

from my_vllm.config.cache import CacheConfig
from my_vllm.config.model import ModelConfig
from my_vllm.config.parallel import ParallelConfig
from my_vllm.config.scheduler import SchedulerConfig


@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)


EngineConfig = VllmConfig

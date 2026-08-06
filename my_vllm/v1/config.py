"""EngineConfig — runtime configuration.

Mirrors vLLM V1 config.py (simplified).

C031: centralized configuration for model, cache, and scheduler
parameters.
"""

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Unified engine configuration.

    Collects parameters previously scattered across EngineCore,
    BlockPool, Scheduler, and GPUModelRunner constructors.
    """

    model_name: str = "opt-125m"
    dtype: str = "bfloat16"

    # KV cache
    block_size: int = 16
    num_kv_blocks: int = 512
    enable_prefix_caching: bool = True

    # Scheduler
    max_running: int = 8
    max_num_batched_tokens: int = 2048
    max_model_len: int = 4096

    # Generation
    max_new_tokens: int = 50
    eos_token_id: int = 2

    # Execution
    enforce_eager: bool = True
    tensor_parallel_size: int = 1

    # GPU
    gpu_memory_utilization: float = 0.90

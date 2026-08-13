"""ModelConfig — aligns with vLLM vllm/config/model.py (simplified)."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "facebook/opt-125m"
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    seed: int = 0
    enforce_eager: bool = True
    """When True, skip CUDA graph capture and run eagerly. Mirrors
    vLLM vllm/config/model.py:ModelConfig.enforce_eager."""

"""LoRAConfig — aligns with vllm/config/lora.py:32."""

from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """Configuration for LoRA.

    Mirrors vllm/config/lora.py:32 — max_loras controls how many distinct
    LoRA adapters can be active in a single batch. The scheduler enforces
    this at schedule() time.
    """

    max_lora_rank: int = 16
    max_loras: int = 1
    """Max number of distinct LoRA adapters in a single batch."""
    max_cpu_loras: int | None = None
    lora_dtype: str = "auto"

    def __post_init__(self) -> None:
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras

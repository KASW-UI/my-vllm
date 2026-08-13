"""LoRAConfig — aligns with vllm/config/lora.py:32."""

from dataclasses import dataclass, field
from typing import Literal

MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
LoRADType = Literal["auto", "float16", "bfloat16"]


@dataclass
class LoRAConfig:
    """Configuration for LoRA.

    Mirrors vLLM V1 config/lora.py:LoRAConfig (subset — pydantic →
    dataclass, simplified field set).

    C145: extended with `fully_sharded_loras` /
    `target_modules` to match vLLM's canonical field set
    (needed by the LoRA model loader to know which layers to
    wrap). The full vLLM config has `loraplus_*-` /
    `bias_enabled` / `use_routing` / `alpha` config knobs —
    out of scope for my-vllm's minimal subset.
    """

    max_lora_rank: int = 16
    """Max LoRA rank."""
    max_loras: int = 1
    """Max number of LoRAs in a single batch."""
    fully_sharded_loras: bool = False
    """If True, fully shard LoRA computation across TP (vs the
    default half-sharded)."""
    max_cpu_loras: int | None = None
    """Maximum number of LoRAs to store in CPU memory. Must be
    >= `max_loras`."""
    lora_dtype: str = "auto"
    """Data type for LoRA. If auto, default to base model dtype."""
    target_modules: list[str] | None = field(default_factory=list)
    """Restrict LoRA to specific module suffixes (e.g.
    ["q_proj", "v_proj"]). None or [] = all linear layers."""

    def __post_init__(self) -> None:
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras

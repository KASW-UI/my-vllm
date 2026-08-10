"""SchedulerConfig — aligns with vLLM vllm/config/scheduler.py (simplified)."""

from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    enable_chunked_prefill: bool = True
    policy: str = "fcfs"

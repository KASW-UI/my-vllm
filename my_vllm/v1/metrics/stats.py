"""Metrics data structures — aligns with vLLM V1 v1/metrics/stats.py.

C048: SchedulerStats, IterationStats, and supporting types for LLM serving metrics.
"""

import time
from dataclasses import dataclass, field


@dataclass
class KVCacheEvictionEvent:
    """Single KV cache block eviction sample — aligns with vLLM KVCacheEvictionEvent."""

    lifetime_seconds: float
    idle_seconds: float
    reuse_gaps_seconds: tuple[float, ...]


@dataclass
class SchedulerStats:
    """Aggregated scheduler state — aligns with vLLM SchedulerStats."""

    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    kv_cache_usage: float = 0.0  # 0.0 .. 1.0
    prefix_cache_stats: "PrefixCacheStats | None" = None
    kv_cache_eviction_events: list[KVCacheEvictionEvent] = field(
        default_factory=list
    )

    def __repr__(self) -> str:
        return (
            f"SchedulerStats(running={self.num_running_reqs}, "
            f"waiting={self.num_waiting_reqs}, kv_usage={self.kv_cache_usage:.1%}"
            f")"
        )


@dataclass
class PrefixCacheStats:
    """Prefix cache hit/miss statistics."""

    queries: int = 0
    hits: int = 0

    @property
    def hit_rate(self) -> float:
        if self.queries == 0:
            return 0.0
        return self.hits / self.queries

    def reset(self) -> None:
        self.queries = 0
        self.hits = 0


@dataclass
class RequestStateStats:
    """Per-request timing deltas."""

    request_id: str = ""
    arrival_time: float = 0.0
    first_token_ts: float | None = None
    last_token_ts: float | None = None
    num_generation_tokens: int = 0


@dataclass
class FinishedRequestStats:
    """Stats emitted when a request finishes."""

    request_id: str
    num_prompt_tokens: int
    num_generation_tokens: int
    e2e_latency: float  # seconds
    ttft: float | None = None  # seconds
    tpot: float = 0.0  # seconds per output token
    itl: float = 0.0  # seconds between consecutive tokens


class IterationStats:
    """Per-iteration latency/throughput stats — aligns with vLLM IterationStats."""

    def __init__(self) -> None:
        self.iteration_timestamp = time.time()
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0
        self.num_preempted_reqs: int = 0
        self.finished_requests: list[FinishedRequestStats] = []
        self.time_to_first_tokens_iter: list[float] = []
        self.inter_token_latencies_iter: list[float] = []
        self._total_prompt_time_s: float = 0.0
        self._total_decode_time_s: float = 0.0

    def _time_since(self, start: float) -> float:
        return self.iteration_timestamp - start

    def record_first_token(self, latency_s: float) -> None:
        self.time_to_first_tokens_iter.append(latency_s)

    def record_inter_token(self, latency_s: float) -> None:
        self.inter_token_latencies_iter.append(latency_s)

    def update_from_output(
        self,
        num_prompt_tokens: int,
        num_generation_tokens: int,
        finished: list[FinishedRequestStats],
        num_preempted: int = 0,
    ) -> None:
        self.num_prompt_tokens = num_prompt_tokens
        self.num_generation_tokens = num_generation_tokens
        self.num_preempted_reqs = num_preempted
        self.finished_requests.extend(finished)

    @property
    def ttft_ms(self) -> float | None:
        if not self.time_to_first_tokens_iter:
            return None
        return sum(self.time_to_first_tokens_iter) / len(self.time_to_first_tokens_iter) * 1000

    @property
    def tpot_ms(self) -> float:
        if self.num_generation_tokens == 0:
            return 0.0
        return self._total_decode_time_s / self.num_generation_tokens * 1000

    @property
    def itl_ms(self) -> float:
        if not self.inter_token_latencies_iter:
            return 0.0
        return sum(self.inter_token_latencies_iter) / len(self.inter_token_latencies_iter) * 1000

    @property
    def prompt_throughput(self) -> float:
        if self._total_prompt_time_s == 0:
            return 0.0
        return self.num_prompt_tokens / self._total_prompt_time_s

    @property
    def generation_throughput(self) -> float:
        if self._total_decode_time_s == 0:
            return 0.0
        return self.num_generation_tokens / self._total_decode_time_s

    def set_prompt_time(self, seconds: float) -> None:
        self._total_prompt_time_s = seconds

    def set_decode_time(self, seconds: float) -> None:
        self._total_decode_time_s = seconds

    def __repr__(self) -> str:
        parts = [
            f"n_prompt={self.num_prompt_tokens}",
            f"n_gen={self.num_generation_tokens}",
        ]
        if self.ttft_ms is not None:
            parts.append(f"ttft={self.ttft_ms:.1f}ms")
        if self.tpot_ms > 0:
            parts.append(f"tpot={self.tpot_ms:.1f}ms")
        if self.itl_ms > 0:
            parts.append(f"itl={self.itl_ms:.1f}ms")
        return f"IterationStats({', '.join(parts)})"

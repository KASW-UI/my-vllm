"""Basic metrics collector for the serving runtime."""

import time
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    num_requests: int = 0
    num_finished: int = 0
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_time_ms: float = 0.0
    _request_start: dict[str, float] = field(default_factory=dict)

    def on_request_start(self, request_id: str):
        self.num_requests += 1
        self._request_start[request_id] = time.time()

    def on_request_finish(
        self,
        request_id: str,
        prompt_tokens: int,
        output_tokens: int,
    ):
        self.num_finished += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens
        start = self._request_start.pop(request_id, None)
        if start is not None:
            self.total_time_ms += (time.time() - start) * 1000

    @property
    def avg_latency_ms(self) -> float:
        if self.num_finished == 0:
            return 0.0
        return self.total_time_ms / self.num_finished

    @property
    def throughput_tok_per_s(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.total_output_tokens / (self.total_time_ms / 1000.0)

    def report(self) -> dict:
        return {
            "num_requests": self.num_requests,
            "num_finished": self.num_finished,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "throughput_tok_per_s": round(self.throughput_tok_per_s, 1),
        }

"""Spec-decode metrics — per-step iteration stats + logging.

Mirrors vLLM V1 v1/spec_decode/metrics.py.

C138: minimal subset — `SpecDecodingStats` dataclass and
`SpecDecodingLogging` aggregator. The full vLLM metrics module
also has Prometheus integration (`prometheus_client` based
counters / gauges) — out of scope for my-vllm (we keep the
metrics path in-process; per-engine Prometheus exposure is a
production-side concern).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler.

    Mirrors vLLM V1 v1/spec_decode/metrics.py:SpecDecodingStats (L17).
    Each scheduler step, statistics on spec decoding performance are
    aggregated across requests by the scheduler and returned to the
    frontend in `EngineCoreOutputs -> SchedulerStats`.
    """

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)
    num_draft_tokens_per_pos: list[int] = field(default_factory=list)

    @classmethod
    def new(cls, num_spec_tokens: int) -> "SpecDecodingStats":
        return cls(
            num_spec_tokens=num_spec_tokens,
            num_accepted_tokens_per_pos=[0] * num_spec_tokens,
            num_draft_tokens_per_pos=[0] * num_spec_tokens,
        )

    def observe_draft(
        self,
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> None:
        """Record the result of one draft round."""
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        assert num_accepted_tokens <= self.num_spec_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
        for i in range(num_draft_tokens):
            self.num_draft_tokens_per_pos[i] += 1

    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens that were accepted.

        Returns 0.0 when no drafts have been observed yet.
        """
        if self.num_draft_tokens == 0:
            return 0.0
        return self.num_accepted_tokens / self.num_draft_tokens


class SpecDecodingLogging:
    """Aggregate and log spec-decoding metrics over a time interval.

    Mirrors vLLM V1 v1/spec_decode/metrics.py:SpecDecodingLogging (L52).
    `LoggingStatLogger` aggregates per-iteration metrics over a set
    time interval using `observe()` and then logs them using `log()`
    before resetting to zero.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.num_drafts: list[int] = []
        self.num_draft_tokens: list[int] = []
        self.num_accepted_tokens: list[int] = []
        self.accepted_tokens_per_pos_lists: list[list[int]] = []
        self.last_log_time: float = time.monotonic()

    def observe(self, spec_decoding_stats: SpecDecodingStats) -> None:
        self.num_drafts.append(spec_decoding_stats.num_drafts)
        self.num_draft_tokens.append(spec_decoding_stats.num_draft_tokens)
        self.num_accepted_tokens.append(spec_decoding_stats.num_accepted_tokens)
        self.accepted_tokens_per_pos_lists.append(
            spec_decoding_stats.num_accepted_tokens_per_pos
        )

    def log(self) -> None:
        """Emit aggregated stats as a single log line, then reset."""
        if not self.num_drafts:
            return
        total_drafts = sum(self.num_drafts)
        total_draft_tokens = sum(self.num_draft_tokens)
        total_accepted = sum(self.num_accepted_tokens)
        rate = total_accepted / max(total_draft_tokens, 1)
        logger.info(
            "spec_decode: drafts=%d draft_tokens=%d accepted=%d "
            "acceptance_rate=%.3f",
            total_drafts,
            total_draft_tokens,
            total_accepted,
            rate,
        )
        self.reset()


__all__ = [
    "SpecDecodingLogging",
    "SpecDecodingStats",
]
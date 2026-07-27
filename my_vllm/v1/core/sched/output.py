"""Scheduler output data structures — mirrors vLLM V1 v1/core/sched/output.py.

C015: placeholder definitions. Internal scheduler still returns list[Request].
These structs will be adopted by the Scheduler in C016.
"""

from dataclasses import dataclass, field


@dataclass
class NewRequestData:
    """Data for a request scheduled for the first time."""

    req_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


@dataclass
class CachedRequestData:
    """Diff data for requests already cached in the Worker."""

    req_ids: list[str] = field(default_factory=list)
    new_block_ids: dict[str, list[int]] = field(default_factory=dict)
    num_computed_tokens: dict[str, int] = field(default_factory=dict)
    num_output_tokens: dict[str, int] = field(default_factory=dict)

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)


@dataclass
class SchedulerOutput:
    """Structured output from Scheduler.schedule().

    C015 placeholder — Scheduler still returns list[Request].
    This struct will become the return type in C016.
    """

    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    scheduled_cached_reqs: CachedRequestData = field(
        default_factory=CachedRequestData
    )
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    finished_req_ids: set[str] = field(default_factory=set)
    new_block_ids_to_zero: list[int] | None = None
    preempted_req_ids: set[str] | None = None

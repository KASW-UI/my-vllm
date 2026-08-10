"""Scheduler output data structures — mirrors vLLM V1 v1/core/sched/output.py.

C016: enhanced from C015 placeholder to full definition.
These structs are now the return type of Scheduler.schedule().
C038: num_common_prefix_blocks tracks per-request shared prefix cache hits.
"""

from dataclasses import dataclass, field

from my_vllm.v1.core.kv_cache_utils import KVCacheBlockCopy


@dataclass
class NewRequestData:
    """Data for a request scheduled for the first time.

    Mirrors vLLM V1 v1/core/sched/output.py:35.
    """

    req_id: str
    prompt_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0


@dataclass
class CachedRequestData:
    """Diff data for requests already cached in the Worker.

    Mirrors vLLM V1 v1/core/sched/output.py:116 — uses index-aligned
    parallel lists instead of dicts, matching real vLLM's IPC-optimized layout.
    """

    req_ids: list[str] = field(default_factory=list)
    new_block_ids: list[list[int] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)

    @property
    def num_reqs(self) -> int:
        return len(self.req_ids)

    @property
    def req_id_to_index(self) -> dict[str, int]:
        return {rid: i for i, rid in enumerate(self.req_ids)}

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(
            req_ids=[],
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        )


@dataclass
class SchedulerOutput:
    """Structured output from Scheduler.schedule().

    Mirrors vLLM V1 v1/core/sched/output.py:193.

    C016: Scheduler.schedule() now returns this struct instead of
    list[Request].  EngineCore uses scheduled_new_reqs to identify
    which requests need prefill and scheduled_cached_reqs for decode.
    C038: num_common_prefix_blocks reports per-request prefix cache hit count.
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
    num_common_prefix_blocks: list[int] = field(default_factory=list)
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    @classmethod
    def make_empty(cls) -> "SchedulerOutput":
        return cls(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            finished_req_ids=set(),
        )

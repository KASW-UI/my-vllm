"""V1 KV connector utility helpers.

Mirrors vllm/distributed/kv_transfer/kv_connector/v1/utils.py.

C257: ``yield_req_data`` extracted from vLLM for the SimpleCPUOffload
eager-mode scheduler path. Walks ``SchedulerOutput.scheduled_new_reqs``
and ``scheduled_cached_reqs`` and yields per-request
``(req_id, new_block_ids_per_group, preempted)`` tuples so the
connector can accumulate block IDs and detect preemptions.
"""

from collections.abc import Iterator

from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)


def yield_req_data(
    scheduler_output: SchedulerOutput,
) -> Iterator[tuple[str, tuple[list[int] | None, ...] | None, bool]]:
    """Yield ``(req_id, new_block_ids, preempted)`` for each scheduled request.

    Mirrors ``vllm/distributed/kv_transfer/kv_connector/v1/utils.py:yield_req_data``.

    For ``NewRequestData`` the block IDs are the per-group initial
    block IDs; for ``CachedRequestData`` they are the per-group newly
    appended block IDs (or ``None`` when the request didn't extend a
    given group this step).

    ``preempted`` is True for cached requests whose ``new_block_ids``
    was wiped due to a preemption event.
    """
    for req in scheduler_output.scheduled_new_reqs:
        assert isinstance(req, NewRequestData)
        # New requests carry a single per-group list of block IDs.
        yield (req.req_id, (list(req.block_ids),), False)

    cached: CachedRequestData = scheduler_output.scheduled_cached_reqs
    for idx, req_id in enumerate(cached.req_ids):
        preempted = cached.new_block_ids[idx] is None
        # cached.new_block_ids[idx] is `list[list[int] | None]` aligned to
        # groups, mirroring vLLM's CachedRequestData.
        per_group = cached.new_block_ids[idx]
        yield (req_id, per_group, preempted)

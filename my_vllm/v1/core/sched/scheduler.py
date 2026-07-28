"""Scheduler component — mirrors vLLM V1 v1/core/sched/scheduler.py.

C016: migrated from scheduler/scheduler.py.
Core change: schedule() now returns SchedulerOutput instead of list[Request].
FCFSRequestQueue replaces plain list for waiting queue.
C029: prefix cache lookup via KVCacheManager.get_computed_blocks().
"""

from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.core.sched.request_queue import FCFSRequestQueue
from my_vllm.v1.request import Request, RequestStatus


class Scheduler:
    """FCFS scheduler with max batch size and KV cache awareness.

    C029: checks prefix cache before allocating new blocks.
    """

    def __init__(
        self,
        max_running: int = 8,
        kv_cache_manager: KVCacheManager | None = None,
    ):
        self.waiting = FCFSRequestQueue()
        self.running: list[Request] = []
        self.max_running = max_running
        self.kv_cache_manager = kv_cache_manager

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)

    def schedule(self) -> SchedulerOutput:
        new_requests: list[Request] = []
        new_blocks: dict[str, list[int]] = {}
        num_cached: dict[str, int] = {}

        while len(self.running) < self.max_running and self.waiting:
            req = self.waiting.pop_request()

            if self.kv_cache_manager is not None:
                # C029: check prefix cache first
                cached_blocks, num_computed = (
                    self.kv_cache_manager.get_computed_blocks(req)
                )
                req.num_computed_tokens = num_computed

                remaining = len(req.prompt_token_ids) - num_computed
                if remaining > 0:
                    blocks = self.kv_cache_manager.allocate_slots(
                        req, len(req.prompt_token_ids)
                    )
                else:
                    blocks = cached_blocks

                if blocks is None or (remaining > 0 and len(blocks) == 0):
                    self.waiting.prepend_request(req)
                    break
                new_blocks[req.request_id] = list(blocks.block_ids)
                num_cached[req.request_id] = num_computed

            req.status = RequestStatus.RUNNING
            self.running.append(req)
            new_requests.append(req)

        new_req_data = [
            NewRequestData(
                req_id=r.request_id,
                prompt_token_ids=r.prompt_token_ids,
                block_ids=new_blocks.get(r.request_id, []),
                num_computed_tokens=num_cached.get(r.request_id, 0),
            )
            for r in new_requests
        ]

        cached_data = CachedRequestData(
            req_ids=[r.request_id for r in self.running],
            num_computed_tokens={
                r.request_id: r.num_computed_tokens
                for r in self.running
            },
            num_output_tokens={
                r.request_id: r.num_output_tokens
                for r in self.running
            },
        )

        total_tokens = sum(len(r.prompt_token_ids) for r in new_requests)
        total_tokens += len(self.running)

        return SchedulerOutput(
            scheduled_new_reqs=new_req_data,
            scheduled_cached_reqs=cached_data,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=total_tokens,
            finished_req_ids=set(),
        )

    def finish(self, request: Request):
        if request in self.running:
            # C029: cache the request's blocks for prefix reuse
            if self.kv_cache_manager is not None:
                blocks = self.kv_cache_manager.get_blocks(
                    request.request_id
                )
                if blocks:
                    self.kv_cache_manager.block_pool.cache_full_blocks(
                        request, blocks
                    )
            self.running.remove(request)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

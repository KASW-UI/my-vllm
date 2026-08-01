"""Scheduler component — mirrors vLLM V1 v1/core/sched/scheduler.py.

C016: migrated from scheduler/scheduler.py.
C029: prefix cache lookup via KVCacheManager.get_computed_blocks().
C030: chunked prefill — long prompts split by token_budget;
      num_computed_tokens tracks prefill progress per-chunk.
"""

from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.core.sched.request_queue import FCFSRequestQueue
from my_vllm.v1.outputs import EngineCoreOutputs, ModelRunnerOutput
from my_vllm.v1.request import Request, RequestStatus


class Scheduler:
    """FCFS scheduler with chunked prefill (C030).

    max_num_batched_tokens limits per-step prefill tokens.
    Long prompts are split across multiple steps; after each
    chunk the request remains running for decode interleaving.
    """

    def __init__(
        self,
        max_running: int = 8,
        kv_cache_manager: KVCacheManager | None = None,
        max_num_batched_tokens: int = 2048,
    ):
        self.waiting = FCFSRequestQueue()
        self.running: list[Request] = []
        self.max_running = max_running
        self.kv_cache_manager = kv_cache_manager
        self.max_num_batched_tokens = max_num_batched_tokens

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)

    def schedule(self) -> SchedulerOutput:
        token_budget = self.max_num_batched_tokens
        new_req_datas: list[NewRequestData] = []
        new_blocks: dict[str, list[int]] = {}

        # C030: handle running requests that still need prefill (chunk continue)
        for req in self.running:
            if not req.is_prefill_chunk:
                continue
            remaining = len(req.prompt_token_ids) - req.num_computed_tokens
            if remaining <= 0 or token_budget <= 0:
                req.is_prefill_chunk = False
                continue

            chunk = min(remaining, token_budget)
            token_budget -= chunk
            req.num_computed_tokens += chunk
            req.is_prefill_chunk = (
                req.num_computed_tokens < len(req.prompt_token_ids)
            )

            # Allocate additional blocks if needed for the new tokens
            if self.kv_cache_manager is not None:
                blocks = self.kv_cache_manager.allocate_slots(
                    req, req.num_computed_tokens
                )
                if blocks is None:
                    req.num_computed_tokens -= chunk
                    continue
                new_blocks[req.request_id] = list(blocks.block_ids)

            new_req_datas.append(NewRequestData(
                req_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                block_ids=new_blocks.get(req.request_id, []),
                num_computed_tokens=req.num_computed_tokens - chunk,
            ))

        # Pull new requests from waiting queue
        while len(self.running) < self.max_running and self.waiting and token_budget > 0:
            req = self.waiting.pop_request()
            prompt_len = len(req.prompt_token_ids)
            num_cached = 0

            if self.kv_cache_manager is not None:
                cached_blocks, num_cached = (
                    self.kv_cache_manager.get_computed_blocks(req)
                )
                req.num_computed_tokens = num_cached

                blocks = self.kv_cache_manager.allocate_slots(
                    req, prompt_len
                )
                if blocks is None:
                    self.waiting.prepend_request(req)
                    break
                new_blocks[req.request_id] = list(blocks.block_ids)

            # C030: chunk if remaining > token_budget
            remaining = prompt_len - num_cached
            chunk = min(remaining, token_budget)
            token_budget -= chunk
            req.num_computed_tokens = num_cached + chunk
            req.is_prefill_chunk = (
                req.num_computed_tokens < prompt_len
            )

            req.status = RequestStatus.RUNNING
            self.running.append(req)

            new_req_datas.append(NewRequestData(
                req_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                block_ids=new_blocks.get(req.request_id, []),
                num_computed_tokens=num_cached,
            ))

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

        total_tokens = sum(
            len(d.prompt_token_ids) for d in new_req_datas
        )
        total_tokens += len(self.running)

        return SchedulerOutput(
            scheduled_new_reqs=new_req_datas,
            scheduled_cached_reqs=cached_data,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=total_tokens,
            finished_req_ids=set(),
        )

    def finish(self, request: Request):
        if request in self.running:
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

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
        eos_token_id: int,
        max_tokens: int,
    ) -> EngineCoreOutputs:
        """Detect finished requests from model output.

        C032: EngineCore delegates finish detection to the scheduler
        instead of iterating requests inline.  OutputProcessor still
        handles token appending and detokenization for correct ordering.
        """
        finished: list[Request] = []
        for rid, token_ids in model_output.sampled_token_ids.items():
            req = self._find_request(rid)
            if req is None:
                continue
            for tok in token_ids:
                if tok == eos_token_id:
                    req.status = RequestStatus.FINISHED_STOPPED
                    break
            limit = req.max_tokens or max_tokens
            if req.num_output_tokens + len(token_ids) >= limit and not req.is_finished():
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
            if req.is_finished():
                finished.append(req)
        return EngineCoreOutputs(
            finished_requests=finished,
            sampled_token_ids=model_output.sampled_token_ids,
        )

    def _find_request(self, request_id: str) -> Request | None:
        for req in self.running:
            if req.request_id == request_id:
                return req
        return None

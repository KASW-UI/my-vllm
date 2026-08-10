"""Scheduler component — mirrors vLLM V1 v1/core/sched/scheduler.py.

C016: migrated from scheduler/scheduler.py.
C029: prefix cache lookup via KVCacheManager.get_computed_blocks().
C030: chunked prefill — long prompts split by token_budget;
      num_computed_tokens tracks prefill progress per-chunk.
C037: implements SchedulerInterface ABC for standardized contract.
C038: resource model — explicit GPU/KV/token budget checks.
C051: multi-policy scheduling — FCFS/Priority via SchedulingPolicy.
C052: preemption — _preempt_request frees KV blocks + resets tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.core.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.interface import SchedulerInterface
from my_vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from my_vllm.v1.core.sched.request_queue import (
    SchedulingPolicy,
    create_request_queue,
)
from my_vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig


class Scheduler(SchedulerInterface):
    """Multi-policy scheduler with chunked prefill, prefix cache, resource model.

    C051: policy parameter selects FCFS or Priority scheduling.
    C052: preempt_request frees KV blocks, resets num_computed_tokens, requeues.
    """

    _EOS_TOKEN_ID = 2
    _DEFAULT_MAX_TOKENS = 256

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_manager: KVCacheManager | None = None,
    ):
        sc = vllm_config.scheduler_config

        if sc.policy == "priority":
            self.scheduling_policy = SchedulingPolicy.PRIORITY
        else:
            self.scheduling_policy = SchedulingPolicy.FCFS

        self.waiting = create_request_queue(self.scheduling_policy)
        self.running: list[Request] = []
        self.max_running = sc.max_num_seqs
        self.kv_cache_manager = kv_cache_manager
        self.max_num_batched_tokens = sc.max_num_batched_tokens

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)

    def schedule(self) -> SchedulerOutput:
        token_budget = self.max_num_batched_tokens
        new_req_datas: list[NewRequestData] = []
        new_blocks: dict[str, list[int]] = {}
        num_common_prefix_blocks: list[int] = []
        preempted_req_ids: set[str] = set()

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
        while self.waiting:
            if not self._has_gpu_budget():
                if not self._try_preempt_for_gpu_budget(preempted_req_ids):
                    break
            if not self._has_token_budget(token_budget):
                break

            req = self.waiting.pop_request()
            prompt_len = len(req.prompt_token_ids)
            num_cached = 0
            n_prefix_blocks = 0

            if self.kv_cache_manager is not None:
                cached_blocks, num_cached = (
                    self.kv_cache_manager.get_computed_blocks(req)
                )
                req.num_computed_tokens = num_cached
                n_prefix_blocks = len(cached_blocks)

                blocks = self.kv_cache_manager.allocate_slots(
                    req, prompt_len
                )
                if not self._has_kv_budget(blocks):
                    if not self._try_preempt_for_kv_budget(
                        req, prompt_len, preempted_req_ids
                    ):
                        self.waiting.prepend_request(req)
                        break
                    blocks = self.kv_cache_manager.allocate_slots(
                        req, prompt_len
                    )
                    if not self._has_kv_budget(blocks):
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
            num_common_prefix_blocks.append(n_prefix_blocks)

        cached_data = CachedRequestData(
            req_ids=[r.request_id for r in self.running
                     if r.status != RequestStatus.PREEMPTED],
            num_computed_tokens=[
                r.num_computed_tokens for r in self.running
                if r.status != RequestStatus.PREEMPTED
            ],
            num_output_tokens=[
                r.num_output_tokens for r in self.running
                if r.status != RequestStatus.PREEMPTED
            ],
        )

        total_tokens = sum(
            len(d.prompt_token_ids) for d in new_req_datas
        )
        total_tokens += len(self.running)

        kv_cache_block_copies = None
        if self.kv_cache_manager is not None:
            kv_cache_block_copies, _ = (
                self.kv_cache_manager.take_kv_cache_block_copies()
            )

        return SchedulerOutput(
            scheduled_new_reqs=new_req_datas,
            scheduled_cached_reqs=cached_data,
            num_scheduled_tokens={},
            total_num_scheduled_tokens=total_tokens,
            finished_req_ids=set(),
            preempted_req_ids=preempted_req_ids or None,
            num_common_prefix_blocks=num_common_prefix_blocks,
            kv_cache_block_copies=kv_cache_block_copies or None,
        )

    # --- C052 preemption (like vLLM _preempt_request) ---

    def _select_preemption_victims(self) -> list[Request]:
        """Select preemptible running requests sorted by priority (lowest first)."""
        candidates = [r for r in self.running
                       if r.status == RequestStatus.RUNNING]
        candidates.sort(key=lambda r: r.priority)
        return candidates

    def _preempt_request(self, victim: Request) -> None:
        """Preempt a running request — free KV blocks, reset tokens, requeue.
        
        Aligned with vLLM Scheduler._preempt_request: frees KV blocks,
        resets num_computed_tokens to 0, increments num_preemptions,
        sets PREEMPTED status, and puts back to waiting queue.
        """
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.free(victim.request_id)
        victim.num_computed_tokens = 0
        victim.num_preemptions += 1
        victim.status = RequestStatus.PREEMPTED
        self.running.remove(victim)
        self.waiting.prepend_request(victim)

    def _try_preempt_for_gpu_budget(
        self, preempted_ids: set[str]
    ) -> bool:
        """Preempt one victim to free a GPU slot."""
        if not self.running:
            return False
        victims = self._select_preemption_victims()
        if not victims:
            return False
        self._preempt_request(victims[0])
        preempted_ids.add(victims[0].request_id)
        return True

    def _try_preempt_for_kv_budget(
        self, request: Request, prompt_len: int,
        preempted_ids: set[str],
    ) -> bool:
        """Preempt victims until enough KV blocks are free for the request."""
        if not self.running or self.kv_cache_manager is None:
            return False
        victims = self._select_preemption_victims()
        for victim in victims:
            if victim.priority >= request.priority:
                continue
            self._preempt_request(victim)
            preempted_ids.add(victim.request_id)
            blocks = self.kv_cache_manager.allocate_slots(
                request, prompt_len
            )
            if blocks is not None:
                return True
        return False

    # --- C038 resource budget checks ---

    def _has_gpu_budget(self) -> bool:
        return len(self.running) < self.max_running

    def _has_token_budget(self, remaining: int) -> bool:
        return remaining > 0

    @staticmethod
    def _has_kv_budget(blocks) -> bool:
        return blocks is not None

    # --- request lifecycle ---

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

    def has_unfinished_requests(self) -> bool:
        return self.has_work()

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def finish_requests(
        self, request_ids: list[str], finished_status: RequestStatus
    ) -> list[Request]:
        finished: list[Request] = []
        for rid in request_ids:
            req = self._find_request(rid)
            if req is not None:
                req.status = finished_status
                self.finish(req)
                finished.append(req)
        return finished

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        """Update request states from model output.

        Mirrors vLLM: scheduler handles ALL EOS/length detection and
        produces EngineCoreOutput per request with finish_reason.
        OutputProcessor only does detokenization.
        """
        outputs: list[EngineCoreOutput] = []

        for i, rid in enumerate(model_output.req_ids):
            token_ids = model_output.sampled_token_ids[i]
            req = self._find_request(rid)
            if req is None:
                continue

            new_tokens: list[int] = []
            for tok in token_ids:
                if tok == self._EOS_TOKEN_ID:
                    req.status = RequestStatus.FINISHED_STOPPED
                    break
                req.append_output_token_ids(tok)
                new_tokens.append(tok)

            limit = req.max_tokens or self._DEFAULT_MAX_TOKENS
            if req.num_output_tokens >= limit and not req.is_finished():
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED

            finish_reason = None
            if req.is_finished():
                finish_reason = RequestStatus.get_finished_reason(req.status)

            outputs.append(EngineCoreOutput(
                request_id=rid,
                new_token_ids=new_tokens,
                finish_reason=finish_reason,
            ))

        return EngineCoreOutputs(outputs=outputs)

    def _find_request(self, request_id: str) -> Request | None:
        for req in self.running:
            if req.request_id == request_id:
                return req
        return None

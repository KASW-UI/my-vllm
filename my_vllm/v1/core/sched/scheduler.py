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
        self.lora_config = vllm_config.lora_config
        self.multi_tenant_config = vllm_config.multi_tenant_config

        if self.multi_tenant_config.enabled:
            self._per_tenant_usage: dict[str, dict[str, int]] = {
                t.tenant_id: {"gpu_slots": 0, "kv_blocks": 0}
                for t in self.multi_tenant_config.tenants
            }

    def add_request(self, request: Request):
        request.status = RequestStatus.WAITING
        self.waiting.add_request(request)

    def schedule(self) -> SchedulerOutput:
        self._promote_remote_kv_requests()
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

        # Record the LoRAs in scheduled running requests
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in self.running
                if req.lora_request and req.lora_request.lora_int_id > 0
            )

        # Pull new requests from waiting queue
        skipped_lora_reqs: list[Request] = []
        while self.waiting:
            if not self._has_gpu_budget():
                if not self._try_preempt_for_gpu_budget(preempted_req_ids):
                    break
            if not self._has_token_budget(token_budget):
                break

            req = self.waiting.pop_request()

            # Check that adding the request still respects max_loras
            if (
                self.lora_config
                and req.lora_request
                and (
                    len(scheduled_loras) == self.lora_config.max_loras
                    and req.lora_request.lora_int_id not in scheduled_loras
                )
            ):
                skipped_lora_reqs.append(req)
                continue

            # C120: check per-tenant resource quotas
            tenant_id = self._get_tenant_id(req)
            if tenant_id is not None:
                if not self._has_tenant_gpu_budget(tenant_id):
                    if not self._try_inter_tenant_preempt(
                        req, tenant_id, preempted_req_ids
                    ):
                        skipped_lora_reqs.append(req)
                        continue
                    if not self._has_tenant_gpu_budget(tenant_id):
                        skipped_lora_reqs.append(req)
                        continue

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

                # C120: check tenant KV budget after block allocation
                if (
                    tenant_id is not None
                    and not self._has_tenant_kv_budget(tenant_id)
                ):
                    self.waiting.prepend_request(req)
                    continue
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

            if self.lora_config and req.lora_request:
                scheduled_loras.add(req.lora_request.lora_int_id)

            self.running.append(req)

            new_req_datas.append(NewRequestData(
                req_id=req.request_id,
                prompt_token_ids=req.prompt_token_ids,
                block_ids=new_blocks.get(req.request_id, []),
                num_computed_tokens=num_cached,
            ))
            num_common_prefix_blocks.append(n_prefix_blocks)

        # Requeue LoRA-rejected requests for the next scheduling step
        for req in reversed(skipped_lora_reqs):
            self.waiting.prepend_request(req)

        # Build cached_data from running reqs that were NOT just scheduled
        # as new in this step. Mirrors vLLM v1/core/sched/scheduler.py:
        # _make_cached_request_data() — new reqs (scheduled_new_reqs) are
        # excluded so the runner consumes each request exactly once per step.
        new_req_ids = {d.req_id for d in new_req_datas}
        cached_running = [
            r for r in self.running
            if r.status != RequestStatus.PREEMPTED
            and r.request_id not in new_req_ids
        ]
        cached_data = CachedRequestData(
            req_ids=[r.request_id for r in cached_running],
            num_computed_tokens=[r.num_computed_tokens for r in cached_running],
            num_output_tokens=[r.num_output_tokens for r in cached_running],
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

    # --- C120 multi-tenant budget checks ---

    def _get_tenant_id(self, request: Request) -> str | None:
        """Get the tenant ID for a request, if multi-tenant is enabled."""
        if not self.multi_tenant_config.enabled:
            return None
        return getattr(request, "tenant_id", None)

    def _count_tenant_gpu_slots(self, tenant_id: str) -> int:
        return sum(
            1 for r in self.running
            if r.status != RequestStatus.PREEMPTED
            and getattr(r, "tenant_id", None) == tenant_id
        )

    def _count_tenant_kv_blocks(self, tenant_id: str) -> int:
        if self.kv_cache_manager is None:
            return 0
        total = 0
        for rid, blocks in self.kv_cache_manager.request_blocks.items():
            req = self._find_request(rid)
            if req is not None and getattr(req, "tenant_id", None) == tenant_id:
                total += len(blocks)
        return total

    def _has_tenant_gpu_budget(self, tenant_id: str) -> bool:
        tenant = self.multi_tenant_config.get_tenant(tenant_id)
        if tenant is None:
            return True
        used = self._count_tenant_gpu_slots(tenant_id)
        if used < tenant.max_gpu_slots:
            return True
        if self.multi_tenant_config.allow_burst and self._has_gpu_budget():
            return True
        return False

    def _has_tenant_kv_budget(self, tenant_id: str) -> bool:
        tenant = self.multi_tenant_config.get_tenant(tenant_id)
        if tenant is None or self.kv_cache_manager is None:
            return True
        used = self._count_tenant_kv_blocks(tenant_id)
        if used < tenant.max_kv_cache_blocks:
            return True
        if self.multi_tenant_config.allow_burst:
            return True
        return False

    def _try_inter_tenant_preempt(
        self, request: Request, tenant_id: str, preempted_ids: set[str],
    ) -> bool:
        """Preempt requests from lower-priority tenants to make room."""
        if not self.multi_tenant_config.preempt_lower_priority:
            return False
        if not self.running:
            return False

        tenant = self.multi_tenant_config.get_tenant(tenant_id)
        if tenant is None:
            return False

        victims = [
            r for r in self.running
            if r.status == RequestStatus.RUNNING
            and getattr(r, "tenant_id", None) != tenant_id
        ]
        # Sort by tenant priority (lowest first)
        victims.sort(key=lambda r: self._get_tenant_priority(getattr(r, "tenant_id", None)))

        for victim in victims:
            victim_priority = self._get_tenant_priority(getattr(victim, "tenant_id", None))
            if victim_priority >= tenant.priority:
                continue
            self._preempt_request(victim)
            preempted_ids.add(victim.request_id)
            return True
        return False

    def _get_tenant_priority(self, tenant_id: str | None) -> int:
        if tenant_id is None:
            return 0
        tenant = self.multi_tenant_config.get_tenant(tenant_id)
        return tenant.priority if tenant else 0

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

    # --- P/D disaggregation ---

    def _promote_remote_kv_requests(self) -> None:
        """Move WAITING_FOR_REMOTE_KVS requests to WAITING for recompute.

        Called at the start of each schedule() call. Requests whose KV
        transfer failed (under recompute policy) retain their valid
        num_computed_tokens and will be re-prefilled from that position.
        Aligns with vLLM Scheduler._update_waiting_for_remote_kv.
        """
        promoted: list[Request] = []
        for req in list(self.waiting):
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                promoted.append(req)
        for req in promoted:
            self.waiting.remove_request(req)
            req.status = RequestStatus.WAITING
            self.waiting.add_request(req)

    def reinsert_for_prefill(self, request: Request) -> None:
        """Insert a request that needs recompute after KV transfer failure.

        Called by the consumer EngineCore when a KV transfer fails under
        the recompute policy. The request's num_computed_tokens is already
        set to the valid prefix by the connector.
        """
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        self.waiting.add_request(request)

    # --- output processing ---

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

            # Mirrors vLLM v1/core/sched/scheduler.py:1240
            # (`request.num_computed_tokens += num_scheduled_token`). Each
            # sampled token advances the request's KV-cache pointer so the
            # next decode step reads the correct slot for the next query.
            req.num_computed_tokens += len(new_tokens)

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

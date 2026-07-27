import torch

from my_vllm.cache.block_pool import BlockPool
from my_vllm.cache.kv_cache_manager import KVCacheManager
from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.request import Request, RequestStatus


class EngineCore:
    """Core execution loop — mirrors vLLM V1 v1/engine/core.py:EngineCore.

    C017: EngineCore calls executor.execute_model(scheduler_output) once
    per step instead of iterating per-request.  Batch prefill replaces
    individual _prefill calls.  Decode is still per-request internally
    (HF past_key_values limitation) but encapsulated in GPUModelRunner.
    """

    def __init__(
        self,
        executor: Executor,
        device: torch.device,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
        max_running: int = 8,
        num_kv_blocks: int = 512,
        block_size: int = 16,
    ):
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id
        self.device = device

        self.executor = executor
        self.worker = executor.get_worker()

        block_pool = BlockPool(num_kv_blocks, block_size)
        self.kv_cache_manager = KVCacheManager(block_pool=block_pool)
        self.scheduler = Scheduler(
            max_running=max_running,
            kv_cache_manager=self.kv_cache_manager,
        )
        self.finished: list[Request] = []

    def add_request(self, request: Request):
        self.scheduler.add_request(request)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[Request]:
        newly_finished: list[Request] = []

        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return newly_finished

        model_output = self.executor.execute_model(scheduler_output)
        if model_output is None:
            return newly_finished

        req_map = {r.request_id: r for r in self.scheduler.running}

        for rid, token_ids in model_output.sampled_token_ids.items():
            req = req_map.get(rid)
            if req is None:
                continue
            for token in token_ids:
                if token == self.eos_token_id:
                    req.status = RequestStatus.FINISHED_STOPPED
                    self._finish_request(req)
                    newly_finished.append(req)
                    break
                req.append_output_token_ids(token)

        for req in list(self.scheduler.running):
            rid = req.request_id
            if req.is_finished():
                continue
            step_val = self.worker.get_step(rid)
            limit = req.max_tokens or self.max_new_tokens
            if step_val >= limit:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._finish_request(req)
                newly_finished.append(req)

        return newly_finished

    def _finish_request(self, req: Request):
        rid = req.request_id
        self.kv_cache_manager.free(rid)
        self.worker.cleanup(rid)
        self.scheduler.finish(req)
        self.finished.append(req)

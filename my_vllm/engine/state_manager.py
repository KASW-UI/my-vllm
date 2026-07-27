import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.allocator import KVCacheAllocator
from my_vllm.cache.kv_cache_manager import KVCacheManager
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.scheduler.scheduler import Scheduler


class StateManager:
    """Execution engine driven by Scheduler + KVCacheManager.

    C008: KV cache management delegated to KVCacheManager.
    Scheduler now queries KVCacheManager before promoting requests.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
        max_running: int = 8,
        max_kv_tokens: int = 8192,
    ):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id

        allocator = KVCacheAllocator(max_total_tokens=max_kv_tokens)
        self.kv_cache_manager = KVCacheManager(allocator=allocator)
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

        new_requests = self.scheduler.schedule()
        for req in new_requests:
            self._prefill(req)

        for req in self.scheduler.running:
            self._decode_step(req)

        for req in list(self.scheduler.running):
            rid = req.request_id
            step = self.kv_cache_manager.get_step(rid)
            if step >= self.max_new_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self.kv_cache_manager.free(rid)
                self.scheduler.finish(req)
                self.finished.append(req)
                newly_finished.append(req)

        return newly_finished

    def _prefill(self, req: Request):
        rid = req.request_id
        cache = self.kv_cache_manager.get_cache(rid)
        if cache is None:
            req.status = RequestStatus.WAITING
            self.scheduler.finish(req)
            self.scheduler.waiting.insert(0, req)
            return
        input_ids = torch.tensor(
            [req.prompt_token_ids], device=self.device
        )
        logits = cache.prefill(self.model, input_ids)
        first = greedy_sample(logits).item()

        if first == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self.kv_cache_manager.free(rid)
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(first)
        self.kv_cache_manager.set_step(rid, 1)
        self.kv_cache_manager.set_next_token(rid, first)

    def _decode_step(self, req: Request):
        rid = req.request_id
        cache = self.kv_cache_manager.get_cache(rid)
        if cache is None:
            return

        token = self.kv_cache_manager.get_next_token(rid)
        step = self.kv_cache_manager.get_step(rid)

        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()

        if next_token == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self.kv_cache_manager.free(rid)
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(next_token)
        self.kv_cache_manager.set_next_token(rid, next_token)
        self.kv_cache_manager.set_step(rid, step + 1)

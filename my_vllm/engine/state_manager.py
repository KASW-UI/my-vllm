import torch

from my_vllm.cache.block_pool import BlockPool
from my_vllm.cache.kv_cache_manager import KVCacheManager
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.scheduler.scheduler import Scheduler
from my_vllm.worker.worker_base import WorkerBase


class StateManager:
    """Execution engine: Scheduler → Worker → KVCacheManager.

    C011: accesses ModelRunner through Worker instead of
    holding model directly.
    """

    def __init__(
        self,
        worker: WorkerBase,
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

        worker.init_device()
        self.worker = worker
        self.runner = worker.get_model_runner()

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

        new_requests = self.scheduler.schedule()
        for req in new_requests:
            self._prefill(req)

        for req in self.scheduler.running:
            self._decode_step(req)

        for req in list(self.scheduler.running):
            rid = req.request_id
            step = self.runner.get_step(rid)
            if step >= self.max_new_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self.kv_cache_manager.free(rid)
                self.runner.cleanup(rid)
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
        first = self.runner.prefill(rid, input_ids, cache)

        if first == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self.kv_cache_manager.free(rid)
            self.runner.cleanup(rid)
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(first)

    def _decode_step(self, req: Request):
        rid = req.request_id
        cache = self.kv_cache_manager.get_cache(rid)
        if cache is None:
            return

        next_token = self.runner.decode(rid, cache)

        if next_token == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self.kv_cache_manager.free(rid)
            self.runner.cleanup(rid)
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(next_token)

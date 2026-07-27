import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.scheduler.scheduler import Scheduler


class StateManager:
    """Execution engine driven by Scheduler.

    Scheduler decides *who* runs; StateManager handles *how*
    (prefill / decode / KV cache lifecycle).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
        max_running: int = 8,
    ):
        self.model = model
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.eos_token_id = eos_token_id

        self.scheduler = Scheduler(max_running=max_running)
        self.finished: list[Request] = []
        self._caches: dict[str, KVCache] = {}
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

    def add_request(self, request: Request):
        self.scheduler.add_request(request)

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[Request]:
        newly_finished: list[Request] = []

        # ── Schedule: promote waiting → running ──
        new_requests = self.scheduler.schedule()
        for req in new_requests:
            self._prefill(req)

        # ── Execute: one decode step per running ──
        for req in self.scheduler.running:
            self._decode_step(req)

        # ── Collect finished ──
        for req in list(self.scheduler.running):
            rid = req.request_id
            step = self._steps.get(rid, 0)
            if step >= self.max_new_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._cleanup(rid)
                self.scheduler.finish(req)
                self.finished.append(req)
                newly_finished.append(req)

        return newly_finished

    def _prefill(self, req: Request):
        cache = KVCache()
        input_ids = torch.tensor(
            [req.prompt_token_ids], device=self.device
        )
        logits = cache.prefill(self.model, input_ids)
        first = greedy_sample(logits).item()

        if first == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(first)
        self._caches[req.request_id] = cache
        self._steps[req.request_id] = 1
        self._next_token[req.request_id] = first

    def _decode_step(self, req: Request):
        rid = req.request_id
        cache = self._caches.get(rid)
        if cache is None:
            return

        token = self._next_token[rid]
        step = self._steps[rid]

        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()

        if next_token == self.eos_token_id:
            req.status = RequestStatus.FINISHED_STOPPED
            self._cleanup(rid)
            self.scheduler.finish(req)
            self.finished.append(req)
            return

        req.output_token_ids.append(next_token)
        self._next_token[rid] = next_token
        self._steps[rid] = step + 1

    def _cleanup(self, request_id: str):
        self._caches.pop(request_id, None)
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)

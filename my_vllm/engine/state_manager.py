import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.engine.waiting_queue import WaitingQueue
from my_vllm.engine.running_queue import RunningQueue


class StateManager:
    """Manages request lifecycle: waiting → running → finished.

    Per-step execution: each call to step() does one decode step
    for every running request.  Scheduling logic is inlined here
    (extracted to Scheduler in C007).
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
        self.max_running = max_running

        self.waiting = WaitingQueue()
        self.running = RunningQueue()
        self.finished: list[Request] = []
        self._caches: dict[str, KVCache] = {}
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

    def add_request(self, request: Request):
        self.waiting.push(request)

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def step(self) -> list[Request]:
        newly_finished: list[Request] = []

        # Promote waiting → running
        while len(self.running) < self.max_running and self.waiting:
            req = self.waiting.pop()
            req.status = RequestStatus.RUNNING
            self.running.add(req)

            cache = KVCache()
            input_ids = torch.tensor(
                [req.prompt_token_ids], device=self.device
            )
            logits = cache.prefill(self.model, input_ids)
            first = greedy_sample(logits).item()

            if first == self.eos_token_id:
                req.status = RequestStatus.FINISHED_STOPPED
                self.running.remove(req)
                self.finished.append(req)
                newly_finished.append(req)
                continue

            req.output_token_ids.append(first)
            self._caches[req.request_id] = cache
            self._steps[req.request_id] = 1
            self._next_token[req.request_id] = first

        # Decode one step for each running request
        for req in self.running:
            rid = req.request_id
            cache = self._caches[rid]
            token = self._next_token[rid]
            step = self._steps[rid]

            next_input = torch.tensor([token], device=self.device)
            logits = cache.decode(self.model, next_input)
            next_token = greedy_sample(logits).item()

            if next_token == self.eos_token_id:
                req.status = RequestStatus.FINISHED_STOPPED
                self._cleanup(rid)
                self.running.remove(req)
                self.finished.append(req)
                newly_finished.append(req)
                continue

            req.output_token_ids.append(next_token)
            self._next_token[rid] = next_token
            self._steps[rid] = step + 1

            if self._steps[rid] >= self.max_new_tokens:
                req.status = RequestStatus.FINISHED_LENGTH_CAPPED
                self._cleanup(rid)
                self.running.remove(req)
                self.finished.append(req)
                newly_finished.append(req)

        return newly_finished

    def _cleanup(self, request_id: str):
        self._caches.pop(request_id, None)
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)

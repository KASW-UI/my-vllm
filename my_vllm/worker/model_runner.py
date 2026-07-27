import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache


class ModelRunner:
    """Encapsulates model forward + sampling.

    Extracted from StateManager in C010.  Owns per-request
    execution state (step count, next token) that was previously
    stored in KVCacheManager.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        eos_token_id: int = 2,
    ):
        self.model = model
        self.device = device
        self.eos_token_id = eos_token_id
        self._steps: dict[str, int] = {}
        self._next_token: dict[str, int] = {}

    def prefill(
        self,
        request_id: str,
        input_ids: torch.Tensor,
        cache: KVCache,
    ) -> int:
        logits = cache.prefill(self.model, input_ids)
        token = greedy_sample(logits).item()
        if token != self.eos_token_id:
            self._steps[request_id] = 1
            self._next_token[request_id] = token
        return token

    def decode(
        self,
        request_id: str,
        cache: KVCache,
    ) -> int:
        token = self._next_token[request_id]
        next_input = torch.tensor([token], device=self.device)
        logits = cache.decode(self.model, next_input)
        next_token = greedy_sample(logits).item()
        self._next_token[request_id] = next_token
        self._steps[request_id] += 1
        return next_token

    def get_step(self, request_id: str) -> int:
        return self._steps.get(request_id, 0)

    def cleanup(self, request_id: str):
        self._steps.pop(request_id, None)
        self._next_token.pop(request_id, None)

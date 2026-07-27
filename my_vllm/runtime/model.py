import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache
from my_vllm.engine.request import Request, RequestStatus
from my_vllm.batch.batch_executor import BatchExecutor


class LLM:
    def __init__(self, hf_model: nn.Module):
        self.model = hf_model
        self.device = next(hf_model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to(self.device))
        return outputs.logits

    def generate(
        self,
        request: Request,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
    ):
        request.status = RequestStatus.RUNNING
        cache = KVCache()

        input_ids = torch.tensor(
            [request.prompt_token_ids], device=self.device
        )
        logits = cache.prefill(self.model, input_ids)
        next_token_id = greedy_sample(logits).item()

        if next_token_id == eos_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            return

        request.output_token_ids.append(next_token_id)

        for _ in range(1, max_new_tokens):
            next_input = torch.tensor(
                [next_token_id], device=self.device
            )
            logits = cache.decode(self.model, next_input)
            next_token_id = greedy_sample(logits).item()

            if next_token_id == eos_token_id:
                request.status = RequestStatus.FINISHED_STOPPED
                return

            request.output_token_ids.append(next_token_id)

        request.status = RequestStatus.FINISHED_LENGTH_CAPPED

    def generate_batch(
        self,
        requests: list[Request],
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
    ):
        executor = BatchExecutor(self.model, self.device)
        executor.execute(requests, max_new_tokens, eos_token_id)

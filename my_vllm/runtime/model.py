import torch
import torch.nn as nn

from my_vllm.generation.sampler import greedy_sample
from my_vllm.cache.kv_cache import KVCache


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
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        generated_ids = []
        cache = KVCache()

        logits = cache.prefill(self.model, input_ids.to(self.device))
        next_token_id = greedy_sample(logits).item()
        if next_token_id == eos_token_id:
            return torch.tensor([], dtype=torch.long)
        generated_ids.append(next_token_id)

        for _ in range(1, max_new_tokens):
            next_input = torch.tensor(
                [next_token_id], device=self.device
            )
            logits = cache.decode(self.model, next_input)
            next_token_id = greedy_sample(logits).item()
            if next_token_id == eos_token_id:
                break
            generated_ids.append(next_token_id)

        return torch.tensor(generated_ids, dtype=torch.long)

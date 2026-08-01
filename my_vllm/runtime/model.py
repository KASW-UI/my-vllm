"""LLM — minimal shim for backward-compatible C001-C005 test support.

C032: legacy KVCache/greedy_sample removed.  This module provides
a thin LLM class using raw HF model API + torch.argmax.
"""

import torch
import torch.nn as nn

from my_vllm.v1.request import Request, RequestStatus


class LLM:
    """Minimal LLM wrapper for C001-C005 era tests.

    Uses HF model directly without v1 infrastructure.
    """

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
        past_key_values = None

        input_ids = torch.tensor(
            [request.prompt_token_ids], device=self.device,
        )
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())

        if next_token_id == eos_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            return

        request.append_output_token_ids(next_token_id)

        for _ in range(1, max_new_tokens):
            next_input = torch.tensor(
                [[next_token_id]], device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            next_token_id = int(
                torch.argmax(outputs.logits[:, -1, :], dim=-1).item()
            )

            if next_token_id == eos_token_id:
                request.status = RequestStatus.FINISHED_STOPPED
                return

            request.append_output_token_ids(next_token_id)

        request.status = RequestStatus.FINISHED_LENGTH_CAPPED

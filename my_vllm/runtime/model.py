import torch
import torch.nn as nn


class LLM:
    def __init__(self, hf_model: nn.Module):
        self.model = hf_model
        self.device = next(hf_model.parameters()).device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to(self.device))
        return outputs.logits

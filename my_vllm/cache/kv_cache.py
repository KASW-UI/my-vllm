import torch
import torch.nn as nn


class KVCache:
    """Continuous-memory KV cache for a single request.

    Uses the HF model's built-in past_key_values mechanism.
    Each layer stores one contiguous tensor for K and one for V,
    which grow token-by-token during decode.
    """

    def __init__(self):
        self._cached: tuple | None = None

    def prefill(
        self, model: nn.Module, input_ids: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        self._cached = outputs.past_key_values
        return outputs.logits

    def decode(
        self, model: nn.Module, input_id: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(
                input_ids=input_id.unsqueeze(0),
                past_key_values=self._cached,
                use_cache=True,
            )
        self._cached = outputs.past_key_values
        return outputs.logits

"""Logits processor — aligns with vLLM vllm/model_executor/layers/logits_processor.py.

Architecture preserved: LogitsProcessor (PluggableLayer) — lm_head projection,
gather (TP), soft_cap, scale.

Engineering details skipped: current_platform.use_all_gather (always all_gather),
head_dtype support, get_top_tokens optimization, from_config dependency.
"""

import torch
import torch.nn.functional as F

from my_vllm.distributed.parallel_state import all_gather
from my_vllm.model_executor.custom_op import PluggableLayer
from my_vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding


class LogitsProcessor(PluggableLayer):
    """Process logits — aligns with vLLM LogitsProcessor.

    This layer does the following:
    1. Project hidden_states through lm_head to get logits.
    2. Gather logits from all TP ranks.
    3. Remove padding (org_vocab_size).
    4. Apply soft_cap if set.
    5. Scale logits if needed.
    """

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        self.logits_as_input = logits_as_input
        self.org_vocab_size = org_vocab_size or vocab_size
        self.soft_cap = soft_cap

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap
            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = all_gather(logits, dim=-1)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s

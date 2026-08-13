"""LogitsProcessor package — penalty functions for pre-sampling.

Mirrors vLLM V1 v1/sample/logits_processor/.

C132: split from the former single-file `v1/sample/logits_processor.py`
into the vLLM-aligned package layout
(vllm/v1/sample/logits_processor/{interface.py,builtin.py,state.py}).

Public API re-exports the penalty functions that `Sampler` calls
before temperature / top-k / top-p filtering.
"""

from .builtin import (
    apply_frequency_penalty,
    apply_presence_penalty,
    apply_repetition_penalty,
)
from .interface import LogitsProcessor

__all__ = [
    "LogitsProcessor",
    "apply_frequency_penalty",
    "apply_presence_penalty",
    "apply_repetition_penalty",
]
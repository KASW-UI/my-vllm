"""DraftModelProposer — separate-checkpoint draft-model proposer.

Mirrors vLLM V1 v1/spec_decode/draft_model.py.

vLLM's `DraftModelProposer` runs a standalone draft model (loaded
from a separate checkpoint with potentially different vocab/tokenizer)
to draft speculative tokens. It supports heterogeneous-vocab mapping
(`VocabMapping`) when target / draft tokenizers differ.

C138: skeleton — class definition + the constructor signature.
Implementation (vocab mapping, draft-model forward, rejection
sampling) is deferred to a follow-up that lands once
`SpeculativeConfig.method == "draft_model"` is wired through the
proposer dispatcher. The class exists for 1:1 file-tree alignment.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from my_vllm.config.vllm import VllmConfig


logger = logging.getLogger(__name__)


class SpecDecodeBaseProposer(ABC):
    """Abstract base for spec-decode proposers.

    Mirrors vLLM V1 v1/spec_decode/llm_base_proposer.py:SpecDecodeBaseProposer
    (simplified; my-vllm doesn't have the full Proposer interface yet).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: "torch.device",
        pass_hidden_states_to_model: bool = False,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.pass_hidden_states_to_model = pass_hidden_states_to_model
        self.runner = runner
        self.speculative_config = vllm_config.speculative_config

    @abstractmethod
    def prepare_inputs(
        self,
        scheduler_output,
    ) -> "tuple | None":
        """Build the spec-decode batch from a `SchedulerOutput`.

        Returns `(input_ids, positions, ...)` ready to feed into the
        draft model, or `None` to skip spec-decode this step.
        """

    @abstractmethod
    def propose(self, input_ids, positions, hidden_states):
        """Run the draft model to produce speculative tokens.

        Returns the draft token IDs (per request, padded to
        `num_speculative_tokens`).
        """


class DraftModelProposer(SpecDecodeBaseProposer):
    """Proposer that runs a separate-checkpoint draft model.

    Mirrors vLLM V1 v1/spec_decode/draft_model.py:DraftModelProposer (L19).

    C138: skeleton — class definition + constructor. The full
    vLLM impl handles vocab mapping, draft TP mismatch detection,
    weight loading, and the draft-model forward pass. None of those
    land in my-vllm's minimal subset today; they require wiring
    through `SpeculativeConfig.method == "draft_model"` end-to-end.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: "torch.device",
        runner=None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=False,
            runner=runner,
        )
        logger.info(
            "DraftModelProposer created (skeleton) — "
            "draft_model=%s num_speculative=%d",
            vllm_config.speculative_config.draft_model_path,
            vllm_config.speculative_config.num_speculative_tokens,
        )

    def prepare_inputs(self, scheduler_output):
        raise NotImplementedError(
            "DraftModelProposer.prepare_inputs: skeleton only — "
            "wires up in a follow-up once SpeculativeConfig.method == "
            "'draft_model' is dispatched end-to-end."
        )

    def propose(self, input_ids, positions, hidden_states):
        raise NotImplementedError(
            "DraftModelProposer.propose: skeleton only — "
            "wires up in a follow-up once the draft model is loaded "
            "and the rejection-sampling pass is implemented."
        )


__all__ = ["DraftModelProposer", "SpecDecodeBaseProposer"]
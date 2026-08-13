"""EagleProposer — EAGLE draft-model proposer (single-layer variant).

Mirrors vLLM V1 v1/spec_decode/eagle.py.

vLLM's `EagleProposer` extends `SpecDecodeBaseProposer` with
`pass_hidden_states_to_model=True` — the draft model consumes the
target model's last hidden states (not the token IDs) to draft the
next token. EAGLE is the canonical example of a hidden-state-driven
draft method.

C138: skeleton — class definition + the vLLM-canonical
constructor flag (`pass_hidden_states_to_model=True`). The full
EAGLE forward pass requires:
  1. Tree-mask attention prep (`v1/spec_decode/utils.py:67`
     Triton kernels — out of scope for my-vllm).
  2. Single-layer draft model loaded from
     `SpeculativeConfig.draft_model_path`.
  3. Top-K token tree generation + rejection sampling.

This skeleton exists for 1:1 file-tree alignment; a follow-up
adds the forward pass + tree-mask prep.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from my_vllm.v1.spec_decode.draft_model import SpecDecodeBaseProposer

if TYPE_CHECKING:
    import torch

    from my_vllm.config.vllm import VllmConfig


logger = logging.getLogger(__name__)


class EagleProposer(SpecDecodeBaseProposer):
    """EAGLE draft-model proposer.

    Mirrors vLLM V1 v1/spec_decode/eagle.py:EagleProposer.

    EAGLE (Extrapolation Algorithm for Greater Language-model
    Efficiency) is a hidden-state-driven speculative decoder:
    the draft model is a single transformer layer that consumes
    the target model's last hidden states to predict the next
    token. Tree-mask attention lets one draft-model forward
    produce a tree of candidate tokens for parallel verification.

    C138: skeleton — constructor only. `prepare_inputs` /
    `propose` raise NotImplementedError; the full impl requires
    tree-mask prep kernels (deferred to 阶段 3 follow-up).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: "torch.device",
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )
        logger.info(
            "EagleProposer created (skeleton) — "
            "layers=%d input_layernorm=%s",
            vllm_config.speculative_config.eagle_layers,
            vllm_config.speculative_config.eagle_use_input_layernorm,
        )

    def prepare_inputs(self, scheduler_output):
        raise NotImplementedError(
            "EagleProposer.prepare_inputs: skeleton only — "
            "tree-mask prep kernels deferred to 阶段 3 follow-up."
        )

    def propose(self, input_ids, positions, hidden_states):
        raise NotImplementedError(
            "EagleProposer.propose: skeleton only — "
            "single-layer draft model + tree-top-k + "
            "rejection sampling deferred to 阶段 3 follow-up."
        )


__all__ = ["EagleProposer"]
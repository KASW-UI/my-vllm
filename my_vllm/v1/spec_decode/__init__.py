"""Spec-decode package — speculative decoding framework.

Mirrors vLLM V1 v1/spec_decode/__init__.py.

C138: re-exports the proposers, base class, metrics, and helpers
created in this stage.  The proposer dispatcher
(``create_spec_decode_proposer(vllm_config, ...)``) is wired in C223
for the ``method == "ngram"`` path; other methods (EAGLE / draft-
model / suffix / etc.) remain stubs and would be dispatched to their
own implementations once those land.

C223: ``NgramProposer`` is the first fully-implemented proposer in
this subset — extracted from
``vllm/v1/spec_decode/ngram_proposer.py``.  See
``my_vllm.v1.spec_decode.ngram_proposer`` for the algorithm and the
Numba-JIT'd batch kernel.
"""

from my_vllm.v1.spec_decode.draft_model import (
    DraftModelProposer,
    SpecDecodeBaseProposer,
)
from my_vllm.v1.spec_decode.eagle import EagleProposer
from my_vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging,
    SpecDecodingStats,
)
from my_vllm.v1.spec_decode.ngram_proposer import NgramProposer
from my_vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    extend_all_queries_by_N,
    next_power_of_2,
)


def create_spec_decode_proposer(
    vllm_config,
    device=None,
    runner=None,
):
    """Dispatcher — returns the proposer for ``vllm_config.
    speculative_config.method``.

    Mirrors vLLM's ``gpu_model_runner.py:596-611`` drafter selection.
    Only the ``"ngram"`` path is implemented in my-vllm today; the
    other methods are documented as deferred to their per-method
    implementation stage.
    """
    spec_config = vllm_config.speculative_config
    if spec_config is None:
        return None
    method = spec_config.method
    if method is None:
        return None
    if method == "ngram":
        # ngram proposer is device-agnostic (pure Numba / NumPy).
        return NgramProposer(vllm_config)
    if method == "draft_model":
        return DraftModelProposer(
            vllm_config=vllm_config, device=device, runner=runner
        )
    if method == "eagle":
        return EagleProposer(
            vllm_config=vllm_config, device=device, runner=runner
        )
    raise NotImplementedError(
        f"Speculative method {method!r} is not implemented in my-vllm "
        f"today. Only 'ngram' has a real implementation; "
        f"'draft_model' / 'eagle' are class shells (see "
        f"my_vllm.v1.spec_decode.{{draft_model,eagle}})."
    )


__all__ = [
    "DraftModelProposer",
    "EagleProposer",
    "NgramProposer",
    "PADDING_SLOT_ID",
    "SpecDecodingLogging",
    "SpecDecodeBaseProposer",
    "SpecDecodingStats",
    "create_spec_decode_proposer",
    "extend_all_queries_by_N",
    "next_power_of_2",
]
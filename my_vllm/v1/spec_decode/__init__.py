"""Spec-decode package — speculative decoding framework.

Mirrors vLLM V1 v1/spec_decode/__init__.py.

C138: re-exports the proposers, base class, metrics, and helpers
created in this stage. The proposer dispatcher
(`create_spec_decode_proposer(vllm_config, ...)`) is deferred to a
follow-up that wires `SpeculativeConfig.method` →
`EagleProposer / DraftModelProposer / ...`.
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
from my_vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    extend_all_queries_by_N,
    next_power_of_2,
)

__all__ = [
    "DraftModelProposer",
    "EagleProposer",
    "PADDING_SLOT_ID",
    "SpecDecodingLogging",
    "SpecDecodeBaseProposer",
    "SpecDecodingStats",
    "extend_all_queries_by_N",
    "next_power_of_2",
]
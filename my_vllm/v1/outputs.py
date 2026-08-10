"""Output data structures — mirrors vLLM V1 v1/outputs.py.

C015: minimal placeholder definitions. Will be extended in C016-C017.
EngineCoreOutput / EngineCoreOutputs moved to v1/engine/__init__.py (L7 alignment).
"""

from dataclasses import dataclass


@dataclass
class SamplerOutput:
    """Output from the Sampler."""

    sampled_token_ids: list[int]  # [num_reqs] — one token per request


@dataclass
class ModelRunnerOutput:
    """Output from ModelRunner after executing a batch.

    C015 minimal definition. Uses index-aligned parallel lists
    matching real vLLM's IPC-optimized layout.
    """

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]

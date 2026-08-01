"""Output data structures — mirrors vLLM V1 v1/outputs.py.

C015: minimal placeholder definitions. Will be extended in C016-C017.
C032: EngineCoreOutputs for scheduler.update_from_output() flow.
"""

from dataclasses import dataclass, field


@dataclass
class SamplerOutput:
    """Output from the Sampler."""

    sampled_token_ids: list[int]  # [num_reqs] — one token per request


@dataclass
class ModelRunnerOutput:
    """Output from ModelRunner after executing a batch.

    C015 minimal definition — only req_ids and sampled_token_ids.
    """

    req_ids: list[str]
    sampled_token_ids: dict[str, list[int]]  # req_id → list of new tokens


@dataclass
class EngineCoreOutput:
    """Output from EngineCore per-step for a finished request."""

    request_id: str
    new_token_ids: list[int]
    finish_reason: str | None  # "stop" | "length" | None


@dataclass
class EngineCoreOutputs:
    """Aggregated output from scheduler.update_from_output().

    C032: carries finished requests and per-request sampled tokens
    so EngineCore.step() delegates state update to the scheduler.
    """

    finished_requests: list = field(default_factory=list)
    sampled_token_ids: dict[str, list[int]] = field(default_factory=dict)

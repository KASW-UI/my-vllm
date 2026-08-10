"""Engine I/O types — mirrors vLLM V1 v1/engine/__init__.py.

C015: FinishReason (IntEnum) matching real vLLM.
"""

import enum
from dataclasses import dataclass, field


class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4


FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


@dataclass
class EngineCoreOutput:
    """Output from EngineCore for a single request in one step.

    Mirrors vLLM V1 v1/engine/__init__.py:EngineCoreOutput (simplified).
    finish_reason is now a proper FinishReason IntEnum (L7 alignment).
    """

    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None


@dataclass
class EngineCoreOutputs:
    """Aggregated output from scheduler.update_from_output().

    Mirrors vLLM V1 v1/engine/__init__.py:EngineCoreOutputs (simplified).
    """

    finished_requests: list = field(default_factory=list)
    sampled_token_ids: dict[str, list[int]] = field(default_factory=dict)

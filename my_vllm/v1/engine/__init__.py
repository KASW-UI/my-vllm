"""Engine I/O types — mirrors vLLM V1 v1/engine/__init__.py.

C015: FinishReason (IntEnum) matching real vLLM.
C124: EngineCoreRequestType (enum) for the busy-loop input queue.
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


class EngineCoreRequestType(enum.Enum):
    """Request types sent over the EngineCore input queue.

    Mirrors vLLM V1 v1/engine/__init__.py:EngineCoreRequestType (simplified).
    Only ADD / ABORT / WAKEUP are kept; UTILITY / EXECUTOR_FAILED /
    START_DP_WAVE are out of scope (LOOP-PLAN §2 C124).

    Note: vLLM uses byte strings (``b"\\x00"`` etc.) so the tag can ride on
    a ZMQ socket without separate encoding. my-vllm's input queue is in-
    process only, so we use plain ints — see LOOP-PLAN §3.
    """

    ADD = 0
    ABORT = 1
    WAKEUP = 2


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
    Contains per-request EngineCoreOutput objects (not raw token dicts).
    """

    outputs: list[EngineCoreOutput] = field(default_factory=list)

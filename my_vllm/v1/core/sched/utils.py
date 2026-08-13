"""Scheduler utility helpers — pure data-model functions.

Mirrors vLLM V1 v1/core/sched/utils.py.

Provides pure helpers used by the Scheduler (and EngineCore's
`update_from_output`) for stop-condition handling, list filtering,
and repetition-pattern detection. The scheduler does not yet
invoke these helpers inline (its stop check is inline in
`update_from_output` for C016 compat); the file exists for 1:1
file-tree alignment so a future stop-handling refactor only
requires scheduler edits, not package refactoring.

C135: extracted from vLLM's
vllm/v1/core/sched/utils.py:1 verbatim (the public functions),
trimmed to the parts relevant to my-vllm's single-process
scheduler — RepetitionDetectionParams is a stub since we don't
yet expose repetition-detection sampling params.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from my_vllm.v1.request import Request, RequestStatus


@dataclass
class RepetitionDetectionParams:
    """Stub for vLLM's sampling_params.RepetitionDetectionParams.

    my-vllm does not yet expose repetition-detection sampling
    params (out of scope per docs/alignment-plan.md — Repetition
    detection is SamplingParams-level, and we keep SamplingParams
    minimal). The stub exists so vLLM's `check_stop` signature
    can be referenced without `ImportError` if a future change
    wires it in.
    """

    max_pattern_size: int = 0
    min_pattern_size: int = 0
    min_count: int = 0


def _has_repeating_pattern(
    token_ids: Sequence[int],
    pattern_len: int,
    repetition_min_count: int,
) -> bool:
    """Check if the tail of token_ids contains a repeating pattern.

    Compares the last `pattern_len` tokens against the preceding
    `(repetition_min_count - 1)` repetitions of the same length.
    """
    for n in range(1, pattern_len + 1):
        target_token = token_ids[-n]
        for m in range(1, repetition_min_count):
            if token_ids[-(pattern_len * m + n)] != target_token:
                return False
    return True


def check_sequence_repetition(
    token_ids: Sequence[int],
    params: RepetitionDetectionParams,
) -> bool:
    """Check if a sequence of token IDs has a repetition pattern.

    Args:
        token_ids: list of token IDs.
        params: repetition detection parameters.

    Returns:
        True if a repetition pattern is found, False otherwise.
    """
    max_pattern_size = params.max_pattern_size
    min_pattern_size = params.min_pattern_size
    min_count = params.min_count

    if min_pattern_size <= 0:
        min_pattern_size = 1

    if max_pattern_size <= 0 or min_count < 2 or min_pattern_size > max_pattern_size:
        return False

    for pattern_len in range(min_pattern_size, max_pattern_size + 1):
        if pattern_len * min_count > len(token_ids):
            return False

        if _has_repeating_pattern(token_ids, pattern_len, min_count):
            return True

    return False


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in `items_to_remove`.

    Optimizes for the common single-item case (in-place `list.remove`)
    and falls back to list comprehension for multi-item removal.
    Returns the modified list (in-place for single item) or a new
    list (for multiple items).
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        item = next(iter(items_to_remove))
        try:
            lst.remove(item)
        except ValueError:
            pass
        return lst
    return [item for item in lst if item not in items_to_remove]


def check_stop(request: Request, max_model_len: int) -> bool:
    """Comprehensive stop-condition check.

    Mirrors vLLM V1 v1/core/sched/utils.py:check_stop (L94).
    Updates `request.status` if a stop condition fires and returns
    True; returns False otherwise. Callers should also handle
    FINISHED_ABORTED separately (it is set externally, not by
    this function).
    """
    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]
    if last_token_id == sampling_params.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    repetition_detection = getattr(sampling_params, "repetition_detection", None)
    if repetition_detection is not None and (
        check_sequence_repetition(
            request.output_token_ids,
            repetition_detection,
        )
    ):
        request.status = RequestStatus.FINISHED_REPETITION
        request.stop_reason = "repetition_detected"
        return True

    return False


__all__ = [
    "RepetitionDetectionParams",
    "check_sequence_repetition",
    "check_stop",
    "remove_all",
]
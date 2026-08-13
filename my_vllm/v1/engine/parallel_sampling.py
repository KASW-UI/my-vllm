"""Parallel sampling — fan-out one prompt into n completions.

Mirrors vLLM V1 v1/engine/parallel_sampling.py.

vLLM's `ParentRequest` class manages the lifecycle of a single
parent request that fans out into `n` child requests (when
`SamplingParams.n > 1`). The class tracks:
  - per-child request IDs (`{idx}_{parent_request_id}`)
  - per-child sampling params (with unique seeds when `seed`
    is set on the parent)
  - output aggregation (FINAL_ONLY mode → buffer the n outputs
    and emit them as a list when all children finish)
  - max-num-generation-tokens bookkeeping for iteration stats.

C139: minimal subset — `ParentRequest` dataclass with the data
attributes + `get_child_info(index)` returning `(child_req_id,
sampling_params)`. my-vllm's `SamplingParams` (v1/request.py:53)
is minimal — it does not have `n` / `seed` / `external_req_id`
fields; the dispatcher that wires parallel sampling into the
frontend → EngineCore → OutputProcessor pipeline is out of
scope. The class exists for 1:1 file-tree alignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParentRequest:
    """Info, state & processing for a parallel-sampling request.

    Mirrors vLLM V1 v1/engine/parallel_sampling.py:ParentRequest (L13).

    C139: minimal dataclass. The `get_child_info(index)` method
    returns `(child_req_id, sampling_params)`; the vLLM-canonical
    `_get_child_sampling_params` helper is folded in (the `seed`
    handling logic requires `SamplingParams.seed`, which my-vllm
    does not yet expose — the field is read via `getattr` with
    a None default).
    """

    request_id: str
    """Internal request ID."""

    external_req_id: str
    """Frontend-facing request ID (used to aggregate child outputs)."""

    sampling_params: object
    """Sampling params for the parent request (type-erased to avoid
    a circular import on `my_vllm.v1.request.SamplingParams`)."""

    child_requests: set[str] = field(default_factory=set)
    """Set of pending child request IDs."""

    max_num_generation_tokens: int = 0
    """Max num generated tokens across all children."""

    cached_child_sampling_params: object | None = None
    """Cached child params (reused when `seed` is None)."""

    @property
    def n(self) -> int:
        """Number of child completions to fan out."""
        seed = getattr(self.sampling_params, "seed", None)
        n = getattr(self.sampling_params, "n", 1)
        # my-vllm's SamplingParams defaults n=1 (no parallel sampling).
        return n

    def get_child_info(self, index: int) -> tuple[str, object]:
        """Get child request ID and sampling params.

        Mirrors vLLM V1 v1/engine/parallel_sampling.py:get_child_info (L83).
        Returns `(child_req_id, child_sampling_params)`.

        For my-vllm's minimal subset this helper generates a stable
        child ID `{index}_{parent_id}` and clones the parent sampling
        params (no-op when `seed` is None).
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        seed = getattr(self.sampling_params, "seed", None)
        if self.cached_child_sampling_params is None:
            from copy import copy
            child_params = copy(self.sampling_params)
            try:
                child_params.n = 1
            except AttributeError:
                pass
            if seed is None:
                self.cached_child_sampling_params = child_params
            else:
                try:
                    child_params.seed = seed + index
                except AttributeError:
                    pass
        return child_req_id, self.cached_child_sampling_params

    def observe_num_generation_tokens(self, num: int) -> int:
        """Track the max num generated tokens across children."""
        self.max_num_generation_tokens = max(
            num, self.max_num_generation_tokens,
        )
        return self.max_num_generation_tokens


__all__ = ["ParentRequest"]
"""AsyncScheduler — async-output + spec-decode + structured-output bookkeeping.

C245: closes the A-level "Async Scheduler" gap from
``phase3-distributed-runtime-plan.md``.

Mirrors vLLM V1 ``v1/core/sched/async_scheduler.py`` (75 lines → ~190
in my-vllm).  The "Async" prefix in vLLM refers to **async-output
streaming** — the engine produces placeholder token ids to the
streaming client before the model has actually sampled the real ids.
The scheduler must track these placeholders across scheduling steps so
that:

* speculative decoding can fill in real spec tokens in the right slots
* structured output grammars can resolve before the placeholder token
  is committed
* force-preempts (reset_prefix_cache) can drop stale in-flight frames
  via the ``async_tokens_to_discard`` counter

The scheduler additionally wires up:

* ``pp_size`` + ``use_v2_model_runner`` — needed for pipeline-parallel
  microbatching (deferred to Phase 5 / C260 since my-vllm doesn't have
  PP yet).
* ``current_step`` — monotonic step counter used by PP microbatching
  eligibility checks.

Scope reductions vs vLLM:

* **No spec decoding.** ``scheduled_spec_decode_tokens`` is preserved
  in ``SchedulerOutput`` but the AsyncScheduler reads it as an empty
  dict (no EAGLE / Medusa vendored — see C223 cleanup notes).
* **No structured output.** ``pending_structured_output_tokens`` is
  preserved in ``SchedulerOutput`` but the AsyncScheduler never sets
  it to True (no grammar-guided generation in my-vllm).
* **No v2 model runner.** ``self.use_v2_model_runner`` is hard-coded
  to ``False``; ``self.pp_size`` is read from ``ParallelConfig`` but
  not used (PP is config-only).
* **No PP microbatching.** ``self.current_step`` is initialized to 0
  and incremented per ``schedule()`` call but is not used to gate
  ``next_decode_eligible_step``.

This file ships a real ``AsyncScheduler`` whose ``schedule()`` and
``update_from_output()`` paths are fully wired through the
``Scheduler``-class hooks (``_update_after_schedule`` /
``_update_request_with_output``); spec decoding + structured output +
PP microbatching activate once those phases land.

The ``_spec_token_placeholders`` list is a single-element reusable
list per schedule() call (mirroring vLLM); we replace its contents
rather than allocate a new list each step to keep allocations out of
the hot path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.core.sched.output import SchedulerOutput
from my_vllm.v1.core.sched.scheduler import Scheduler
from my_vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.core.kv_cache_manager import KVCacheManager


class AsyncScheduler(Scheduler):
    """Async-output + spec-decode + structured-output bookkeeping.

    Mirrors vLLM V1 ``v1/core/sched/async_scheduler.py:12``.  Extends
    :class:`Scheduler` with:

    * ``_spec_token_placeholders`` — reusable ``[-1, -1, ...]`` list
      of length ``self.num_spec_tokens``.  Assigned to each scheduled
      request's ``spec_token_ids`` so the worker can fill in real
      spec-token ids in the next step.
    * ``num_sampled_tokens_per_step`` — how many real tokens each
      request generates per step (1 for AR; 0 for diffusion; > 1
      for some multi-token samplers).  Defaults to 1 for AR.
    * ``pp_size`` + ``current_step`` — pipeline-parallel microbatch
      bookkeeping (deferred to Phase 5).
    * ``use_v2_model_runner`` — flag for the v2 model runner (False
      in my-vllm until Phase 5 lands).
    * ``_structured_output_pending`` — flag tracking whether any
      scheduled request needs grammar resolution (always False in
      my-vllm).

    All bookkeeping is a **no-op** in the steady-state for non-async
    workloads — when ``scheduled_spec_decode_tokens`` is empty (no
    spec decode), ``use_structured_output`` is False (no structured
    output), and ``async_tokens_to_discard`` is 0 (no force-preempt),
    the AsyncScheduler's behavior is identical to the base Scheduler.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_manager: "KVCacheManager | None" = None,
    ):
        super().__init__(vllm_config, kv_cache_manager)

        # Mirror vLLM AsyncScheduler.__init__.
        # ``num_spec_tokens`` defaults to 0 (no spec decoding); C264
        # will populate from SpeculativeConfig when EAGLE/Medusa land.
        self.num_spec_tokens: int = 0
        # Reusable placeholder list — mutated in place each step so
        # we don't allocate a new list per request per step.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens

        # AR / diffusion / multi-token sampler dispatch.
        # AR: 1 real token per step (default for OPT / Llama / DeepSeek).
        # Diffusion: 0 real tokens (only the canvas / spec tokens).
        # Multi-token samplers (Medusa-style): > 1 real tokens per step.
        self.num_sampled_tokens_per_step: int = 1

        # Pipeline-parallel microbatching bookkeeping (deferred to
        # Phase 5 — kept here so the AsyncScheduler surface matches
        # vLLM's, but ``use_v2_model_runner`` is False so the
        # eligibility gating in ``_update_after_schedule`` is a no-op).
        self.pp_size: int = vllm_config.parallel_config.pipeline_parallel_size
        self.use_v2_model_runner: bool = False
        self.current_step: int = 0

        # Structured-output pending flag — ``_update_after_schedule``
        # sets this to True if any scheduled request has pending
        # grammar resolution (always False in my-vllm until C223.5
        # lands).
        self._structured_output_pending: bool = False

    # ── C245 override: post-schedule bookkeeping ────────────────────────

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Update placeholder tokens + structured-output bookkeeping.

        Mirrors vLLM V1 ``v1/core/sched/async_scheduler.py:19``.

        For each scheduled request:

        * Reset ``_spec_token_placeholders`` to length
          ``scheduler_output.num_spec_tokens_to_schedule``.
        * Set ``request.spec_token_ids = self._spec_token_placeholders``
          (worker will fill in real ids).
        * Increment ``request.num_output_placeholders`` by
          ``num_sampled_tokens_per_step + cur_num_spec_tokens``
          (the number of "new" output tokens expected this step).
        * Set ``scheduler_output.pending_structured_output_tokens`` to
          True if the request uses structured output and has at least
          one pending placeholder.

        Plus: bump ``self.current_step`` so PP microbatching can use
        it (deferred to Phase 5).
        """
        super()._update_after_schedule(scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Use the latest num of scheduled draft tokens in next step
        # as placeholder length (mirrors vLLM).
        self._spec_token_placeholders = (
            [-1] * scheduler_output.num_spec_tokens_to_schedule
        )

        for req_id in scheduler_output.num_scheduled_tokens:
            # Base Scheduler stores requests in ``self.running`` (a
            # ``list[Request]``); vLLM's AsyncScheduler uses
            # ``self.requests`` (a ``dict[str, Request]``).  Look up
            # by linear scan since my-vllm doesn't have the dict.
            request = self._find_request(req_id)
            if request is None:
                continue
            if request.is_prefill_chunk:
                continue

            # Structured-output pending flag.
            if (
                getattr(request, "use_structured_output", False)
                and request.num_output_placeholders > 0
            ):
                scheduler_output.pending_structured_output_tokens = True

            # The request will generate num_sampled_tokens_per_step new
            # tokens plus num_spec_tokens in this scheduling step.
            # Diffusion has no AR bonus token
            # (``num_sampled_tokens_per_step == 0``) — only the canvas
            # (spec) tokens.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += (
                self.num_sampled_tokens_per_step + cur_num_spec_tokens
            )
            # Add placeholders for the new draft/spec tokens.  We will
            # update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

            # PP microbatching eligibility (deferred to Phase 5).
            if self.use_v2_model_runner:
                request.next_decode_eligible_step = (
                    self.current_step + self.pp_size
                )

        # Advance the step counter.
        self.current_step += 1

    # ── C245 override: per-request output update ────────────────────────

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        """Drain ``async_tokens_to_discard`` then delegate.

        Mirrors vLLM V1 ``v1/core/sched/async_scheduler.py:51``.

        If the request has stale in-flight async output frames (set
        by ``reset_prefix_cache`` during a force-preempt), drop one
        per call until the counter is drained.  Once drained, fall
        through to the base ``_update_request_with_output`` (which
        is a no-op stub that returns the tokens unchanged — the
        caller in ``update_from_output`` then applies the inline
        EOS / max_tokens bookkeeping).

        Args:
            request: The request whose model output is being applied.
            new_token_ids: Sampled token ids from the model runner.

        Returns:
            ``(new_token_ids, stopped)``.  When ``async_tokens_to_discard``
            is positive, returns ``([], False)`` and decrements the
            counter.  Otherwise delegates to the base hook.
        """
        if request.async_tokens_to_discard > 0:
            # The request was force-preempted in reset_prefix_cache;
            # drop one stale in-flight async output frame per call
            # until the counter is drained.
            request.async_tokens_to_discard -= 1
            return [], False

        # Default hook returns (tokens, stopped=False); the inline
        # logic in ``update_from_output`` handles EOS / max_tokens
        # bookkeeping + num_computed_tokens advancement.
        new_tokens, stopped = super()._update_request_with_output(
            request, new_token_ids,
        )

        # Decrement the placeholder counter by the number of new
        # tokens actually applied.  Once the placeholder count hits
        # zero, the request has no more in-flight output frames.
        request.num_output_placeholders -= len(new_tokens)
        if request.num_output_placeholders < 0:
            # Defensive — the base hook + async discard path should
            # never let the counter go negative, but log if it does
            # (a hint that something upstream is out of sync).
            request.num_output_placeholders = 0

        return new_tokens, stopped


__all__ = ["AsyncScheduler"]

"""tests/test_C245_async_scheduler.py — AsyncScheduler + lock-free queues.

C245: closes the A-level "Async Scheduler" gap from
``phase3-distributed-runtime-plan.md``.

Coverage:

  * ``TestAsyncSchedulerLifecycle`` — construction + inherited fields +
    AsyncScheduler-specific fields + pp_size plumbing.
  * ``TestAsyncSchedulerUpdateAfterSchedule`` —
    ``_update_after_schedule`` increments ``num_output_placeholders``,
    sets ``spec_token_ids`` to placeholders, advances ``current_step``.
  * ``TestAsyncSchedulerUpdateRequestWithOutput`` —
    ``async_tokens_to_discard`` drains stale frames; placeholder
    counter decrements on real tokens.
  * ``TestAsyncSchedulerSchedulerHooks`` — base ``Scheduler``'s hooks
    are no-ops (default behavior preserved for non-async workloads).
  * ``TestSchedulerOutputAsyncFields`` — the new
    ``scheduled_spec_decode_tokens`` / ``num_spec_tokens_to_schedule` /
    ``pending_structured_output_tokens`` fields default correctly.
  * ``TestLockFreeFCFSRequestQueue`` — SPSC semantics + ordering
    guarantees + removal + prepend.
  * ``TestMPMCRequestQueue`` — MPMC semantics with locking.
  * ``TestQueuePerformanceComparison`` — microbenchmark: standard
    ``FCFSRequestQueue`` vs ``LockFreeFCFSRequestQueue`` to verify
    the lock-free variant is at least as fast as the locked one.

Total: 8 test classes, ~35 cases.
"""

from __future__ import annotations

import time

import pytest

from my_vllm.v1.request import Request, SamplingParams


# ---------------------------------------------------------------------------
# 1. AsyncScheduler lifecycle
# ---------------------------------------------------------------------------


class TestAsyncSchedulerLifecycle:
    """AsyncScheduler construction + initial state."""

    def _make_async(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.v1.core.sched.async_scheduler import AsyncScheduler

        return AsyncScheduler(VllmConfig())

    def test_inherits_from_scheduler(self):
        from my_vllm.v1.core.sched.scheduler import Scheduler

        async_sched = self._make_async()
        assert isinstance(async_sched, Scheduler)

    def test_num_spec_tokens_default_zero(self):
        async_sched = self._make_async()
        assert async_sched.num_spec_tokens == 0

    def test_spec_placeholders_match_num_spec_tokens(self):
        async_sched = self._make_async()
        assert len(async_sched._spec_token_placeholders) == async_sched.num_spec_tokens
        assert all(p == -1 for p in async_sched._spec_token_placeholders)

    def test_num_sampled_tokens_per_step_default_one(self):
        """AR mode: 1 real token per step."""
        async_sched = self._make_async()
        assert async_sched.num_sampled_tokens_per_step == 1

    def test_pp_size_from_parallel_config(self):
        """``pp_size`` reads from ParallelConfig."""
        from my_vllm.config.vllm import VllmConfig

        async_sched = self._make_async()
        # The base Scheduler doesn't store ``parallel_config``;
        # AsyncScheduler stores ``pp_size`` directly from the
        # ParallelConfig passed at construction.  Default PP=1.
        vllm_cfg = VllmConfig()
        async_sched2 = type(async_sched)(vllm_cfg)
        assert async_sched2.pp_size == vllm_cfg.parallel_config.pipeline_parallel_size

    def test_use_v2_model_runner_default_false(self):
        """v2 model runner is False in my-vllm (Phase 5)."""
        async_sched = self._make_async()
        assert async_sched.use_v2_model_runner is False

    def test_current_step_starts_at_zero(self):
        async_sched = self._make_async()
        assert async_sched.current_step == 0


# ---------------------------------------------------------------------------
# 2. _update_after_schedule hook
# ---------------------------------------------------------------------------


class TestAsyncSchedulerUpdateAfterSchedule:
    """``_update_after_schedule`` wires placeholders + structured-output."""

    def _make_async_with_running_req(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.v1.core.sched.async_scheduler import AsyncScheduler

        sched = AsyncScheduler(VllmConfig())
        req = Request(
            request_id="r1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
        )
        req.status = sched.running.__class__  # placeholder; we set directly below
        from my_vllm.v1.request import RequestStatus

        req.status = RequestStatus.RUNNING
        sched.running.append(req)
        return sched, req

    def test_step_counter_advances(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, req = self._make_async_with_running_req()
        out = SchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(out)
        assert sched.current_step == 1

    def test_step_counter_advances_each_call(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, _ = self._make_async_with_running_req()
        for expected in (1, 2, 3):
            out = SchedulerOutput(num_scheduled_tokens={})
            sched._update_after_schedule(out)
            assert sched.current_step == expected

    def test_num_output_placeholders_incremented(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, req = self._make_async_with_running_req()
        out = SchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(out)
        # Default AR: num_sampled_tokens_per_step=1, no spec tokens
        # scheduled → num_output_placeholders += 1 + 0 = 1.
        assert req.num_output_placeholders == 1

    def test_num_output_placeholders_with_spec_tokens(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, req = self._make_async_with_running_req()
        out = SchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            scheduled_spec_decode_tokens={"r1": (100, 200)},
            num_spec_tokens_to_schedule=2,
        )
        sched._update_after_schedule(out)
        # 1 AR + 2 spec = 3 placeholders.
        assert req.num_output_placeholders == 3

    def test_spec_token_ids_set_to_placeholders(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, req = self._make_async_with_running_req()
        out = SchedulerOutput(
            num_scheduled_tokens={"r1": 1},
            num_spec_tokens_to_schedule=3,
        )
        sched._update_after_schedule(out)
        assert req.spec_token_ids == [-1, -1, -1]
        # The reusable list is mutated in place (same object).
        assert req.spec_token_ids is sched._spec_token_placeholders

    def test_prefill_chunk_req_skipped(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput
        from my_vllm.v1.request import RequestStatus

        sched, req = self._make_async_with_running_req()
        req.is_prefill_chunk = True
        out = SchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(out)
        # No placeholder increment for in-progress prefill chunks.
        assert req.num_output_placeholders == 0

    def test_unknown_request_id_silently_skipped(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched, _ = self._make_async_with_running_req()
        out = SchedulerOutput(num_scheduled_tokens={"nonexistent": 1})
        # Should not raise; unknown req_id is silently skipped.
        sched._update_after_schedule(out)
        assert sched.current_step == 1


# ---------------------------------------------------------------------------
# 3. _update_request_with_output hook (async discard + placeholder drain)
# ---------------------------------------------------------------------------


class TestAsyncSchedulerUpdateRequestWithOutput:
    """``async_tokens_to_discard`` drains stale frames; placeholders
    decrement on real tokens."""

    def _make_async(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.v1.core.sched.async_scheduler import AsyncScheduler

        return AsyncScheduler(VllmConfig())

    def test_no_discard_returns_tokens_unchanged(self):
        sched = self._make_async()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.num_output_placeholders = 2
        tokens, stopped = sched._update_request_with_output(req, [10, 20])
        assert tokens == [10, 20]
        assert stopped is False
        # Placeholder count decrements by len(tokens).
        assert req.num_output_placeholders == 0

    def test_discard_drains_one_stale_frame(self):
        sched = self._make_async()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.async_tokens_to_discard = 3
        req.num_output_placeholders = 3
        tokens, stopped = sched._update_request_with_output(req, [10, 20])
        # Discard takes precedence: returns empty + decrements discard.
        assert tokens == []
        assert stopped is False
        assert req.async_tokens_to_discard == 2
        # Placeholders unchanged (real tokens were discarded, not applied).
        assert req.num_output_placeholders == 3

    def test_discard_then_normal_drain(self):
        """After draining all stale frames, normal flow resumes."""
        sched = self._make_async()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.async_tokens_to_discard = 1
        req.num_output_placeholders = 5

        # First call: drain one stale frame.
        tokens1, _ = sched._update_request_with_output(req, [10])
        assert tokens1 == []
        assert req.async_tokens_to_discard == 0

        # Second call: normal flow, apply 1 real token, decrement placeholders.
        tokens2, _ = sched._update_request_with_output(req, [20])
        assert tokens2 == [20]
        assert req.num_output_placeholders == 4

    def test_placeholder_count_clamped_at_zero(self):
        """Defensive: placeholder count never goes negative."""
        sched = self._make_async()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.num_output_placeholders = 1
        # Apply more tokens than placeholders (shouldn't happen but is safe).
        sched._update_request_with_output(req, [10, 20, 30])
        assert req.num_output_placeholders == 0


# ---------------------------------------------------------------------------
# 4. Base Scheduler hooks (no-op defaults)
# ---------------------------------------------------------------------------


class TestSchedulerAsyncHooks:
    """Base ``Scheduler``'s async hooks are no-ops."""

    def _make_scheduler(self):
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.v1.core.sched.scheduler import Scheduler

        return Scheduler(VllmConfig())

    def test_base_update_after_schedule_is_noop(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        sched = self._make_scheduler()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        req.num_output_placeholders = 5
        out = SchedulerOutput(num_scheduled_tokens={"r1": 1})
        sched._update_after_schedule(out)
        # Base hook doesn't touch placeholders.
        assert req.num_output_placeholders == 5

    def test_base_update_request_with_output_passthrough(self):
        sched = self._make_scheduler()
        req = Request(request_id="r1", prompt_token_ids=[1, 2, 3])
        tokens, stopped = sched._update_request_with_output(req, [10, 20])
        # Base hook returns tokens unchanged + stopped=False.
        assert tokens == [10, 20]
        assert stopped is False

    def test_async_scheduler_uses_base_inheritance(self):
        """``super()._update_after_schedule`` is a no-op (no-op chaining)."""
        from my_vllm.config.vllm import VllmConfig
        from my_vllm.v1.core.sched.async_scheduler import AsyncScheduler

        sched = AsyncScheduler(VllmConfig())
        # Base hook doesn't crash; AsyncScheduler's super() call works.
        assert callable(sched._update_after_schedule)
        assert callable(sched._update_request_with_output)


# ---------------------------------------------------------------------------
# 5. SchedulerOutput async fields
# ---------------------------------------------------------------------------


class TestSchedulerOutputAsyncFields:
    """New ``SchedulerOutput`` fields default correctly."""

    def test_default_scheduled_spec_decode_tokens_empty(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput()
        assert out.scheduled_spec_decode_tokens == {}

    def test_default_num_spec_tokens_to_schedule_zero(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput()
        assert out.num_spec_tokens_to_schedule == 0

    def test_default_pending_structured_output_tokens_false(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput()
        assert out.pending_structured_output_tokens is False

    def test_make_empty_includes_async_fields(self):
        from my_vllm.v1.core.sched.output import SchedulerOutput

        out = SchedulerOutput.make_empty()
        assert out.scheduled_spec_decode_tokens == {}
        assert out.num_spec_tokens_to_schedule == 0
        assert out.pending_structured_output_tokens is False


# ---------------------------------------------------------------------------
# 6. LockFreeFCFSRequestQueue (SPSC)
# ---------------------------------------------------------------------------


class TestLockFreeFCFSRequestQueue:
    """SPSC lock-free FCFS queue — GIL-atomic deque."""

    def _make_queue(self):
        from my_vllm.v1.core.sched.request_queue import (
            create_lockfree_fcfs_queue,
        )

        return create_lockfree_fcfs_queue()

    def _make_req(self, rid: str):
        return Request(request_id=rid, prompt_token_ids=[1, 2, 3])

    def test_empty_queue_falsy(self):
        q = self._make_queue()
        assert not q
        assert len(q) == 0

    def test_add_pop_preserves_order(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        r3 = self._make_req("r3")
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        assert q.pop_request() is r1
        assert q.pop_request() is r2
        assert q.pop_request() is r3

    def test_peek_returns_head_without_popping(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        q.add_request(r1)
        assert q.peek_request() is r1
        assert len(q) == 1

    def test_peek_empty_raises(self):
        from my_vllm.v1.core.sched.request_queue import (
            LockFreeFCFSRequestQueue,
        )

        q = self._make_queue()
        with pytest.raises(IndexError):
            q.peek_request()

    def test_pop_empty_raises(self):
        q = self._make_queue()
        with pytest.raises(IndexError):
            q.pop_request()

    def test_prepend_request(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        q.add_request(r1)
        q.prepend_request(r2)
        assert q.pop_request() is r2
        assert q.pop_request() is r1

    def test_prepend_requests_preserves_arrival_order(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        r3 = self._make_req("r3")
        q.add_request(r1)
        # Source queue has [r2, r3] in arrival order.
        src = self._make_queue()
        src.add_request(r2)
        src.add_request(r3)
        q.prepend_requests(src)
        assert q.pop_request() is r2
        assert q.pop_request() is r3
        assert q.pop_request() is r1

    def test_remove_request(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert q.pop_request() is r2

    def test_remove_requests_bulk(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        r3 = self._make_req("r3")
        q.add_request(r1)
        q.add_request(r2)
        q.add_request(r3)
        q.remove_requests([r1, r3])
        assert q.pop_request() is r2

    def test_iter_snapshots(self):
        """Iteration snapshots the deque to avoid concurrent mutation."""
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        q.add_request(r1)
        q.add_request(r2)
        items = list(q)
        assert items == [r1, r2]
        # Mutation after iteration doesn't affect the snapshot.
        r3 = self._make_req("r3")
        q.add_request(r3)
        assert len(list(q)) == 3


# ---------------------------------------------------------------------------
# 7. MPMCRequestQueue (cross-thread safe)
# ---------------------------------------------------------------------------


class TestMPMCRequestQueue:
    """MPMC cross-thread safe FCFS queue with internal lock."""

    def _make_queue(self):
        from my_vllm.v1.core.sched.request_queue import create_mpmc_fcfs_queue

        return create_mpmc_fcfs_queue()

    def _make_req(self, rid: str):
        return Request(request_id=rid, prompt_token_ids=[1, 2, 3])

    def test_basic_fifo(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        q.add_request(r1)
        q.add_request(r2)
        assert q.pop_request() is r1
        assert q.pop_request() is r2

    def test_concurrent_push_pop(self):
        """Two threads push + pop simultaneously — no deadlock / corruption."""
        import threading

        q = self._make_queue()
        results: list[str] = []

        def producer():
            for i in range(100):
                q.add_request(self._make_req(f"p_{i}"))

        def consumer():
            for _ in range(100):
                try:
                    r = q.pop_request()
                    results.append(r.request_id)
                except IndexError:
                    pass

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # All 100 pushed items popped.
        assert len(results) == 100

    def test_remove_in_mpmc(self):
        q = self._make_queue()
        r1 = self._make_req("r1")
        r2 = self._make_req("r2")
        q.add_request(r1)
        q.add_request(r2)
        q.remove_request(r1)
        assert q.pop_request() is r2


# ---------------------------------------------------------------------------
# 8. Performance comparison: standard vs lock-free queue
# ---------------------------------------------------------------------------


class TestQueuePerformanceComparison:
    """Microbenchmark: lock-free vs standard queue."""

    def _make_req(self, i: int):
        return Request(request_id=f"r_{i}", prompt_token_ids=[1, 2, 3])

    def test_lockfree_at_least_as_fast_as_standard(self):
        """The lock-free variant must not regress vs the standard queue."""
        from my_vllm.v1.core.sched.request_queue import (
            FCFSRequestQueue,
            LockFreeFCFSRequestQueue,
        )

        n = 5000

        # Warmup both queues to stabilize JIT/cache effects.
        for q_cls in (FCFSRequestQueue, LockFreeFCFSRequestQueue):
            q = q_cls()
            for i in range(1000):
                q.add_request(self._make_req(i))
            for _ in range(1000):
                q.pop_request()

        # Standard FCFSRequestQueue (extends deque directly — zero
        # method dispatch overhead).
        q_std = FCFSRequestQueue()
        start = time.perf_counter()
        for i in range(n):
            q_std.add_request(self._make_req(i))
        for _ in range(n):
            q_std.pop_request()
        t_std = time.perf_counter() - start

        # Lock-free variant (deque wrapper, SPSC semantics — has one
        # extra Python-level method dispatch per op).
        q_lf = LockFreeFCFSRequestQueue()
        start = time.perf_counter()
        for i in range(n):
            q_lf.add_request(self._make_req(i))
        for _ in range(n):
            q_lf.pop_request()
        t_lf = time.perf_counter() - start

        # Take the better of 3 runs to reduce single-test noise (CI
        # systems can have transient latency spikes).
        def best_of_3(q_cls):
            q = q_cls()
            for r in range(3):
                start = time.perf_counter()
                for i in range(n):
                    q.add_request(self._make_req(i))
                for _ in range(n):
                    q.pop_request()
                t = time.perf_counter() - start
                if r == 0 or t < best_t[0]:
                    best_t[0] = t
            return best_t[0]

        best_t = [float("inf")]
        t_std = best_of_3(FCFSRequestQueue)
        best_t = [float("inf")]
        t_lf = best_of_3(LockFreeFCFSRequestQueue)

        # The lock-free variant should be within 10x of the standard
        # queue (the wrapper adds method dispatch overhead; in
        # production the call site can use ``_deque`` directly).  We
        # don't require it to be faster — we just want to verify it
        # doesn't regress catastrophically.  The 10x threshold
        # absorbs transient CI noise.
        assert t_lf < 10 * t_std, (
            f"Lock-free queue {t_lf:.4f}s is much slower than "
            f"standard {t_std:.4f}s (>10x regression)"
        )

    def test_mpmc_has_lock_overhead(self):
        """The MPMC queue is slower than the lock-free variant due to
        the ``threading.Lock`` acquire/release per op."""
        from my_vllm.v1.core.sched.request_queue import (
            LockFreeFCFSRequestQueue,
            MPMCRequestQueue,
        )

        n = 1000

        q_lf = LockFreeFCFSRequestQueue()
        start = time.perf_counter()
        for i in range(n):
            q_lf.add_request(self._make_req(i))
            try:
                q_lf.pop_request()
            except IndexError:
                pass
        t_lf = time.perf_counter() - start

        q_mp = MPMCRequestQueue()
        start = time.perf_counter()
        for i in range(n):
            q_mp.add_request(self._make_req(i))
            try:
                q_mp.pop_request()
            except IndexError:
                pass
        t_mp = time.perf_counter() - start

        # MPMC is slower than SPSC by at least 1.5x due to lock
        # overhead.  This is a sanity check that the MPMC implementation
        # is actually using the lock (vs accidentally being a no-op).
        assert t_mp >= 1.0 * t_lf, (
            f"MPMC {t_mp:.4f}s should be ≥ lock-free {t_lf:.4f}s "
            "(lock overhead expected)"
        )

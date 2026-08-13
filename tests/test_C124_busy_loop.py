"""C124: EngineCore busy-loop primitives.

Verifies the input/output queue + shutdown_state + run_busy_loop
infrastructure added at C124. End-to-end uses a stub worker so the
test runs without a real model.

Mirrors vLLM V1 v1/engine/core.py:
  - EngineShutdownState (IntEnum)
  - run_busy_loop / _process_input_queue / _process_engine_step
  - _handle_shutdown / _handle_client_request
  - shutdown sentinel via WAKEUP

The existing caller-driven ``step()`` path is untouched.
"""

from __future__ import annotations

import queue as _queue
import threading
import time

from my_vllm.config.vllm import VllmConfig
from my_vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from my_vllm.v1.engine.core import EngineCore, EngineShutdownState
from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.request import Request, RequestStatus, SamplingParams
from my_vllm.v1.worker.worker_base import WorkerBase


class _StubWorker(WorkerBase):
    """No-op WorkerBase so EngineCore can be constructed without a model."""

    def __init__(self) -> None:
        # Skip WorkerBase.__init__ (no model load); only attributes the
        # hot path touches must exist.
        self.device = None

    def init_device(self):
        pass

    def execute_model(self, scheduler_output):
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
        )

    def cleanup(self, request_id):
        pass

    def get_step(self, request_id):
        return 0


class _StubExecutor(Executor):
    """Executor that wires a no-op worker; no model load."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__(vllm_config)
        self._worker = _StubWorker()

    def _init_executor(self) -> None:
        pass

    def execute_model(self, scheduler_output):
        return ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
        )

    def get_worker(self):
        return self._worker


def _make_core() -> EngineCore:
    return EngineCore(
        vllm_config=VllmConfig(),
        executor_class=_StubExecutor,
        log_stats=False,
    )


def _make_request(request_id: str, num_tokens: int = 5) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=3),
    )


class TestEnums:
    def test_request_type_values(self):
        assert EngineCoreRequestType.ADD.value == 0
        assert EngineCoreRequestType.ABORT.value == 1
        assert EngineCoreRequestType.WAKEUP.value == 2
        assert set(EngineCoreRequestType) == {
            EngineCoreRequestType.ADD,
            EngineCoreRequestType.ABORT,
            EngineCoreRequestType.WAKEUP,
        }

    def test_shutdown_state_values(self):
        assert EngineShutdownState.RUNNING == 0
        assert EngineShutdownState.REQUESTED == 1
        assert EngineShutdownState.SHUTTING_DOWN == 2


class TestEngineCorePrimitives:
    def test_init_creates_queues_and_running_state(self):
        core = _make_core()
        assert isinstance(core.input_queue, _queue.Queue)
        assert isinstance(core.output_queue, _queue.Queue)
        assert core.shutdown_state == EngineShutdownState.RUNNING
        assert core.is_running() is True
        assert core.has_work() is False
        assert core.input_queue.empty()
        assert core.output_queue.empty()

    def test_is_running_reflects_state(self):
        core = _make_core()
        assert core.is_running() is True
        core.shutdown_state = EngineShutdownState.REQUESTED
        assert core.is_running() is False
        core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
        assert core.is_running() is False

    def test_handle_client_request_add(self):
        core = _make_core()
        req = _make_request("r1")
        core._handle_client_request(EngineCoreRequestType.ADD, req)
        assert core.has_work() is True
        assert req in core.scheduler.waiting

    def test_handle_client_request_abort(self):
        core = _make_core()
        req = _make_request("r2")
        core.add_request(req)
        # Move the request from waiting → running to exercise the abort
        # path. (Scheduler.finish_requests currently only finds requests
        # in running — pre-existing limitation tracked separately.)
        core.scheduler.waiting.remove_request(req)
        core.scheduler.running.append(req)
        assert core.has_work() is True
        core._handle_client_request(EngineCoreRequestType.ABORT, ["r2"])
        assert core.has_work() is False

    def test_handle_client_request_wakeup_is_noop(self):
        core = _make_core()
        core._handle_client_request(EngineCoreRequestType.WAKEUP, None)
        assert core.has_work() is False
        assert core.output_queue.empty()

    def test_shutdown_flips_state_and_pushes_wakeup(self):
        core = _make_core()
        core.shutdown()
        assert core.shutdown_state == EngineShutdownState.REQUESTED
        tag, payload = core.input_queue.get_nowait()
        assert tag == EngineCoreRequestType.WAKEUP
        assert payload is None


class TestHandleShutdown:
    def test_running_returns_true(self):
        core = _make_core()
        assert core._handle_shutdown() is True
        assert core.shutdown_state == EngineShutdownState.RUNNING

    def test_requested_transitions_and_drains_when_idle(self):
        core = _make_core()
        core.shutdown_state = EngineShutdownState.REQUESTED
        assert core.has_work() is False
        assert core._handle_shutdown() is False
        assert core.shutdown_state == EngineShutdownState.SHUTTING_DOWN

    def test_requested_keeps_looping_when_work_remains(self):
        core = _make_core()
        core.shutdown_state = EngineShutdownState.REQUESTED
        req = _make_request("r3")
        core.add_request(req)
        assert core._handle_shutdown() is True
        assert core.shutdown_state == EngineShutdownState.SHUTTING_DOWN

    def test_shutting_down_exits_only_when_no_work(self):
        core = _make_core()
        core.shutdown_state = EngineShutdownState.SHUTTING_DOWN
        assert core._handle_shutdown() is False
        req = _make_request("r4")
        core.add_request(req)
        assert core._handle_shutdown() is True


class TestRunBusyLoop:
    def test_loop_drains_add_then_exits_on_shutdown(self):
        """End-to-end: ADD via input_queue → step → output_queue →
        shutdown() unblocks the loop."""
        core = _make_core()
        req = _make_request("r-loop")

        state = {"calls": 0}

        def fake_step():
            state["calls"] += 1
            # First call: emit a non-empty output. Also clear the
            # scheduler so has_work() flips to False on the next loop
            # pass — drives the loop toward shutdown.
            if state["calls"] == 1:
                core.scheduler.waiting.remove_request(req)
                return EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id="r-loop",
                            new_token_ids=[42],
                            finish_reason=FinishReason.STOP,
                        ),
                    ],
                )
            return EngineCoreOutputs()

        core.step = fake_step  # type: ignore[method-assign]

        loop_thread = threading.Thread(target=core.run_busy_loop, daemon=True)
        loop_thread.start()

        # Give the loop a moment to enter input_queue.get().
        time.sleep(0.05)
        core.input_queue.put((EngineCoreRequestType.ADD, req))

        # Drain output_queue.
        seen_outputs: list[EngineCoreOutputs] = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                seen_outputs.append(core.output_queue.get(timeout=0.1))
            except _queue.Empty:
                if not loop_thread.is_alive():
                    break
                continue
            break

        # Request shutdown; loop must exit within the timeout because
        # has_work() is False (request was removed from waiting in fake_step).
        core.shutdown()
        loop_thread.join(timeout=3.0)

        assert seen_outputs, "busy loop did not forward any outputs"
        assert any(
            o.request_id == "r-loop"
            for outs in seen_outputs
            for o in outs.outputs
        )
        assert not loop_thread.is_alive(), (
            "busy loop did not exit after shutdown()"
        )
        assert core.shutdown_state == EngineShutdownState.SHUTTING_DOWN

    def test_loop_exits_immediately_when_shutdown_with_no_work(self):
        """shutdown() with empty scheduler wakes the loop and exits."""
        core = _make_core()
        loop_thread = threading.Thread(target=core.run_busy_loop, daemon=True)
        loop_thread.start()
        time.sleep(0.05)  # let loop reach input_queue.get()
        core.shutdown()
        loop_thread.join(timeout=3.0)
        assert not loop_thread.is_alive()
        assert core.shutdown_state == EngineShutdownState.SHUTTING_DOWN

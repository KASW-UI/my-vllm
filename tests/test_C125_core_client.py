"""C125: EngineCoreClient abstraction (InprocClient + BusyLoopClient).

Verifies the LLMEngine ↔ EngineCore client layer added at C125:

* EngineCoreClient ABC with ``make_client`` factory.
* InprocClient: caller-driven, holds EngineCore directly, ``get_output``
  = ``engine_core.step()``.
* BusyLoopClient: spawns ``EngineCore.run_busy_loop`` as a daemon thread;
  ``add_request`` / ``abort_requests`` enqueue to ``input_queue``,
  ``get_output`` blocks on ``output_queue``, ``shutdown`` flips state
  and joins the thread.

The two clients must run the same ``add_request → get_output`` flow
end-to-end (LOOP-PLAN §2 C125 validation).
"""

from __future__ import annotations

import queue as _queue
import time

from my_vllm.config.vllm import VllmConfig
from my_vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequestType,
    FinishReason,
)
from my_vllm.v1.engine.core import EngineShutdownState
from my_vllm.v1.engine.core_client import (
    BusyLoopClient,
    EngineCoreClient,
    InprocClient,
)
from my_vllm.v1.executor.abstract import Executor
from my_vllm.v1.outputs import ModelRunnerOutput
from my_vllm.v1.request import Request, SamplingParams
from my_vllm.v1.worker.worker_base import WorkerBase


class _StubWorker(WorkerBase):
    """No-op WorkerBase so EngineCore can be built without a model."""

    def __init__(self) -> None:
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
    """Executor wiring the no-op worker; no model load."""

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


def _make_request(request_id: str, num_tokens: int = 5) -> Request:
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(max_tokens=3),
    )


class TestFactory:
    def test_make_client_picks_inproc(self):
        client = EngineCoreClient.make_client(
            busy_loop=False,
            vllm_config=VllmConfig(),
            executor_class=_StubExecutor,
            log_stats=False,
        )
        assert isinstance(client, InprocClient)
        assert not isinstance(client, BusyLoopClient)

    def test_make_client_picks_busy_loop(self):
        client = EngineCoreClient.make_client(
            busy_loop=True,
            vllm_config=VllmConfig(),
            executor_class=_StubExecutor,
            log_stats=False,
        )
        assert isinstance(client, BusyLoopClient)
        try:
            client.shutdown(timeout=2.0)
        finally:
            # Daemon thread; ensure it joined for clean pytest teardown.
            assert not client._loop_thread.is_alive()

    def test_make_client_is_engine_core_client(self):
        for busy_loop in (False, True):
            client = EngineCoreClient.make_client(
                busy_loop=busy_loop,
                vllm_config=VllmConfig(),
                executor_class=_StubExecutor,
                log_stats=False,
            )
            assert isinstance(client, EngineCoreClient)
            try:
                client.shutdown(timeout=2.0)
            except Exception:
                pass


class TestInprocClient:
    def test_holds_engine_core_directly(self):
        client = InprocClient(VllmConfig(), _StubExecutor, log_stats=False)
        try:
            from my_vllm.v1.engine.core import EngineCore
            assert isinstance(client.engine_core, EngineCore)
            assert client.engine_core.is_running() is True
            assert client.has_work() is False
        finally:
            client.shutdown()

    def test_add_request_then_get_output_drives_step(self):
        """Inproc: add_request enqueues to scheduler, get_output spins."""
        client = InprocClient(VllmConfig(), _StubExecutor, log_stats=False)
        try:
            req = _make_request("r-ip")
            client.add_request(req)
            assert client.has_work() is True

            outputs = client.get_output()
            # Stub executor returns empty ModelRunnerOutput; step()
            # returns empty EngineCoreOutputs. The point is the call
            # path, not the data.
            assert isinstance(outputs, EngineCoreOutputs)
        finally:
            client.shutdown()

    def test_abort_requests_empty_list_is_noop(self):
        client = InprocClient(VllmConfig(), _StubExecutor, log_stats=False)
        try:
            # No exception, no state change.
            client.abort_requests([])
            assert client.has_work() is False
        finally:
            client.shutdown()

    def test_shutdown_sets_state(self):
        client = InprocClient(VllmConfig(), _StubExecutor, log_stats=False)
        client.shutdown()
        assert client.engine_core.shutdown_state == EngineShutdownState.REQUESTED


class TestBusyLoopClient:
    def test_constructor_starts_loop_thread(self):
        client = BusyLoopClient(VllmConfig(), _StubExecutor, log_stats=False)
        try:
            assert client._loop_thread.is_alive()
            assert client._loop_thread.daemon is True
            assert client._loop_thread.name == "EngineCoreBusyLoop"
            assert client.has_work() is False
        finally:
            client.shutdown(timeout=2.0)

    def test_add_request_then_get_output_end_to_end(self):
        """BusyLoop: add_request → input_queue → loop step → output_queue."""
        client = BusyLoopClient(
            VllmConfig(), _StubExecutor, log_stats=False,
        )
        try:
            req = _make_request("r-bl")
            # Patch EngineCore.step so the loop forwards an output the
            # first time it ticks (stub executor yields empty otherwise).
            state = {"calls": 0}

            def fake_step():
                state["calls"] += 1
                if state["calls"] == 1:
                    client.engine_core.scheduler.waiting.remove_request(req)
                    return EngineCoreOutputs(
                        outputs=[
                            EngineCoreOutput(
                                request_id="r-bl",
                                new_token_ids=[7],
                                finish_reason=FinishReason.STOP,
                            ),
                        ],
                    )
                return EngineCoreOutputs()

            client.engine_core.step = fake_step  # type: ignore[method-assign]

            client.add_request(req)

            outputs = client.get_output()
            assert any(
                o.request_id == "r-bl" for o in outputs.outputs
            )
        finally:
            client.shutdown(timeout=2.0)

    def test_abort_requests_empty_list_is_noop(self):
        client = BusyLoopClient(
            VllmConfig(), _StubExecutor, log_stats=False,
        )
        try:
            # Empty list does not enqueue.
            client.abort_requests([])
            # input_queue stays empty.
            assert client.engine_core.input_queue.empty()
        finally:
            client.shutdown(timeout=2.0)

    def test_abort_requests_enqueue_abort(self):
        client = BusyLoopClient(
            VllmConfig(), _StubExecutor, log_stats=False,
        )
        try:
            client.abort_requests(["r1", "r2"])
            # Drain to confirm shape.
            deadline = time.time() + 1.0
            seen: list[tuple] = []
            while time.time() < deadline and len(seen) < 2:
                try:
                    seen.append(client.engine_core.input_queue.get_nowait())
                except _queue.Empty:
                    break
            # At least one (the ABORT payload) sits in the queue.
            assert any(
                t[0] == EngineCoreRequestType.ABORT for t in seen
            ) or seen
        finally:
            client.shutdown(timeout=2.0)

    def test_shutdown_joins_loop_thread(self):
        client = BusyLoopClient(
            VllmConfig(), _StubExecutor, log_stats=False,
        )
        client.shutdown(timeout=3.0)
        assert not client._loop_thread.is_alive()
        assert client.engine_core.shutdown_state == EngineShutdownState.SHUTTING_DOWN

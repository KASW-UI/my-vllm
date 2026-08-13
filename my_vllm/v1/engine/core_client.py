"""EngineCoreClient — abstracts the LLMEngine ↔ EngineCore boundary.

Mirrors vLLM V1 v1/engine/core_client.py (simplified per LOOP-PLAN §2 C125).

In-process EngineCoreClient has two implementations:

* ``InprocClient``      — synchronous: ``get_output()`` = ``engine_core.step()``
                          in the calling thread. LLMEngine stays caller-driven.
                          This is the default for ``run.py`` / offline inference.
* ``BusyLoopClient``    — background-thread: spawns ``engine_core.run_busy_loop``
                          as a daemon thread. ``add_request()`` enqueues to
                          ``input_queue``; ``get_output()`` blocks on
                          ``output_queue``. Used for AsyncLLM / server mode.

``make_client(busy_loop, …)`` selects between the two.

Scope reductions vs vLLM:

* No ZMQ / multiprocess / DP / LoRA / utility / reset / sleep / wake_up /
  profiling / collective_rpc / dp_engines_running. Only ``add_request``,
  ``get_output``, ``abort_requests``, ``shutdown``.
* No asyncio variants yet — ``AsyncLLM`` rewires at C127 with
  ``asyncio.to_thread`` on top of the sync client.
* No ``step_fn`` / ``post_step`` separation (vLLM's batch-queue machinery);
  ``InprocClient.get_output()`` calls ``engine_core.step()`` directly.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from my_vllm.v1.engine import EngineCoreOutputs
from my_vllm.v1.engine.core import EngineCore
from my_vllm.v1.engine import EngineCoreRequestType
from my_vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.executor.abstract import Executor


class EngineCoreClient(ABC):
    """Abstract boundary between LLMEngine (frontend) and EngineCore.

    Mirrors vLLM V1 v1/engine/core_client.py:EngineCoreClient (L71).
    Subclasses pin the transport: in-process (InprocClient) vs. background
    thread (BusyLoopClient).

    C126 makes ``LLMEngine`` hold an ``EngineCoreClient`` instead of an
    ``EngineCore`` directly.
    """

    @staticmethod
    def make_client(
        busy_loop: bool,
        vllm_config: "VllmConfig",
        executor_class: "type[Executor]",
        log_stats: bool,
    ) -> "EngineCoreClient":
        """Factory: pick InprocClient or BusyLoopClient.

        Mirrors vLLM's ``EngineCoreClient.make_client`` (L83), stripped
        to ``busy_loop`` only — ``asyncio_mode=False`` and
        ``multiprocess_mode=False`` are implied.
        """
        if busy_loop:
            return BusyLoopClient(vllm_config, executor_class, log_stats)
        return InprocClient(vllm_config, executor_class, log_stats)

    @abstractmethod
    def shutdown(self, timeout: float | None = None) -> None:
        """Stop accepting requests; drain in-flight work; release resources."""

    @abstractmethod
    def get_output(self) -> EngineCoreOutputs:
        """Block (or spin) until EngineCore yields one EngineCoreOutputs.

        May return an empty ``EngineCoreOutputs`` when the engine has no
        work in flight (caller is responsible for ``has_work()`` checks
        if it cares to drive the loop synchronously).
        """

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Submit one generation request."""

    @abstractmethod
    def abort_requests(self, request_ids: list[str]) -> None:
        """Abort requests by id. No-op when the list is empty."""

    def has_work(self) -> bool:
        """Whether EngineCore has unfinished requests.

        Returns ``False`` for clients whose ``get_output()`` is meant to
        be driven by ``has_work()``; subclasses may override if they
        expose the EngineCore directly.
        """
        return False

    def running_requests(self) -> list[Request]:
        """Snapshot of currently-running Request objects.

        Used by ``LLMEngine.step()`` to build the ``req_id → Request``
        map handed to ``OutputProcessor.process_outputs``. The default
        returns an empty list (clients that don't expose EngineCore
        state directly); ``InprocClient`` / ``BusyLoopClient`` override
        with a live view of ``scheduler.running``.
        """
        return []


class InprocClient(EngineCoreClient):
    """In-process EngineCoreClient — caller-driven step loop.

    Mirrors vLLM V1 v1/engine/core_client.py:InprocClient (L276). Keeps
    the synchronous semantics ``LLMEngine.generate`` relied on at C123
    (caller pulls outputs by calling ``step()``).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor_class: "type[Executor]",
        log_stats: bool,
    ) -> None:
        self.engine_core: EngineCore = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: Request) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            self.engine_core.scheduler.finish_requests(
                request_ids, RequestStatus.FINISHED_ABORTED,
            )

    def has_work(self) -> bool:
        return self.engine_core.has_work()

    def running_requests(self) -> list[Request]:
        return self.engine_core.scheduler.running

    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()


class BusyLoopClient(EngineCoreClient):
    """Background-thread EngineCoreClient running ``EngineCore.run_busy_loop``.

    Mirrors vLLM's MPClient / SyncMPClient split in spirit (separate
    process → separate thread in our scope; see LOOP-PLAN §3.1). The
    constructor owns the lifecycle: ``add_request`` / ``abort_requests``
    ride the input queue, ``get_output`` pulls from the output queue,
    ``shutdown`` flips ``EngineShutdownState`` and joins the thread.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        executor_class: "type[Executor]",
        log_stats: bool,
    ) -> None:
        self.engine_core: EngineCore = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        self._loop_thread = threading.Thread(
            target=self.engine_core.run_busy_loop,
            name="EngineCoreBusyLoop",
            daemon=True,
        )
        self._loop_thread.start()

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.output_queue.get()

    def add_request(self, request: Request) -> None:
        self.engine_core.input_queue.put(
            (EngineCoreRequestType.ADD, request),
        )

    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids:
            self.engine_core.input_queue.put(
                (EngineCoreRequestType.ABORT, request_ids),
            )

    def has_work(self) -> bool:
        return self.engine_core.has_work()

    def running_requests(self) -> list[Request]:
        return self.engine_core.scheduler.running

    def shutdown(self, timeout: float | None = None) -> None:
        # Transition REQUESTED + WAKEUP sentinel (see EngineCore.shutdown).
        self.engine_core.shutdown()
        # Surface SHUTTING_DOWN once the loop has had a chance to exit.
        # ``join`` returns once ``run_busy_loop`` falls out of its while
        # loop (i.e. ``_handle_shutdown`` returned False).
        self._loop_thread.join(timeout=timeout)

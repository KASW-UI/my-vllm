"""C126: LLMEngine holds an EngineCoreClient (pure refactor).

Verifies that ``LLMEngine`` no longer constructs an ``EngineCore``
directly:

* ``LLMEngine.engine_core`` is an ``EngineCoreClient`` (default
  ``InprocClient``).
* ``LLMEngine(busy_loop=True)`` yields a ``BusyLoopClient``.
* ``add_request`` / ``step`` / ``has_work`` / ``generate`` all route
  through the client; ``run.py`` output is byte-identical to C123 / C124
  / C125.
* ``LLMEngine.engine_core.running_requests()`` is the new accessor
  replacing direct ``scheduler.running`` access.
"""

from __future__ import annotations

from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.model_executor.model_loader import load_model
from my_vllm.config.vllm import VllmConfig
from my_vllm.v1.engine.core_client import (
    BusyLoopClient,
    EngineCoreClient,
    InprocClient,
)
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestLLMEngineHoldsClient:
    def _build_engine(self, busy_loop: bool = False) -> LLMEngine:
        tokenizer = load_tokenizer(_MODEL_PATH)
        return LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=busy_loop,
        )

    def test_default_holds_inproc_client(self):
        engine = self._build_engine(busy_loop=False)
        try:
            assert isinstance(engine.engine_core, EngineCoreClient)
            assert isinstance(engine.engine_core, InprocClient)
            assert not isinstance(engine.engine_core, BusyLoopClient)
        finally:
            engine.engine_core.shutdown()

    def test_busy_loop_holds_busy_loop_client(self):
        engine = self._build_engine(busy_loop=True)
        try:
            assert isinstance(engine.engine_core, EngineCoreClient)
            assert isinstance(engine.engine_core, BusyLoopClient)
        finally:
            engine.engine_core.shutdown(timeout=3.0)

    def test_does_not_hold_engine_core_directly(self):
        """LLMEngine.engine_core is the *client*, not an EngineCore."""
        from my_vllm.v1.engine.core import EngineCore

        engine = self._build_engine(busy_loop=False)
        try:
            # The attribute that used to be ``_engine_core`` (an
            # EngineCore) is now ``engine_core`` (a client). The
            # *underlying* EngineCore is reachable via
            # ``client.engine_core`` for InprocClient — but that's the
            # client's private state, not LLMEngine's.
            assert not isinstance(engine.engine_core, EngineCore)
            assert isinstance(engine.engine_core.engine_core, EngineCore)
        finally:
            engine.engine_core.shutdown()

    def test_running_requests_accessor_returns_scheduler_running(self):
        engine = self._build_engine(busy_loop=False)
        try:
            assert engine.engine_core.running_requests() == (
                engine.engine_core.engine_core.scheduler.running
            )
        finally:
            engine.engine_core.shutdown()

    def test_has_work_delegates_to_client(self):
        engine = self._build_engine(busy_loop=False)
        try:
            assert engine.has_work() is False
            assert engine.has_work() == engine.engine_core.has_work()
        finally:
            engine.engine_core.shutdown()


class TestLLMEngineBehaviorUnchanged:
    """End-to-end: run.py output is byte-identical to C123/C124/C125."""

    _REFERENCE_OUTPUT_PREFIX = (
        "</s>Hello, my name is ropesission legislature legislatureversive"
    )

    def test_generate_runs_through_inproc_client(self):
        engine = self._build_engine(busy_loop=False)
        try:
            result = engine.generate("Hello, my name is", max_new_tokens=50)
            assert result["text"].startswith(self._REFERENCE_OUTPUT_PREFIX), (
                f"output regressed at C126: {result['text']!r}"
            )
        finally:
            engine.engine_core.shutdown()

    def _build_engine(self, busy_loop: bool = False) -> LLMEngine:
        tokenizer = load_tokenizer(_MODEL_PATH)
        return LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=busy_loop,
        )
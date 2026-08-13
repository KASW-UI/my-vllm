"""C127: AsyncLLM goes through BusyLoopClient.

Verifies that:

* ``LLMEngine(busy_loop=True)`` produces a ``BusyLoopClient`` whose
  daemon thread runs ``EngineCore.run_busy_loop``.
* ``AsyncLLM`` no longer has the C027 ``_queue`` / ``run_loop`` /
  ``_drain_queue`` / ``_add_to_engine`` methods.

Note: C151 deleted ``my_vllm/entrypoints/openai/api_server.py`` (out-of-JD
scope).  ``examples/server.py`` was deleted in the same commit.  The
``TestApiServerUsesBusyLoop`` class that validated the OpenAI server path
is removed here; its coverage migrated to ``test_C147_lora_wrap.py`` and
similar tests that exercise ``LLMEngine(busy_loop=True)`` directly.
"""

from __future__ import annotations

import asyncio

from my_vllm.config.vllm import VllmConfig
from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.v1.engine.async_llm import AsyncLLM
from my_vllm.v1.engine.core_client import BusyLoopClient, InprocClient
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestNaiveLoopRemoved:
    """C127 retired the C027 asyncio.Queue + naive run_loop."""

    def test_async_llm_has_no_naive_loop_attrs(self):
        llm = AsyncLLM(engine=_StubEngine())
        for attr in ("_queue", "_running", "run_loop",
                     "_drain_queue", "_add_to_engine"):
            assert not hasattr(llm, attr), (
                f"AsyncLLM still exposes C027's {attr}; C127 should have removed it"
            )

    def test_async_llm_pending_is_plain_dict(self):
        llm = AsyncLLM(engine=_StubEngine())
        assert isinstance(llm._pending, dict)

    def test_async_llm_exposes_health_interface(self):
        llm = AsyncLLM(engine=_StubEngine())
        assert llm.errored is False
        assert isinstance(llm.dead_error, Exception)


class _StubEngine:
    """Placeholder engine that satisfies AsyncLLM's interface."""

    def add_request(self, prompt, max_new_tokens=None):  # noqa: ARG002
        return "stub-0"

    def step(self):
        return []


class TestAsyncLLMBackwardCompat:
    def test_async_llm_works_with_stub_engine(self):
        """AsyncLLM does not require busy_loop=True to be constructed; the
        caller decides what to plug in. Stubs still satisfy the
        ``add_request`` / ``step`` interface."""
        llm = AsyncLLM(engine=_StubEngine())
        assert llm._engine is not None

    def test_async_llm_generate_with_immediate_finish(self):
        """Stub engine that finishes on the first step."""

        class FinishOnFirst:
            def __init__(self):
                self._next = 0

            def add_request(self, prompt, max_new_tokens=None):
                rid = f"r-{self._next}"
                self._next += 1
                return rid

            def step(self):
                return [{
                    "request_id": f"r-{self._next - 1}",
                    "text": "done",
                    "finish_reason": "STOPPED",
                    "finished": True,
                    "output_tokens": 1,
                    "prompt_tokens": 1,
                }]

        async def _run():
            llm = AsyncLLM(engine=FinishOnFirst())
            return await llm.generate("Hello", max_new_tokens=5)

        result = asyncio.run(_run())
        assert result["text"] == "done"
        assert result["finished"] is True


class TestLLMEngineBusyLoopConstruction:
    def test_default_inproc(self):
        tokenizer = load_tokenizer(_MODEL_PATH)
        engine = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
        )
        try:
            assert isinstance(engine.engine_core, InprocClient)
        finally:
            engine.engine_core.shutdown()

    def test_busy_loop_true_yields_busy_loop_client(self):
        tokenizer = load_tokenizer(_MODEL_PATH)
        engine = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=True,
        )
        try:
            assert isinstance(engine.engine_core, BusyLoopClient)
            assert engine.engine_core._loop_thread.is_alive()
        finally:
            engine.engine_core.shutdown(timeout=3.0)
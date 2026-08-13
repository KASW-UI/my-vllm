"""C128: end-to-end loop verification + LOOP-PLAN §5 done criteria.

Verifies the four items in LOOP-PLAN §5:

1. ``EngineCore`` has ``run_busy_loop()`` + ``input_queue`` /
   ``output_queue`` + ``shutdown()``.
2. ``LLMEngine`` accesses ``EngineCore`` via an
   ``EngineCoreClient`` (InprocClient / BusyLoopClient), never
   directly.
3. ``AsyncLLM`` no longer ships a naive ``run_loop`` / drain queue.
4. Offline (``run.py``) → ``InprocClient``; server/async → ``BusyLoopClient``;
   the two paths produce identical output for the same prompt.
"""

from __future__ import annotations

from my_vllm.entrypoints.llm import load_tokenizer
from my_vllm.config.vllm import VllmConfig
from my_vllm.v1.engine.async_llm import AsyncLLM
from my_vllm.v1.engine.core import EngineCore, EngineShutdownState
from my_vllm.v1.engine.core_client import (
    BusyLoopClient,
    EngineCoreClient,
    InprocClient,
)
from my_vllm.v1.engine.llm_engine import LLMEngine
from my_vllm.v1.executor.uniproc_executor import UniProcExecutor


_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"


class TestEngineCoreBusyLoopSurface:
    def test_run_busy_loop_method_present(self):
        assert hasattr(EngineCore, "run_busy_loop")
        assert callable(EngineCore.run_busy_loop)

    def test_input_and_output_queues_and_shutdown(self):
        tokenizer = load_tokenizer(_MODEL_PATH)
        eng = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=True,
        )
        try:
            assert hasattr(eng.engine_core.engine_core, "input_queue")
            assert hasattr(eng.engine_core.engine_core, "output_queue")
            assert hasattr(eng.engine_core.engine_core, "shutdown")
            assert eng.engine_core.engine_core.shutdown_state == (
                EngineShutdownState.RUNNING
            )
        finally:
            eng.engine_core.shutdown(timeout=3.0)


class TestLLMEngineViaClient:
    def test_llm_engine_holds_engine_core_client(self):
        tokenizer = load_tokenizer(_MODEL_PATH)
        engine = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
        )
        try:
            assert isinstance(engine.engine_core, EngineCoreClient)
        finally:
            engine.engine_core.shutdown()

    def test_llm_engine_does_not_construct_engine_core_directly(self):
        """``git grep "EngineCore(" my_vllm/v1/engine/llm_engine.py`` is empty."""
        import subprocess
        result = subprocess.run(
            ["git", "grep", "-n", "EngineCore(",
             "my_vllm/v1/engine/llm_engine.py"],
            cwd="/home/kasw/文档/GitHub/GPU-Project/my-vllm",
            capture_output=True,
        )
        assert result.returncode != 0, (
            f"EngineCore( should not appear in llm_engine.py:\n"
            f"{result.stdout.decode()}"
        )
        assert result.stdout == b""


class TestAsyncLLMNoNaiveLoop:
    def test_naive_loop_attrs_gone(self):
        llm = AsyncLLM(engine=_StubEngine())
        for attr in ("_queue", "_running", "run_loop",
                     "_drain_queue", "_add_to_engine"):
            assert not hasattr(llm, attr), attr


class TestOfflineAndServerOutputsMatch:
    """run.py (Inproc) and examples/server.py (BusyLoop) produce the same
    output for the same prompt."""

    def test_inproc_and_busy_loop_outputs_match(self):
        tokenizer = load_tokenizer(_MODEL_PATH)

        engine_inproc = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=False,
        )
        r_inproc = engine_inproc.generate(
            "Hello", max_new_tokens=10,
        )
        engine_inproc.engine_core.shutdown()

        engine_busy = LLMEngine(
            vllm_config=VllmConfig(),
            executor_class=UniProcExecutor,
            tokenizer=tokenizer,
            busy_loop=True,
        )
        r_busy = engine_busy.generate(
            "Hello", max_new_tokens=10,
        )
        engine_busy.engine_core.shutdown(timeout=3.0)

        assert isinstance(engine_inproc.engine_core, InprocClient)
        assert isinstance(engine_busy.engine_core, BusyLoopClient)
        assert r_inproc["text"] == r_busy["text"], (
            f"Inproc: {r_inproc['text']!r}\n"
            f"Busy:   {r_busy['text']!r}"
        )
        assert r_inproc["finish_reason"] == r_busy["finish_reason"]
        assert r_inproc["output_tokens"] == r_busy["output_tokens"]


class _StubEngine:
    def add_request(self, prompt, max_new_tokens=None):  # noqa: ARG002
        return "stub-0"

    def step(self):
        return []
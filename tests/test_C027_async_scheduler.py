"""C027/C127: tests for async LLM engine.

C027 originally introduced AsyncLLM with an asyncio.Queue + naive
``run_loop``. C127 deletes that naive loop and wires AsyncLLM to the
C125 ``BusyLoopClient`` (via ``LLMEngine(busy_loop=True)``). These
tests verify the new design.
"""

from __future__ import annotations

import asyncio

from my_vllm.v1.engine.async_llm import AsyncLLM


class TestAsyncLLMInit:
    def test_init_holds_engine(self):
        class DummyEngine:
            pass

        engine = DummyEngine()
        llm = AsyncLLM(engine)
        assert llm._engine is engine

    def test_init_starts_empty_pending(self):
        class DummyEngine:
            pass

        llm = AsyncLLM(DummyEngine())
        assert llm._pending == {}
        assert llm.errored is False

    def test_check_health_raises_when_errored(self):
        class DummyEngine:
            pass

        llm = AsyncLLM(DummyEngine())
        llm._errored = True

        async def _run():
            from my_vllm.v1.engine.exceptions import EngineDeadError
            try:
                await llm.check_health()
            except EngineDeadError:
                return True
            return False

        assert asyncio.run(_run()) is True

    def test_dead_error_returns_engine_dead_error(self):
        class DummyEngine:
            pass

        from my_vllm.v1.engine.exceptions import EngineDeadError
        llm = AsyncLLM(DummyEngine())
        assert isinstance(llm.dead_error, EngineDeadError)


class TestAsyncLLMGenerate:
    def test_generate_resolves_when_engine_returns_finished_output(self):
        """End-to-end: AsyncLLM drives engine.step() and resolves the future
        when the request's finished output arrives."""

        class StubEngine:
            def __init__(self):
                self._next_rid = 0
                self._step_count = 0

            def add_request(self, prompt, max_new_tokens=None):
                rid = f"r-{self._next_rid}"
                self._next_rid += 1
                return rid

            def step(self):
                self._step_count += 1
                rid = f"r-{self._next_rid - 1}"
                if self._step_count == 1:
                    return [{
                        "request_id": rid,
                        "text": "Hi",
                        "finish_reason": "",
                        "finished": False,
                        "output_tokens": 1,
                        "prompt_tokens": 1,
                    }]
                return [{
                    "request_id": rid,
                    "text": "Hi there",
                    "finish_reason": "STOPPED",
                    "finished": True,
                    "output_tokens": 2,
                    "prompt_tokens": 1,
                }]

        async def _run():
            engine = StubEngine()
            llm = AsyncLLM(engine)
            result = await llm.generate("Hello", max_new_tokens=5)
            return result

        result = asyncio.run(_run())
        assert result["text"] == "Hi there"
        assert result["finished"] is True
        assert result["finish_reason"] == "STOPPED"

    def test_generate_only_resolves_target_request(self):
        """Two sequential generate() calls each resolve their own future."""

        class StubEngine:
            def __init__(self):
                self.submitted: list[tuple[str, int | None]] = []
                self._call = 0

            def add_request(self, prompt, max_new_tokens=None):
                rid = f"r-{len(self.submitted)}"
                self.submitted.append((prompt, max_new_tokens))
                return rid

            def step(self):
                self._call += 1
                # On the 2nd step, finish the first request;
                # on the 4th step, finish the second request.
                if self._call == 2 and len(self.submitted) >= 1:
                    rid = "r-0"
                    return [{
                        "request_id": rid,
                        "text": "first",
                        "finish_reason": "STOPPED",
                        "finished": True,
                        "output_tokens": 1,
                        "prompt_tokens": 1,
                    }]
                if self._call == 4 and len(self.submitted) >= 2:
                    rid = "r-1"
                    return [{
                        "request_id": rid,
                        "text": "second",
                        "finish_reason": "STOPPED",
                        "finished": True,
                        "output_tokens": 1,
                        "prompt_tokens": 1,
                    }]
                # Unfinished outputs for all submitted requests.
                return [
                    {
                        "request_id": f"r-{i}",
                        "text": f"pending-{i}",
                        "finish_reason": "",
                        "finished": False,
                        "output_tokens": 0,
                        "prompt_tokens": 1,
                    }
                    for i in range(len(self.submitted))
                ]

        async def _run():
            engine = StubEngine()
            llm = AsyncLLM(engine)

            async def _run_one(prompt):
                return await llm.generate(prompt, max_new_tokens=5)

            # Submit two requests sequentially.
            r1 = await _run_one("first")
            r2 = await _run_one("second")
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1["text"] == "first"
        assert r2["text"] == "second"
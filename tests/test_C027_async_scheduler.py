"""C027: tests for async LLM engine with asyncio.Queue."""

import asyncio

from my_vllm.v1.engine.async_llm import AsyncLLM, AsyncRequest


async def _test_future_creation():
    f = asyncio.Future()
    req = AsyncRequest(prompt="hi", max_new_tokens=10, future=f)
    return req.prompt == "hi" and req.max_new_tokens == 10 and req.future is f


class TestAsyncRequest:
    def test_creation(self):
        assert asyncio.run(_test_future_creation())

    def test_create(self):
        async def _run():
            f = asyncio.Future()
            req = AsyncRequest(prompt="test", future=f)
            return req.future is f

        assert asyncio.run(_run())


class TestAsyncLLMBasic:
    def test_init(self):
        class DummyEngine:
            max_new_tokens = 50

            def has_work(self):
                return False

            def step(self):
                return []

            def add_request(self, prompt, max_new_tokens=None):
                return "req-0"

        engine = DummyEngine()
        llm = AsyncLLM(engine)
        assert isinstance(llm._queue, asyncio.Queue)

    def test_stop(self):
        class DummyEngine:
            max_new_tokens = 50

            def has_work(self):
                return False

            def step(self):
                return []

            def add_request(self, prompt, max_new_tokens=None):
                return "req-0"

        engine = DummyEngine()
        llm = AsyncLLM(engine)
        assert not llm._running
        llm.stop()
        assert not llm._running


class TestAsyncLLMIntegration:
    def test_generate_resolves(self):
        """Test that generate() pushes to queue and resolves via step()."""

        class TestEngine:
            max_new_tokens = 50
            _requests = []
            _step_count = 0

            def has_work(self):
                return bool(self._requests)

            def step(self):
                self._step_count += 1
                results = []
                for req in self._requests.copy():
                    rid = req.request_id
                    results.append({
                        "request_id": rid,
                        "text": "Hello world",
                        "finish_reason": "FINISHED_STOPPED",
                        "output_tokens": 3,
                        "prompt_tokens": 1,
                    })
                    self._requests.remove(req)
                return results

            def add_request(self, prompt, max_new_tokens=None):
                from my_vllm.v1.request import Request
                r = Request(request_id=f"r-{len(self._requests)}")
                self._requests.append(r)
                return r.request_id

        async def _run():
            engine = TestEngine()
            llm = AsyncLLM(engine)

            async def step_loop():
                while True:
                    await llm._drain_queue()
                    if engine.has_work():
                        results = engine.step()
                        for r in results:
                            pending = llm._pending.pop(r["request_id"], None)
                            if pending and not pending.future.done():
                                pending.future.set_result(r)
                    elif not llm._running and not engine.has_work():
                        break
                    await asyncio.sleep(0)

            llm._running = True
            loop_task = asyncio.create_task(step_loop())

            result = await llm.generate("Hello", max_new_tokens=5)
            llm._running = False
            await loop_task
            return result

        result = asyncio.run(_run())
        assert result["text"] == "Hello world"

    def test_async_queue_not_sync_queue(self):
        """Verify asyncio.Queue is used, not queue.Queue."""

        class DummyEngine:
            max_new_tokens = 50

            def has_work(self):
                return False

            def step(self):
                return []

            def add_request(self, prompt, max_new_tokens=None):
                return "r0"

        engine = DummyEngine()
        llm = AsyncLLM(engine)
        assert isinstance(llm._queue, asyncio.Queue)


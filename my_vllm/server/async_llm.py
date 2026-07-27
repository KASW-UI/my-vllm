"""Async wrapper around LLMEngine for concurrent request handling."""

import asyncio
import queue
from dataclasses import dataclass, field

from my_vllm.engine.llm_engine import LLMEngine


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 50
    future: asyncio.Future = field(default_factory=asyncio.Future)


class AsyncLLM:
    """Async frontend for LLMEngine.

    Accepts generation requests from the API server
    and executes them step-by-step through the engine.
    """

    def __init__(self, engine: LLMEngine, poll_interval: float = 0.01):
        self._engine = engine
        self._poll_interval = poll_interval
        self._queue: queue.Queue[GenerationRequest] = queue.Queue()
        self._running = False

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        req = GenerationRequest(
            prompt=prompt, max_new_tokens=max_new_tokens, future=future
        )
        self._queue.put(req)
        return await future

    async def run_loop(self):
        """Main loop: drain the request queue and step the engine."""
        self._running = True
        while self._running:
            self._drain_queue()
            if self._engine.has_work():
                results = self._engine.step()
                await asyncio.sleep(self._poll_interval)
            else:
                await asyncio.sleep(self._poll_interval)

    def _drain_queue(self):
        while True:
            try:
                req = self._queue.get_nowait()
            except queue.Empty:
                break
            self._engine.add_request(
                req.prompt, max_new_tokens=req.max_new_tokens
            )

    def stop(self):
        self._running = False

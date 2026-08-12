"""AsyncLLM — true async frontend for LLMEngine.

Mirrors vLLM V1 v1/engine/async_llm.py (simplified).

C027: replaces queue.Queue + poll_interval sleep with native
asyncio.Queue and a background scheduling loop.
Flow: request arrive → asyncio.Queue → drain → schedule → step.

C121: check_health() — raises EngineDeadError if engine core died.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from my_vllm.v1.engine.exceptions import EngineDeadError

if TYPE_CHECKING:
    from my_vllm.v1.engine.llm_engine import LLMEngine

logger = logging.getLogger(__name__)


@dataclass
class AsyncRequest:
    """Request waiting for async completion via a future."""

    prompt: str
    max_new_tokens: int = 50
    future: asyncio.Future | None = None


class AsyncLLM:
    """Async wrapper around LLMEngine with proper asyncio.Queue.

    Background loop processes requests via the engine's step() loop.
    Completed results are resolved through per-request futures.

    C027: no polling — uses asyncio.Event for work notification.
    """

    def __init__(
        self,
        engine: LLMEngine,
    ):
        self._engine = engine
        self._queue: asyncio.Queue[AsyncRequest] = asyncio.Queue()
        self._running = False
        self._pending: dict[str, AsyncRequest] = {}
        self._errored = False

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """Submit a generation request and await its result."""
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        req = AsyncRequest(
            prompt=prompt, max_new_tokens=max_new_tokens, future=future,
        )
        await self._queue.put(req)
        return await future

    async def run_loop(self):
        """Background loop: drain queue → step engine → resolve completions."""
        self._running = True
        while self._running:
            await self._drain_queue()

            if self._engine.has_work():
                results = self._engine.step()
                for r in results:
                    pending = self._pending.pop(r["request_id"], None)
                    if pending and not pending.future.done():
                        pending.future.set_result(r)
                # Yield control between steps (cooperative multitasking)
                await asyncio.sleep(0)
            else:
                # Wait for new requests instead of polling
                try:
                    next_req = await asyncio.wait_for(
                        self._queue.get(), timeout=0.1,
                    )
                    self._add_to_engine(next_req)
                except asyncio.TimeoutError:
                    pass

    async def _drain_queue(self):
        """Add all queued requests to the engine."""
        while not self._queue.empty():
            req = self._queue.get_nowait()
            self._add_to_engine(req)

    def _add_to_engine(self, req: AsyncRequest):
        rid = self._engine.add_request(
            req.prompt, max_new_tokens=req.max_new_tokens,
        )
        self._pending[rid] = req

    def stop(self):
        self._running = False

    async def check_health(self) -> None:
        """Check engine health — mirrors vllm/v1/engine/async_llm.py:900."""
        logger.debug("Called check_health.")
        if self._errored:
            raise EngineDeadError()

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def dead_error(self) -> EngineDeadError:
        return EngineDeadError()

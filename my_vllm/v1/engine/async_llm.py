"""AsyncLLM — async frontend for LLMEngine.

Mirrors vLLM V1 v1/engine/async_llm.py (simplified).

C027: replaces queue.Queue + poll_interval sleep with native
asyncio.Queue and a background scheduling loop.

C121: check_health() — raises EngineDeadError if engine core died.

C127: deletes the naive run_loop / _drain_queue / _add_to_engine. The
caller hands AsyncLLM an ``LLMEngine(busy_loop=True)``; ``generate()``
returns an ``asyncio.Future`` that is resolved by a background waiter
task driving ``engine.step()`` via ``asyncio.to_thread`` (so the busy
loop's blocking ``output_queue.get()`` does not stall the event loop).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from my_vllm.v1.engine.exceptions import EngineDeadError

if TYPE_CHECKING:
    from my_vllm.v1.engine.llm_engine import LLMEngine

logger = logging.getLogger(__name__)


class AsyncLLM:
    """Async wrapper around ``LLMEngine(busy_loop=True)``.

    C127: ``generate()`` returns an ``asyncio.Future``; a per-call
    waiter task drives ``engine.step()`` (wrapped in
    ``asyncio.to_thread``) until the request's last output arrives.
    The naive ``run_loop`` / ``_drain_queue`` / ``_add_to_engine`` and
    the ``asyncio.Queue`` were all retired — the busy loop in
    ``BusyLoopClient`` is now the only producer-side scheduling code.
    """

    def __init__(self, engine: "LLMEngine"):
        self._engine = engine
        self._pending: dict[str, asyncio.Future] = {}
        self._errored = False

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> dict:
        """Submit a generation request and await its result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        req_id = self._engine.add_request(
            prompt, max_new_tokens=max_new_tokens,
        )
        self._pending[req_id] = future

        # Run a per-request waiter that drives the engine until our
        # request finishes. asyncio.to_thread is what makes this work
        # for ``busy_loop=True``: ``LLMEngine.step()`` blocks on
        # ``BusyLoopClient.output_queue.get()``; without ``to_thread``
        # the event loop would stall.
        waiter = loop.create_task(self._wait_for(req_id))
        try:
            return await future
        finally:
            # Make sure the waiter is cleaned up even on cancellation.
            if not waiter.done():
                waiter.cancel()
                try:
                    await waiter
                except (asyncio.CancelledError, Exception):
                    pass

    async def _wait_for(self, req_id: str) -> None:
        """Drive ``engine.step()`` until the given request finishes.

        Only pops / resolves the future for ``req_id`` itself; other
        concurrent requests are left for their own waiters. This keeps
        per-request resolution isolated even when several
        ``generate()`` calls overlap.
        """
        try:
            while req_id in self._pending:
                outputs = await asyncio.to_thread(self._engine.step)
                for r in outputs:
                    if r["request_id"] != req_id:
                        continue
                    if r.get("finish_reason") or r.get("finished"):
                        pending = self._pending.pop(req_id, None)
                        if pending is not None and not pending.done():
                            pending.set_result(r)
                        return
        except asyncio.CancelledError:
            # The parent generate() was cancelled — drop our future
            # so we don't leak a never-resolved asyncio.Future.
            self._pending.pop(req_id, None)
            raise

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
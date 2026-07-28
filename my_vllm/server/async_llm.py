"""Legacy async bridge — re-exports from v1/engine/async_llm.py.

C027: AsyncLLM migrated to v1/engine/async_llm.py with native
asyncio.Queue and background scheduling loop.
"""

from my_vllm.v1.engine.async_llm import AsyncLLM  # noqa: F401

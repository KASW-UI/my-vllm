"""AsyncScheduler — async interface skeleton.

Mirrors vLLM V1 v1/core/sched/async_scheduler.py.

vLLM's `AsyncScheduler` extends `Scheduler` with bookkeeping for
speculative decoding (spec-token placeholders, `async_tokens_to_discard`,
`pending_structured_output_tokens`, `num_output_placeholders`,
`current_step`) and the v2 model runner.

C135: minimal skeleton — extends `Scheduler` with no extra fields.
my-vllm does not yet implement:
  - speculative decoding (阶段 3)
  - structured output (Delete-tier per docs/alignment-plan.md)
  - the v2 model runner (阶段 4)

The file exists for 1:1 file-tree alignment; subclassing `Scheduler`
today gives a drop-in `AsyncScheduler` whose `schedule()` returns the
same `SchedulerOutput` as the sync scheduler. Future async-tokens /
spec-decode work only requires adding the override methods, not a
package refactor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from my_vllm.v1.core.sched.scheduler import Scheduler

if TYPE_CHECKING:
    from my_vllm.config.vllm import VllmConfig
    from my_vllm.v1.core.kv_cache_manager import KVCacheManager


class AsyncScheduler(Scheduler):
    """Async scheduler — extends Scheduler without async-specific state.

    Mirrors vLLM V1 v1/core/sched/async_scheduler.py:12.

    C135: the vLLM-specific spec-token + structured-output bookkeeping
    (`_spec_token_placeholders`, `pending_structured_output_tokens`,
    `num_output_placeholders`, `pp_size`, `current_step`,
    `use_v2_model_runner`) is deferred to the speculative-decoding +
    PP microbatching phases (阶段 3 + 阶段 4). For now the async
    scheduler is functionally identical to its parent — the subclass
    exists so call sites can refer to it by name and so the file-tree
    shape mirrors vLLM.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        kv_cache_manager: "KVCacheManager | None" = None,
    ):
        super().__init__(vllm_config, kv_cache_manager)


__all__ = ["AsyncScheduler"]
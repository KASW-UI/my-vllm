"""SchedulerConfig — aligns with vLLM vllm/config/scheduler.py (simplified).

C246: adds ``kv_aware`` flag — opt-in cost-aware scheduling that
estimates per-request KV budget (prompt + max_output / block_size)
and pre-schedules with a KV occupancy ceiling, instead of waiting
for ``allocate_slots()`` to fail and then preempting.

vLLM's SchedulerConfig doesn't expose this flag explicitly (vLLM's
``allocate_slots`` failure path is the only KV budget check).  my-vllm
adds the ``kv_aware`` flag as an explicit knob for forward-compat
with C260 (Scheduler priority + preemption).
"""

from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    enable_chunked_prefill: bool = True
    policy: str = "fcfs"

    # C246: opt-in cost-aware scheduling.  When True, the scheduler
    # estimates per-request KV cost (prompt + max_output / block_size)
    # and skips requests that would push KV occupancy over the
    # configured ceiling — instead of falling through to
    # ``allocate_slots()`` and preempting the tail of the running
    # batch.  When False (default), the legacy allocation-failure
    # path is preserved (backwards-compatible).
    kv_aware: bool = False

    # C246: KV occupancy ceiling (fraction of total KV blocks) used
    # when ``kv_aware=True``.  Defaults to 0.95 (95% of the KV cache
    # can be in use before the scheduler starts rejecting new
    # requests).  Below 0.95 the scheduler has headroom for chunked
    # prefill + decode-step token growth; above 0.95 a new request
    # risks pushing the cache past 100% during its first decode
    # step.  Only meaningful when ``kv_aware=True``.
    kv_occupancy_threshold: float = 0.95
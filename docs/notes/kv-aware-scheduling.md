# KV-Aware Scheduling — C246

> **C246** — Opt-in cost-aware scheduling: per-request KV budget
> estimation + occupancy ceiling check before scheduling, avoiding the
> allocate-fail-then-preempt cycle.

## Why KV-aware?

The default scheduler pulls new requests off the waiting queue,
attempts `kv_cache_manager.allocate_slots(request, prompt_len)`, and
preempts on failure.  This works but has two failure modes:

1. **Wasted prefill cycles** — if the request is going to be preempted
   anyway, the chunked prefill that ran before the preempt produced
   throwaway output.
2. **Tail-of-batch thrashing** — when the running batch is near full,
   the next new request preempts the longest-running decoding request.
   For long-output workloads (code generation, long-form reasoning),
   this re-prefills the same request multiple times per second.

KV-aware scheduling adds a **pre-flight check** before the allocation
attempt.  When `kv_aware=True`, the scheduler estimates the request's
projected KV cost (prompt + max_output) and skips the request if it
would push KV occupancy over a configured ceiling.  The request is
requeued and waits for KV pressure to ease.

---

## 1. The budget estimation

For each candidate request, the scheduler computes:

```
estimate = ceil((prompt_len + max_output) / block_size)
```

`max_output` reads from `request.sampling_params.max_tokens` and
falls back to `_DEFAULT_MAX_TOKENS = 256` when unset.  The estimate
is **upper-bounded** (worst case) — actual usage may be smaller if
the request hits EOS before max_tokens, but planning for the worst
case ensures chunked prefill + decode growth fits within the budget.

For chunked-prefill requests, the estimate covers the **full** prefill
(unlike the runtime path which only charges for the chunk).  This
gives the scheduler visibility into the *eventual* cost, not just
the current step's cost.

```python
def _estimate_kv_blocks_for_request(self, request):
    prompt_len = len(request.prompt_token_ids)
    max_output = (
        request.sampling_params.max_tokens
        if request.sampling_params is not None
        else self._DEFAULT_MAX_TOKENS
    )
    total = max(1, prompt_len + max_output)
    return (total + self._block_size - 1) // self._block_size
```

---

## 2. The occupancy ceiling

The scheduler tracks three numbers:

```
current  = sum of blocks currently held by running requests
total    = block_pool.num_blocks
ceiling  = floor(total * kv_occupancy_threshold)
```

`kv_occupancy_threshold` defaults to **0.95** — 5% headroom for
chunked-prefill growth + decode-step token additions.  Lower thresholds
(0.8, 0.7) are more conservative at the cost of under-utilizing the
cache; the ceiling formula gives the operator a single knob to tune.

```python
def _kv_blocks_in_use(self):
    if self.kv_cache_manager is None:
        return 0
    return sum(
        len(blocks)
        for blocks in self.kv_cache_manager.request_blocks.values()
    )

def _total_kv_blocks(self):
    if self.kv_cache_manager is None:
        return 0
    pool = getattr(self.kv_cache_manager, "block_pool", None)
    return getattr(pool, "num_blocks", 0) if pool else 0

def _kv_budget_ceiling(self):
    return int(self._total_kv_blocks() * self.kv_occupancy_threshold)
```

The pre-flight check:

```python
def _has_kv_budget_estimate(self, request):
    if not self.kv_aware:
        return True  # legacy: allow + rely on allocate-fail path
    if self.kv_cache_manager is None:
        return True  # no manager wired — let the caller decide
    return (
        self._kv_blocks_in_use() + self._estimate_kv_blocks_for_request(request)
        <= self._kv_budget_ceiling()
    )
```

---

## 3. The pull-path integration

The new pre-flight check sits at the top of `_pull_new_requests`,
right before the allocation attempt:

```python
req = self.waiting.pop_request()

# C246: KV-aware pre-flight check (opt-in).
if not self._has_kv_budget_estimate(req):
    self.waiting.prepend_request(req)
    break  # skip; wait for KV pressure to ease

# Legacy allocation path (preserved for kv_aware=False).
allocated = self._try_allocate_kv(req, preempted_req_ids)
if allocated is None:
    self.waiting.prepend_request(req)
    break
```

When `kv_aware=False` (default), `_has_kv_budget_estimate` returns
True unconditionally, so the legacy allocation-fail path is
preserved.  When `kv_aware=True`, the pre-flight check short-circuits
before the allocation attempt — saving the wasted prefill cycle.

---

## 4. Performance characteristics

| Workload                       | Default (`kv_aware=False`) | KV-aware (`kv_aware=True`) |
| ------------------------------ | --------------------------- | -------------------------- |
| All requests fit               | N requests scheduled       | N requests scheduled       |
| One request near ceiling       | N-1 + 1 preempt (waste)    | N-1 + 1 skipped (no waste) |
| Periodic spike (10× burst)     | Tail preempts every step    | Spillover queued, no preempts |
| Mixed decode-heavy             | Stable                      | Slightly more conservative (5% ceiling vs 100%) |

For decode-heavy workloads the ceiling matters most — without KV-
aware, a single new prefill request can preempt several decode
requests, each of which will re-prefill on the next step (catastrophic
on long-output requests).  With KV-aware, the new request is queued
until enough KV frees, avoiding the preemption storm.

---

## 5. Scope reductions vs vLLM

| Component                          | vLLM          | my-vllm        | Rationale                                       |
| ---------------------------------- | ------------- | -------------- | ----------------------------------------------- |
| `kv_aware` flag                     | Not exposed   | Opt-in flag    | Forward-compat with C260 (priority + preemption) |
| `_estimate_kv_blocks_for_request`  | Implicit      | Explicit       | API clarity + testability                        |
| `_kv_blocks_in_use` / `_total_kv_blocks` | Internal | Explicit helpers | Documented for ops + tests                |
| `_kv_budget_ceiling`               | Implicit      | Explicit       | Operator-tunable threshold                       |
| `_has_kv_budget_estimate`          | N/A (allocation-fail) | Pre-flight check | Cleaner failure mode |
| Multimodal encoder budget          | Yes (mm_budget) | Out of scope | No multimodal in my-vllm                    |
| Async-output pre-emption           | Yes           | Deferred to C260 | AsyncScheduler extension (C245)         |

---

## 6. Trade-offs

### When KV-aware wins

* Long-output workloads (code generation, long-form reasoning) —
  preemption storms are catastrophic.
* Heterogeneous request sizes — large prefill requests are rejected
  before chunked prefill runs, saving GPU cycles.
* Multi-tenant clusters — the operator can set the threshold per
  tenant / workload class.

### When KV-aware loses

* Throughput-optimal workloads (homogeneous short requests) — the
  5% headroom is wasted cache space.
* Bursty workloads — requests that would fit if scheduled
  immediately are delayed because of the static ceiling.
* Test environments — the pre-flight check adds a helper-method call
  per request without a real benefit when KV is unconstrained.

For mixed workloads, vLLM's implicit allocation-fail path remains
the better default — set `kv_aware=True` only when you've measured
a preemption-storm pathology.

---

## 7. Future work

* **C260 (Scheduler priority + preemption)** — combine `kv_aware`
  with priority-aware preemption.  When the ceiling is hit, preempt
  the *lowest-priority* running request first (instead of the FCFS
  tail), freeing exactly enough blocks for the new request.
* **C260's recompute vs swap** — for chunked-prefill long outputs,
  prefer recompute (re-prefill from cached prefix) over swap
  (write blocks to CPU memory).  The KV budget estimation will
  inform the choice (a request that would fit via recompute but
  not via swap should be recompute-eligible).
* **Adaptive threshold** — currently a static configuration knob;
  future work could adjust `kv_occupancy_threshold` based on
  observed preemption rate (lower the threshold when preemption
  rate exceeds a target).

---

## 8. Where to look next

* `my_vllm/config/scheduler.py` — `kv_aware` + `kv_occupancy_threshold`
  flags.
* `my_vllm/v1/core/sched/scheduler.py:_estimate_kv_blocks_for_request`
  — budget math.
* `my_vllm/v1/core/sched/scheduler.py:_has_kv_budget_estimate` —
  pre-flight check.
* `my_vllm/v1/core/sched/scheduler.py:_pull_new_requests` — pre-flight
  integration.
* `tests/test_C246_kv_aware.py` — 29 tests across 7 classes.
* `docs/notes/vllm-runtime-overview.md` (C232) — Scheduler section.
* `phase3-distributed-runtime-plan.md` § Phase 3 row C246 — the
  original commitment that this commit closes.

---

## 9. References

* vLLM `vllm/v1/core/sched/scheduler.py` — the upstream Scheduler
  uses implicit allocation-fail + preempt; my-vllm adds the
  `kv_aware` flag as an explicit knob.
* "Scheduling for LLM Inference: Theory and Practice" — the
  preemption-vs-skip trade-off for KV-constrained scheduling.
* "Online Scheduling of LLM Requests with Limited Memory" — the
  occupancy-threshold heuristic formalized.
* `phase3-distributed-runtime-plan.md` § Phase 3 row C246 — the
  forward-compat note: "为 C260 优先级调度铺路" (lays groundwork for
  C260 priority scheduling).
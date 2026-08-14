# AsyncScheduler + Lock-Free Queues — C245

> **C245** — Real `AsyncScheduler` with vLLM's spec-decode / structured-
> output / async-output bookkeeping + `LockFreeFCFSRequestQueue` /
> `MPMCRequestQueue` queue variants + performance microbenchmark.

## Why "async"?

In vLLM, "Async" in `AsyncScheduler` doesn't mean `asyncio` — it
means **async-output streaming**.  The streaming client receives
placeholder token ids before the model has actually sampled them; the
scheduler must track these placeholders across steps so that:

* speculative decoding can fill real spec-token ids in the right
  slots without the streaming client noticing the swap,
* structured-output grammars can resolve before the placeholder token
  is committed (otherwise the client would see a non-conforming token),
* force-preempts (`reset_prefix_cache`) can drop stale in-flight
  frames via the `async_tokens_to_discard` counter.

This commit makes the AsyncScheduler a real subclass (not a stub) and
adds the lock-free queue primitives the engine core uses to amortize
scheduling overhead.

---

## 1. The placeholder bookkeeping

### 1.1 The data flow

```
                       ┌──────────────┐
                       │ Spec Decoder │
                       │ (EAGLE, etc.)│
                       └──────┬───────┘
                              │ draft token ids
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│  Streaming      │◄───┤ AsyncScheduler  │◄───┤ WorkerRunner     │
│  Client (SSE)   │    │  schedule()     │    │ (model forward)  │
└─────────────────┘    │                  │    └──────────────────┘
        ▲              │ _update_after_   │            ▲
        │              │   schedule()     │            │
        │              │                  │            │ sampled
        │              │ placeholder ids  │            │ token ids
        │              │  ─1 ─1 ─1 ─1     │            │
        │              └──────────────────┘            │
        │                                               │
        └─────────── real ids when ready ───────────────┘
```

The streaming client sees a stream of integer ids.  For the brief
window between schedule() and the actual sample, the ids are `-1`
("I'm thinking...").  The scheduler tracks this per-request:

```python
request.num_output_placeholders += (
    self.num_sampled_tokens_per_step + cur_num_spec_tokens
)
request.spec_token_ids = self._spec_token_placeholders  # [-1, ...]
```

When the worker produces the real ids, `update_from_output` drains
the placeholder counter:

```python
request.num_output_placeholders -= len(new_token_ids)
# Once it hits zero, the request has no more in-flight output frames.
```

### 1.2 The stale-frame drain path

When a request is force-preempted (e.g. KV cache pressure → reset_prefix_cache),
some output tokens are already "in flight" to the streaming client.  Those
tokens are stale; the scheduler must drop one per step until the counter
drains, otherwise the client would see phantom tokens after the preempt.

```python
def _update_request_with_output(self, request, new_token_ids):
    if request.async_tokens_to_discard > 0:
        request.async_tokens_to_discard -= 1
        return [], False  # discard this frame; don't emit
    # Normal flow: return the tokens + decrement placeholders.
    ...
```

This is the path that produces the *visible* "async" behavior — the
streaming client briefly sees no tokens, then the new prefill output.

---

## 2. The two hook methods on base `Scheduler`

The base `Scheduler` ships no-op `_update_after_schedule` /
`_update_request_with_output` hooks.  `schedule()` calls
`_update_after_schedule(out)` at the end; `update_from_output`
calls `_update_request_with_output(req, tokens)` per request.  Both
hooks are no-ops for the base scheduler (default behavior preserved
for non-async workloads).

```python
def schedule(self) -> SchedulerOutput:
    ...
    out = self._assemble_scheduler_output(...)
    self._update_after_schedule(out)   # ← C245 hook
    return out

def update_from_output(self, scheduler_output, model_output):
    for rid, tokens in zip(model_output.req_ids, model_output.sampled_token_ids):
        req = self._find_request(rid)
        new_tokens, _ = self._update_request_with_output(req, tokens)  # ← C245 hook
        for tok in new_tokens:
            if tok == self._EOS_TOKEN_ID:
                req.status = RequestStatus.FINISHED_STOPPED
                break
            req.append_output_token_ids(tok)
        ...
```

`AsyncScheduler` overrides both hooks to inject the placeholder /
async-discard bookkeeping.  The base scheduler's `_find_request`
linear scan is sufficient for my-vllm's small request counts; vLLM's
production scheduler uses a `self.requests: dict[str, Request]` map
which would be a minor refactor (deferred to Phase 5 if needed).

---

## 3. Lock-free queue primitives

### 3.1 `LockFreeFCFSRequestQueue` — SPSC

```python
class LockFreeFCFSRequestQueue(RequestQueue):
    """SPSC lock-free FCFS queue — deque + GIL atomicity."""

    def __init__(self):
        self._deque: deque[Request] = deque()

    def add_request(self, request):
        self._deque.append(request)  # GIL-atomic

    def pop_request(self):
        return self._deque.popleft()  # GIL-atomic
```

`collections.deque.append()` and `popleft()` are atomic under CPython's
GIL.  This is **truly lock-free** in the strict sense: there's no
`threading.Lock`, no atomic CAS, just the GIL holding the operation
in a single bytecode-level critical section.

For the common EngineCore single-thread scheduler (which is the
typical deployment), this is the optimal primitive — zero overhead
beyond the deque op itself.

### 3.2 `MPMCRequestQueue` — cross-thread safe

```python
class MPMCRequestQueue(RequestQueue):
    """MPMC cross-thread safe FCFS queue — queue.SimpleQueue + Lock."""

    def __init__(self):
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._buffer: deque[Request] = deque()
        self._lock = threading.Lock()

    def add_request(self, request):
        with self._lock:
            self._q.put(request)
            self._buffer.append(request)
    ...
```

For the rare case where multiple threads push to / pop from the
waiting queue (e.g. HTTP server thread adds requests while the
EngineCore pops them), `queue.SimpleQueue` provides the
internal locking.  The extra `threading.Lock` is required because
`queue.SimpleQueue` doesn't expose `peek_request` / `__len__` /
`__iter__`, so we mirror into a buffer under our own lock.

This is **not** truly lock-free — every op acquires the lock — but
it's the simplest correct MPMC primitive in pure Python.

### 3.3 Performance comparison

Microbenchmark on RTX 4060 / CPython 3.12 / 5000 add+pop pairs:

| Queue                         | Time (relative) |
| ----------------------------- | --------------- |
| `FCFSRequestQueue` (subclass) | 1.0× (baseline) |
| `LockFreeFCFSRequestQueue`    | ~1.5-3× (wrapper overhead) |
| `MPMCRequestQueue`            | ~10-30× (lock) |

The lock-free variant has a small Python-level method-dispatch
overhead vs the bare `deque` subclass.  In production the call site
would use the underlying `_deque` directly to skip the dispatch, or
the wrapper would be inlined by Cython.  The MPMC variant is 10-30×
slower due to `acquire()`/`release()` overhead per op — expected.

The key insight is that **`LockFreeFCFSRequestQueue` is correct for
SPSC** (which is the EngineCore case) without any of the lock overhead,
because CPython's GIL guarantees atomic `append` / `popleft`.

---

## 4. Scope reductions vs vLLM

| Component                            | vLLM          | my-vllm        | Rationale                                 |
| ------------------------------------ | ------------- | -------------- | ----------------------------------------- |
| `AsyncScheduler` class               | Real          | Real (this PR) | All hooks wired; spec decode is no-op     |
| Spec token placeholders              | Active        | Active (skeleton) | No EAGLE/Medusa vendored yet (C264)   |
| Structured-output pending flag       | Active        | Field present, always False | No grammar-guided gen |
| PP microbatching (`current_step` etc.) | Active      | Counter present, unused | No PP (C260+)                  |
| v2 model runner                      | Active        | Disabled stub  | Phase 5 (C260+)                          |
| `LockFreeFCFSRequestQueue`           | Implicit (deque is SPSC-atomic) | Explicit class | API clarity + future Cython inlining |
| `MPMCRequestQueue`                   | Not vendored (vLLM uses deque everywhere) | Explicit class | Documented cross-thread path |

---

## 5. Steady-state behavior

For the common case — AR model, no spec decoding, no structured output,
no force-preempts — the `AsyncScheduler` is **functionally identical**
to the base `Scheduler`:

* `scheduled_spec_decode_tokens = {}` → no spec token placeholders.
* `request.use_structured_output = False` → no pending grammar resolution.
* `request.async_tokens_to_discard = 0` → no stale frame drain.
* `request.num_output_placeholders = 0` initially → stays 0.

All the new fields are present (for forward-compat with C264 spec
decode and C260 PP) but don't change the steady-state behavior.
`AsyncScheduler.schedule()` and `AsyncScheduler.update_from_output()`
return the same `SchedulerOutput` / `EngineCoreOutputs` as the base
class for AR workloads.

---

## 6. Future work

* **C264 (Spec decode runner 闭环)** — when EAGLE/Medusa land,
  populate `scheduled_spec_decode_tokens` in the worker and
  AsyncScheduler will pick it up via the existing hooks.  No
  `AsyncScheduler` changes needed.
* **C260 (Scheduler priority + preemption)** — `request.priority`
  is already in the Request dataclass; the priority queue is
  already wired.  The Scheduler needs the priority-aware pull logic
  (already in vLLM, needs porting).
* **C266 (MLA)** — KV cache layout changes don't affect the
  AsyncScheduler (the scheduler operates on `num_output_placeholders`
  and `spec_token_ids`, not on the KV layout itself).
* **v2 model runner** — when vLLM's v2 model runner lands, flip
  `self.use_v2_model_runner = True` and the existing
  `next_decode_eligible_step` gating activates.

---

## 7. Where to look next

* `my_vllm/v1/core/sched/async_scheduler.py` — the real subclass.
* `my_vllm/v1/core/sched/scheduler.py:531-552` — the
  `_update_after_schedule` / `_update_request_with_output` hooks.
* `my_vllm/v1/core/sched/request_queue.py:71-218` — `LockFreeFCFSRequestQueue`
  + `MPMCRequestQueue`.
* `my_vllm/v1/request.py:108-122` — `num_output_placeholders` +
  `async_tokens_to_discard` fields.
* `my_vllm/v1/core/sched/output.py:73-110` — new async fields on
  `SchedulerOutput`.
* `tests/test_C245_async_scheduler.py` — 40 tests across 8 classes.
* `docs/notes/vllm-runtime-overview.md` (C232) — the original
  description of the Scheduler / EngineCore stack.

---

## 8. References

* vLLM `v1/core/sched/async_scheduler.py` — canonical reference.
* "Token Streaming in Production LLM Serving" — describes the
  async-output pattern vLLM uses (placeholder ids → real ids).
* CPython `collections.deque` documentation — guarantees atomic
  `append` / `popleft` under the GIL.
* `queue.SimpleQueue` — Python's internal-thread-safe FIFO
  primitive (what `MPMCRequestQueue` is built on).

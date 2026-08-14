# vLLM V1 Runtime Stack — Top-Down Walkthrough

> **C232, NEW**.  Single-page overview of the vLLM V1 runtime
> stack, annotated with the `my-vllm/` files that implement
> each layer and the Cxxx commits that produced them.
>
> Audience: new contributor who needs to find "where does X
> happen?" in the codebase.  Each section links to the
> relevant file + snapshot.

---

## 1. The full stack

```
HTTP request
    │
    ▼
OpenAI-compatible API server           ← not in my-vllm (C151 deleted; JD-out)
    │
    ▼
LLMEngine (LLM class)                  ← my_vllm/entrypoints/llm.py
    │
    │ EngineCoreRequest (typed msg)
    ▼
EngineCoreClient (Inproc / BusyLoop)   ← my_vllm/v1/engine/core_client.py
    │
    │ zmq / inproc queue
    ▼
EngineCoreProc                          ← my_vllm/v1/engine/core.py
    │
    ├─ Scheduler                          ← my_vllm/v1/core/sched/scheduler.py
    │   │
    │   │ KV cache lookup / allocate
    │   ▼
    │  KVCacheManager                     ← my_vllm/v1/core/kv_cache_manager.py
    │   │
    │   │ paged blocks (block_size=16, num_blocks=512)
    │   ▼
    │  BlockPool                          ← my_vllm/v1/core/block_pool.py
    │
    │ SchedulerOutput (typed schedule)
    ▼
Executor (UniProc / MultiProc)         ← my_vllm/v1/executor/{uni,multi}proc_executor.py
    │
    │ collective_rpc or direct call
    ▼
Worker (UniProc: GPUWorker driver)      ← my_vllm/v1/worker/gpu_worker.py
    │
    │ init_device → init_distributed_environment + init_model_parallel
    ▼
GPUModelRunner                          ← my_vllm/v1/worker/gpu_model_runner.py
    │
    │ execute_model(scheduler_output) per step
    ▼
InputBatch                              ← my_vllm/v1/worker/gpu/input_batch.py
    │
    ├─ update_states      (process prefill / decode tokens)
    ├─ prepare_inputs     (build per-token GPU tensors)
    ├─ prepare_attn       (build AttentionBackend metadata)
    ├─ set_forward_context (set the active attention backend + KV cache spec)
    ├─ model.forward(input_ids, positions)
    │   │
    │   │ for each layer:
    │   ▼
    │  OPTDecoderLayer (or Llama / future)
    │   │
    │   ├─ input_layernorm
    │   ├─ self_attn (QKV linear → rotary → unified_attention → O linear)
    │   │    │
    │   │    └─ TritonAttentionBackend
    │   │        ├─ triton_reshape_and_cache_flash (write new KV to paged cache)
    │   │        └─ triton_unified_attention       (paged attention decode/prefill)
    │   ├─ post_attn_layernorm
    │   ├─ mlp (gate + up + down, optionally MoE)
    │   └─ residual add
    │
    ├─ sample (Sampler.forward → token ids)
    ▼
ModelRunnerOutput
    │
    ▼
OutputProcessor                         ← my_vllm/v1/engine/output_processor.py
    │
    │ Streaming / async iterator
    ▼
LLMEngine.step()
    │
    ▼
HTTP response (via API server — out of scope)
```

---

## 2. Each layer, with files

### 2.1 LLMEngine

**File**: `my_vllm/entrypoints/llm.py`

The Python-side entry point.  Owns an `EngineCoreClient` (inproc
or busy-loop) and translates `LLM.generate()` calls into typed
`EngineCoreRequest` messages.  Not the focus of engineering
alignment (C151 deleted the OpenAI server; the LLM class is
minimal).

### 2.2 EngineCoreClient

**File**: `my_vllm/v1/engine/core_client.py`

Two implementations: `InprocClient` (direct method call, no
serialization) and `BusyLoopClient` (serializes via msgpack over
a ZMQ pair socket, drives the engine on a daemon thread).  The
busy-loop is what allows the engine to run independently of
LLMEngine — when LLMEngine is on a separate process from
EngineCore (the production deployment), this is the bridge.

C124-C128 covers the busy-loop wiring (see
`docs/snapshots/C124.md` through `C128.md`).

### 2.3 EngineCore

**File**: `my_vllm/v1/engine/core.py`

The scheduler + driver.  Owns the scheduler, the KV cache
manager, and the executor.  `step()` calls `scheduler.schedule()`
→ `executor.execute_model(scheduler_output)` → `output_processor.process_outputs()`.

### 2.4 Scheduler

**File**: `my_vllm/v1/core/sched/scheduler.py`

FCFS + priority + chunked prefill + prefix-cache aware.  The
scheduler is responsible for:

* Picking the next batch of requests (decode requests first,
  then prefill tokens up to `max_num_batched_tokens`).
* Calling the KV cache manager to allocate / extend / drop blocks.
* Emitting a `SchedulerOutput` with the batched tokens + the
  block table per request.

Scheduler states per request: `WAITING` → `RUNNING` → `PREEMPTED`
(or `FINISHED`).  Preemption: copy-then-recompute (drain KV,
re-prefill on next step) — slower than swap-preempt but
requires no host memory.

### 2.5 KVCacheManager

**File**: `my_vllm/v1/core/kv_cache_manager.py`

Owns the BlockPool + implements the prefix-cache lookup
(`BlockHashToBlockMap` from `block_pool.py`).  The dataflow
contract:

* `get_or_alloc_blocks(request_id, num_blocks)` → returns a
  list of physical block ids for the request.
* `free_blocks(request_id)` → returns them to the pool.

Prefix cache: when a new request's tokens match an existing
block hash, the KV cache manager reuses the block (ref_cnt > 0,
so the old request doesn't free it).  Copy-on-Write: when a
request writes to a shared block, it's split (new block
allocated, old block ref_cnt decremented).

### 2.6 BlockPool

**File**: `my_vllm/v1/core/block_pool.py`

Free-block queue + ref-counted block allocation.  O(1)
alloc / free.  Manages physical block IDs (the actual GPU
memory locations).

### 2.7 Executor

**Files**: `my_vllm/v1/executor/uniproc_executor.py`,
`my_vllm/v1/executor/multiproc_executor.py`

The interface between EngineCore (Python) and the workers
(C++/CUDA).  Two implementations:

* **UniProcExecutor**: EngineCore and the worker run in the
  same process.  Calls are direct method invocations.
  Simplest case; what `python run.py` uses by default.

* **MultiprocExecutor** (C150): EngineCore spawns one worker
  subprocess per rank.  Communication via `multiprocessing`
  Queues + RPC.  Multi-GPU without NCCL setup (each rank is a
  separate process with one GPU).

Ray executor was deleted in C151 (out of JD scope).

### 2.8 Worker

**Files**: `my_vllm/v1/worker/gpu_worker.py`,
`my_vllm/v1/worker/worker_base.py`

Per-rank process that owns a GPU.  Responsibilities:

* `init_device()`: allocate the CUDA context, initialize
  NCCL if multi-rank, load the model.
* `execute_model(scheduler_output)`: drives the model forward
  on this rank's slice of the data.
* `collective_rpc(method, *args)`: cross-rank method call
  (used for elastic EP rebalance — see C230).

### 2.9 GPUModelRunner

**File**: `my_vllm/v1/worker/gpu_model_runner.py`

The single most complex file in the codebase.  Owns:

* **InputBatch** (`my_vllm/v1/worker/gpu/input_batch.py`) — the
  GPU-friendly batch state: token tensors, position tensors,
  block-table tensors, sampling metadata.  All slots
  pre-allocated, no per-step allocations.

* **forward_context** — sets the active attention backend + KV
  cache spec via `set_forward_context`.  TritonAttentionBackend
  reads this to know which kernel to launch.

* The unified `execute_model` — handles prefill / decode in
  the same step (no separate `_execute_prefill` /
  `_execute_decode` paths).  Single batch, single forward.

### 2.10 Model (OPT)

**Files**: `my_vllm/model_executor/models/opt.py`

The actual neural network.  Per layer: input_layernorm →
self_attn → residual add → post_attn_layernorm → MLP → residual
add.  Self-attention uses `OPTAttention` (QKV linear → rotary
(not actually — OPT uses learned pos; rotary is for future
Llama integration) → paged attention → O linear).

The MLP is the FFN (gelu + linear).  When MoE is enabled
(C221, future), MLP becomes a routed expert dispatch (FusedMoE
with token alignment).

### 2.11 Attention layer

**File**: `my_vllm/model_executor/layers/attention/attention.py`

Wraps the attention backend.  Today: TritonAttentionBackend
(C211).  In production vLLM there's also FlashAttention,
FlashInfer, xFormers — my-vllm ships the Triton path (the one
we have C211 verified) and a stub for the others.

The attention forward:

1. QKV linear projection (`Linear` layer, optionally
   `QKVParallelLinear` for TP).
2. RoPE (`RotaryEmbedding`, C224 — fused Triton kernel).
3. `triton_reshape_and_cache_flash` — write new K, V to the
   paged KV cache (`block_table` indexed).
4. `triton_unified_attention` — paged attention (decode / prefill
   / mixed chunked).
5. Output linear projection.

### 2.12 Sampler

**File**: `my_vllm/v1/sample/sampler.py`

The Sampler module (`nn.Module`) — takes the logits from the
model forward, applies penalties → temperature → top-k → top-p →
multinomial sampling, returns sampled token IDs.

The sampler is split into:

* `Sampler` (the module that ties everything together).
* `LogitsProcessor` (penalties, temperatures, top-k/top-p) — in
  `my_vllm/v1/sample/logits_processor/`.

### 2.13 OutputProcessor

**File**: `my_vllm/v1/engine/output_processor.py`

Maps sampled token IDs to text (detokenization), tracks per-
request output state, and emits `RequestOutput` objects.  This
is the boundary between "GPU ran a step" and "LLMEngine
returns a result".

---

## 3. Architectural boundaries

The stack has four clean boundaries:

### Boundary 1: EngineCore ↔ Executor

* **Above**: Python (EngineCoreProc in `my_vllm/v1/engine/core.py`)
* **Below**: Same Python (Executor) → workers (Python or C++)
* **Contract**: `SchedulerOutput` (typed) → `ModelRunnerOutput`
  (typed).  No raw tensors cross this boundary in the inproc
  case.

### Boundary 2: Worker ↔ GPUModelRunner

* **Above**: Worker process (Python)
* **Below**: Same Python (GPUModelRunner)
* **Contract**: `execute_model(scheduler_output)` →
  `ModelRunnerOutput`.  Holds the model + KV cache + InputBatch.

### Boundary 3: GPUModelRunner ↔ Model

* **Above**: Python runner
* **Below**: PyTorch model (Python, with Triton/CUDA kernels)
* **Contract**: `forward(input_ids, positions)` → logits.
  This is where `set_forward_context` lives — the runner
  publishes the attention backend + KV cache spec; the model
  reads it via `get_forward_context`.

### Boundary 4: Model ↔ Kernels

* **Above**: PyTorch ops (`torch.nn.functional.linear`,
  `RotaryEmbedding.forward`, `triton_unified_attention`,
  `FusedMoE.forward`).
* **Below**: Triton / CUDA / CUTLASS kernels.
* **Contract**: Tensor shapes + dtypes + custom op registration
  via `my_vllm._custom_ops`.

These boundaries are stable — the engineering-alignment plan
preserves them 1:1 with vLLM.

---

## 4. Performance hotspots

Per step on a typical Llama-7B decode batch (batch=8, seq=2048):

| Hotspot                              | % of step time | Mitigation path          |
|--------------------------------------|----------------|--------------------------|
| `triton_unified_attention`            | ~40%           | better tile shapes, FP8 KV (C211) |
| `triton_reshape_and_cache_flash`      | ~5%            | FP8 storage (C211/C213)  |
| Linear (`q_proj / k_proj / v_proj / o_proj / gate / up / down`) | ~35% | FP8 / INT8 quant (C220) |
| MoE `fused_moe`                       | ~10% (only MoE) | better tile shapes (C221) |
| Sampler                               | ~5%            | speculative decode (C223) |
| KV transfer (PD disagg)              | ~5% (only disagg) | real vLLM has GPUDirect RDMA (C231) |

The four big wins from the C200-C231 era:

* **FP8/INT8 quantization** (C220) — 2× GEMM throughput.
* **Unified paged attention** (C211) — eliminates the prefill /
  decode split; one kernel handles both.
* **NCCL + RDMA bandwidth** (C201) — multi-node scaling stops
  being bandwidth-limited.
* **VMM allocator** (C231) — >50% KV cache savings on Hopper-
  class GPUs serving many small sequences (the elastic-EP
  scale-down case).

---

## 5. Where to start when debugging

| Symptom                                | Look at                                       |
|----------------------------------------|------------------------------------------------|
| Slow decode throughput                 | `triton_unified_attention` profile + `triton_reshape_and_cache_flash` (C211 / C213) |
| Slow prefill                           | Same — prefill is dominated by the GEMM (Linear) + initial KV write |
| Memory blow-up under high concurrency  | `KVCacheManager.get_or_alloc_blocks` + `BlockPool` — check fragmentation + eviction policy |
| Bad multi-node scaling                  | `nccl_benchmark.py` + `nccl-architecture.md` (C201) |
| Wrong outputs                          | `Sampler` + `LogitsProcessor` (C222 was deleted; check vLLM V1 sampler changes) |
| Slow wake-up from sleep (sleep mode)   | `CumemAllocator` + `cumem_allocator.cpp` (C231) |

---

## 6. Where C230 (elastic EP) lives

Elastic EP scale-up / scale-down is *outside* the steady-state
scheduler.  It runs:

1. **Trigger**: rebalance policy (`ThresholdEPLoadPolicy`,
   C230) decides to scale.  Called periodically from the
   scheduler when KV cache utilization crosses the threshold.
2. **Build state machine**: `create_scaling_state_for_decision(vllm_config, decision)`
   returns an `ElasticEPScalingState` (C230).
3. **Drive state machine**: EngineCore driver calls
   `state.progress()` until `is_complete`.  Each progress step
   performs barrier sync, weight transfer, KV memory sync,
   EPLB reshuffle (in production vLLM; my-vllm's `progress()`
   is a no-op advancing the state enum — see C230's limitations).
4. **Engine swap**: after COMPLETE, EngineCore switches to the
   new DP group + resets the scheduler / KV cache manager.

In vLLM production, the state machine's per-step actions are
TCP-store barrier + NCCL collective_rpc.  In my-vllm's
engineering subset, the state machine advances the enum
without doing the actual work — that's the documented
limitation (C230-snapshot § "Engine-side wiring").

---

## 7. Where C231 (VMM allocator) lives

The VMM allocator (`cumem_allocator.cpp`) is *outside* the
request path.  It runs:

1. **At model load time**: EngineCore calls
   `CumemAllocator.use_memory_pool(tag="kv_cache")` so the KV
   cache tensor allocations go through `cuMemCreate` /
   `cuMemMap` (real physical chunks).
2. **At elastic EP scale-down**: EngineCore calls
   `CumemAllocator.sleep(offload_tags=("kv_cache",))` to free
   GPU memory on the engines being removed.  The C extension
   un-maps physical chunks via `cuMemUnmap` + releases them
   via `cuMemRelease`.  (Production vLLM also copies tensor
   contents to pinned host memory here; my-vllm doesn't
   vendor `CudaRTLibrary` yet — see C231-snapshot limitations.)
3. **At elastic EP scale-up**: `wake_up(tags=["kv_cache"])`
   re-allocates + re-maps physical chunks to the original VA
   ranges.

The cumem extension is loaded as `my_vllm.cumem_allocator.so`
(see C231-snapshot); PyTorch's `CUDAPluggableAllocator(lib,
"my_malloc", "my_free")` calls the C entry points via dlsym.

---

## 8. References

- vLLM V1 architecture overview — vLLM blog post "vLLM V1:
  A Major Upgrade to vLLM's Architecture" (Jan 2024).
- vLLM `vllm/v1/engine/` — the engine directory (the
  authoritative reference for the EngineCore ↔ Worker ↔
  Runner boundary).
- vLLM `vllm/v1/worker/gpu_model_runner.py` — the reference
  for `execute_model` (588 lines in vLLM 0.5.x; 7902 in
  current main).
- Each Cxxx snapshot in `docs/snapshots/` — the architecture
  notes for that specific subsystem.

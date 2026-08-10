# AGENTS.md

## Project identity

my-vllm is an **engineering subset** of the vLLM V1 serving runtime. It takes real vLLM as the reference codebase and reproduces its engineering organization, abstraction boundaries, core data structures, execution paths, and GPU Runtime interfaces as faithfully as possible. Features removed are limited to business/ecosystem functionality unrelated to career goals.

**Career target**: AI Systems / AI Infrastructure Engineer → LLM Inference / Distributed Inference.
**Focus**: GPU Performance + Runtime + Distributed Systems.

my-vllm's final form:

```
real vLLM
    │
    ├── career-relevant → reproduce as faithfully as possible
    ├── runtime-relevant → reproduce as faithfully as possible
    └── unrelated to career → delete
             ↓
        my-vLLM（strict engineering subset of vLLM）
```

Every file in `my_vllm/` must correspond to a real file in `vllm/`. No invented patterns or APIs.

**Previous history**: Stages C000→C098 were implemented under a prior "educational simulation" model. That history is preserved (never squash/rebase). The project now enters a new phase of **engineering alignment** (C110+) — see `Engineer-Spec/README.md` for the full specification.

**Current status**: C110 baseline. 6 invented files deleted, file tree now strict subset of vLLM. FlashAttentionBackend operational (SDPA-based).

**Next**: Phase 1.2 per Engineer-Spec — `runtime/` → `entrypoints/` path alignment. 10 P0 skeletons pending thereafter.

## Commands

```bash
# Activate the virtual environment first
source /home/kasw/workspace/gpu-infra/.venv/bin/activate

# Offline inference (V1 engine)
python run.py

# Run all tests
pytest tests/

# Run a single test file or class
pytest tests/test_C031_runtime.py -v
```

There is no CI, no lint, no formatter, no typecheck configured.

## Configuration

`my_vllm/config/` — `VllmConfig` aggregates `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `ParallelConfig`. Defaults:
- model: `facebook/opt-125m`, dtype: `bfloat16`, block_size: `16`, num_kv_blocks: `512`, max_num_seqs: `8`
- enforce_eager: `True`, max_num_batched_tokens: `2048`, eos_token_id: `2`
- TP=1, PP=1 (config-only; distributed infrastructure exists but not yet wired end-to-end)

## Package layout

```
my_vllm/
  config/              # → vllm/config/  VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, ParallelConfig
  distributed/         # → vllm/distributed/  parallel_state.py (GroupCoordinator), elastic_ep/
  model_executor/      # → vllm/model_executor/  model_loader, layers/linear, layers/fused_moe, models/opt
  v1/                  # → vllm/v1/
    engine/            # EngineCore, LLMEngine, AsyncLLM, InputProcessor, OutputProcessor, detokenizer, exceptions
    core/              # BlockPool, KVCacheManager, kv_cache_utils, kv_cache_metrics, kv_connector
      sched/           # Scheduler, SchedulerInterface, SchedulerOutput, request_queue
    worker/            # WorkerBase, Worker, GPUModelRunner
      gpu/             # BlockTables, InputBatch, RequestState, PPHandler, model_states/
    executor/          # Executor ABC, UniProcExecutor, MultiprocExecutor, RayDistributedExecutor, ray_utils
    attention/         # AttentionBackend ABC, CommonAttentionMetadata, selector, backends/flash_attn, backends/ops/paged_attn
    sample/            # Sampler (nn.Module), LogitsProcessor, SamplingMetadata
    metrics/           # stats, loggers, prometheus
    engine/__init__.py # FinishReason, EngineCoreOutput, EngineCoreOutputs
    request.py         # Request (12 statuses), SamplingParams
    outputs.py         # SamplerOutput, ModelRunnerOutput
    serial_utils.py    # MsgpackEncoder/Decoder (binary format)
    utils.py           # ConstantList
    kv_cache_interface.py  # KVCacheSpec hierarchy
  runtime/             # ← Phase 1.2: migrate to entrypoints/ (real vLLM path)
  server/              # ← Phase 1.2: migrate to entrypoints/ (real vLLM path)
  run.py               # CLI entry point
```


**Path mapping is 1:1**: `my_vllm/` → `vllm/` (e.g., `my_vllm/config/vllm.py` → `vllm/config/vllm.py`).

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`. **All new and modified code must be directly extracted from the real vLLM codebase** — class bodies, method logic, data structures, and internal algorithms are taken from vLLM source files. Simplifications (fewer fields, no quantization backends) are principled scope reductions — each field or feature is omitted because the functionality it supports is excluded from scope, not because the architecture is simplified for whimsiness. No invented patterns or APIs. Legacy code from C000–C098 is gradually being replaced under this standard.

The current concrete model is `facebook/opt-125m` (Dense Transformer). The architecture is designed to accommodate Dense, GQA/MQA, and MoE through reusable layers — model count increases should add config/thin adapter layers, not duplicate Runtime.

The **engineering spec** (`Engineer-Spec/README.md`) defines the alignment standard:
- **P0** (10 items): Core skeletons that must be reproduced — Unified Execution Model, Paged KV path, KV Cache hierarchy, Block identity, Scheduler model, Request/RequestState, InputBatch, Attention Backend, Executor/Worker/Runner boundary, Distributed abstraction
- **P1** (5 items): Should be gradually aligned — Dynamic KV init, Model layers, OPT model, Sampler boundary, Metrics
- **Must implement**: Speculative Decoding, EP, CP, KV Connector, at least one Quantization path — these alter GPU execution, KV Cache, batching, scheduler, communication, or model parallelism and cannot remain NoOp/ABC
- **Architecture + limited implementation**: LoRA, Mamba, Multimodal, Structured Output — understand Runtime impact; implement one working path where necessary
- **Excluded from core code**: K8s, Ray, Autoscaling — understand theoretically, but not in `my_vllm/`; may exist as a separate `deployment/` experiment

## 4-Level alignment framework

| Level | Standard |
|---|---|
| **Level 1: File tree** | `my-vLLM files ⊆ vLLM files`. Every file must correspond to a real vLLM file. Delete invented files with no counterpart. |
| **Level 2: Objects & interfaces** | Class names, method names, responsibility, data ownership match vLLM. Not 100% parameter identical, but design intent must match. |
| **Level 3: Core logic** | `Scheduler → KV allocation → Input preparation → Unified batch → Model forward → Paged attention → Sampler → Output` — this chain must be faithfully reproduced. |
| **Level 4: Implementation tech** | `GPU memory → CUDA/Triton → Paged KV → Batch execution → CUDA Graph / async → NCCL → Multi-GPU` |

## Before EVERY code modification:

1. Read `Engineer-Spec/README.md` for the target alignment standard
2. Check the Implementation scope tiers — confirm the feature is in P0/P1/Must Implement scope, not Architecture-only or Excluded
3. Read the real vLLM source at `/home/kasw/文档/GitHub/GPU-Project/vllm` for API verification
4. Confirm the file exists in real vLLM's file tree
5. Match vLLM class names, method names, dataclass field names exactly
6. Never invent API names or file paths
7. Verify `python run.py` or `pytest` passes before committing

## KV Cache dtype abstractions

See `Engineer-Spec/README.md` § KV Cache dtype — FP16/BF16 baseline, FP8 quantization/dequantization, KVCacheSpec abstraction chain.

## Model architecture priorities

See `Engineer-Spec/README.md` § Model architecture — Dense/GQA/MoE are must-master; model count increases should add config, not duplicate Runtime.

## Model loading: checkpoint → GPU

See `Engineer-Spec/README.md` § Model loading — init/load separation, meta device, weight mapping, parallel linear, TP-aware shard.  Current `model_loader.py` uses `from_pretrained()` + `.to("cuda")`, bypassing the Runtime boundary.  P1-2/P1-3 close this gap.

## Commit workflow (per `Principle/Commit-Standard.md`)

1. One commit = one P0/P1 alignment item. Format: `Cxxx: Module: Description` (C110+)
2. Each commit must be runnable (`python run.py` or `pytest` must pass)
3. Create `docs/snapshots/Cxxx.md` with required sections:
   Goal, New Features, File Tree, Runtime Flow (mermaid),
   Current Capability, Current Limitation, Next Stage
4. Tag with `Cxxx-description` (e.g. `git tag C033-kernel`)
5. Never modify earlier stages' code; if a fix is needed, create a new fix commit
6. See `Principle/Commit-Standard.md` for the full rules
7. After commit, output: commit message, file diff summary, file tree,
   architecture diagram (mermaid), capabilities, limitations,
   benchmark (skip if no performance code — see Commit-Standard rule 7)
8. Update AGENTS.md in the same commit if module or architecture facts changed
9. **Stop** — await human confirmation before proceeding to next stage

## Key architecture facts

- EngineCore `step()` = `schedule() → execute_model() → update_from_output()` → EngineCoreOutputs
- LLMEngine `step()` = EngineCore.step() → EngineCoreOutputs → OutputProcessor.process_outputs()
- SchedulerInterface ABC: schedule() → SchedulerOutput, update_from_output(), finish_requests()
- Model loading chain: `Executor._init_executor() → Worker.init_device() → GPUModelRunner.load_model()`
- KV cache is physical-block-based (BlockPool + FreeKVCacheBlockQueue, doubly-linked, O(1) alloc)
- Prefix caching: BlockHashToBlockMap (hash→list[block]), free_blocks() with ref_cnt cascading release, CoW
- Scheduler: FCFS + Priority scheduling + chunked-prefill + prefix-cache aware + resource model + Preemption
- Request PREEMPTED status supports resume/retry
- Chunked prefill splits long prompts by `token_budget` at the Scheduler level
- FlashAttentionBackend: SDPA kernel with paged KV interface (kv_cache + block_table + seq_lens)
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- TP=2: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear
- Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization

## Model dependency

Hardcoded to `facebook/opt-125m`. Tests load from the HuggingFace cache. The model must be available for any run or test.

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

**Previous history**: Stages C000→C098 were implemented under a prior "educational simulation" model. That history is preserved (never squash/rebase). The project now enters a new phase of **engineering alignment** — see `Engineer-Spec/README.md` for the full specification.

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
  distributed/         # → vllm/distributed/  parallel_state.py (GroupCoordinator)
  model_executor/      # → vllm/model_executor/  model_loader, layers/linear, layers/fused_moe, models/opt
  v1/                  # → vllm/v1/
    engine/            # EngineCore, LLMEngine, AsyncLLM, InputProcessor, OutputProcessor, detokenizer, exceptions
    core/              # BlockPool, KVCacheManager, kv_cache_utils, kv_cache_metrics, kv_connector
      sched/           # Scheduler, SchedulerInterface, SchedulerOutput, request_queue
    worker/            # WorkerBase, Worker, GPUModelRunner
      gpu/             # BlockTables, InputBatch, RequestState, PPHandler, model_states/
    executor/          # Executor ABC, UniProcExecutor, MultiprocExecutor, RayDistributedExecutor, ray_utils
    attention/         # AttentionBackend ABC, CommonAttentionMetadata, selector, backends/torch_attn, backends/ops/paged_attn
    sample/            # Sampler (nn.Module), LogitsProcessor, SamplingMetadata
    metrics/           # collector, stats, loggers, prometheus
    engine/__init__.py # FinishReason, EngineCoreOutput, EngineCoreOutputs
    request.py         # Request (12 statuses), SamplingParams
    outputs.py         # SamplerOutput, ModelRunnerOutput
    serial_utils.py    # MsgpackEncoder/Decoder (binary format)
    utils.py           # ConstantList
    kv_cache_interface.py  # KVCacheSpec hierarchy
  runtime/             # tokenizer.py, main.py (entry bridge)
  server/              # api_server.py (HTTP entry point)
  run.py               # CLI entry point
```

**Path mapping is 1:1**: `my_vllm/` → `vllm/` (e.g., `my_vllm/config/vllm.py` → `vllm/config/vllm.py`).

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`. **All code implementation is directly extracted from the real vLLM codebase** — class bodies, method logic, data structures, and internal algorithms are taken from vLLM source files, with educational simplifications applied (fewer fields, single model, no quantization) but no invented patterns or APIs.

The **engineering spec** (`Engineer-Spec/README.md`) defines the alignment standard:
- **P0** (10 items): Core skeletons that must be reproduced — Unified Execution Model, Paged KV path, KV Cache hierarchy, Block identity, Scheduler model, Request/RequestState, InputBatch, Attention Backend, Executor/Worker/Runner boundary, Distributed abstraction
- **P1** (5 items): Should be gradually aligned — Dynamic KV init, Model layers, OPT model, Sampler boundary, Metrics
- **Interface-only**: KV Connector, LoRA, Quantization, Spec Decode, Structured Output, Multimodal, Mamba, EP, CP — keep as ABC/NoOp stubs for architectural completeness
- **Excluded**: Multimodal, Encoder-Decoder, Mamba, all quantization backends, non-OPT models, platform backends, K8s, Ray full, Autoscaling

## 4-Level alignment framework

| Level | Standard |
|---|---|
| **Level 1: File tree** | `my-vLLM files ⊆ vLLM files`. Every file must correspond to a real vLLM file. Delete invented files with no counterpart. |
| **Level 2: Objects & interfaces** | Class names, method names, responsibility, data ownership match vLLM. Not 100% parameter identical, but design intent must match. |
| **Level 3: Core logic** | `Scheduler → KV allocation → Input preparation → Unified batch → Model forward → Paged attention → Sampler → Output` — this chain must be faithfully reproduced. |
| **Level 4: Implementation tech** | `GPU memory → CUDA/Triton → Paged KV → Batch execution → CUDA Graph / async → NCCL → Multi-GPU` |

## Before EVERY code modification:

1. Read `Engineer-Spec/README.md` for the target alignment standard
2. Read the real vLLM source at `/home/kasw/文档/GitHub/GPU-Project/vllm` for API verification
3. Confirm the file exists in real vLLM's file tree
4. Match vLLM class names, method names, dataclass field names exactly
5. Never invent API names or file paths
6. Verify `python run.py` or `pytest` passes before committing

## Commit workflow

1. One commit = one P0/P1 alignment item. Format: `Cxxx: Module: Description` (C110+)
2. Create `docs/snapshots/Cxxx.md` with: Goal, Changes, Runtime Flow (mermaid), Capabilities, Limitations
3. Tag with `Cxxx-description`
4. After commit, output: commit message, file diff summary, file tree, architecture diagram, capabilities, limitations
5. **Stop** — await human confirmation before proceeding to next item

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
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- TP=2: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear
- Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization

## Model dependency

Hardcoded to `facebook/opt-125m`. Tests load from the HuggingFace cache. The model must be available for any run or test.

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

**Previous history**: Stages C000→C098 were implemented under a prior "educational simulation" model. That history is preserved (never squash/rebase). The project now enters a new phase of **engineering alignment** (C110+) — see `Prompt/Engineer-Spec/README.md` for the full specification.

**Current status**: C150. All alignment phases + LoRA专项 complete (55/55 items closed); follow-up wire-ups 4/4 done. KV cache hierarchy mirrors vLLM (Manager → Coordinator → SingleTypeKVCacheManager → FullAttentionManager). KV transfer migrated to vLLM-canonical `distributed/.../v1/base.py` with `KVConnectorFactory` + `ExampleConnector`. Scheduler has 3-level pipeline + utils + AsyncScheduler skeleton. FP8 KV cache end-to-end: `kv_cache_dtype=fp8_e4m3` allocates the cache in `float8_e4m3fn`, `reshape_and_cache_flash` takes the FP8 path with per-tensor K/V scales (C149). spec_decode / parallel_sampling / logprobs skeletons exist. Multi-process NCCL end-to-end: `init_distributed_environment` + canonical `Worker.init_device` sequence + `MultiprocExecutor._init_executor` exports `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE` env vars (C150). LoRA Dense-Linear forward path end-to-end runnable (verified: LoRARequest → WorkerManager → ModelManager → LoRAModel → LoRA layer → forward). `update_lora_modules()` actually wraps OPTForCausalLM targets (C147). `GPUModelRunner` now inherits `LoRAModelRunnerMixin` and `_set_active_lora()` activates the per-step adapter inside `execute_model` (C148). Invented-file count: 0. See `docs/snapshots/C150.md` for closure write-up.

**Next**: All alignment phases (55/55) + all follow-up wire-ups (4/4) complete. Production-side concerns next: (a) `deployment/` — k8s manifests, helm charts, docker entrypoint; (b) entrypoints business surface — OpenAI-compatible `/v1/chat/completions` server; (c) `monitor` — prometheus metrics, structured logging, tracing.

## Scope decision (JD-driven)

The final form is a **minimal clean subset mapped to the job description (JD)**, not a scaled-down vLLM. A module is kept only if it serves one of the JD keywords: 性能/成本/稳定性, PD 分离, 分布式 KV Cache, 异构推理, 弹性计算, GPU/CUDA, 分布式系统 (+ 网络/RDMA). Clean removal = delete the file **and** its imports, config fields, and all references — no NoOp/ABC placeholders, no dangling imports. Never keep something just because vLLM has it.

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
<<<<<<< Updated upstream
  distributed/         # → vllm/distributed/  parallel_state.py (GroupCoordinator), kv_transfer/ (KVConnectorBase_V1, factory, nixl), elastic_ep/
  model_executor/      # → vllm/model_executor/  model_loader, layers/linear, layers/fused_moe, models/opt
=======
  distributed/         # → vllm/distributed/  parallel_state.py (GroupCoordinator), elastic_ep/, kv_transfer/
  model_executor/      # → vllm/model_executor/  model_loader, custom_op, layers/, models/
    layers/            # linear (Column/Row/QKV), attention/attention.py, attention_layer_base.py,
                       #   activation, layernorm, logits_processor, vocab_parallel_embedding, fused_moe/, quantization/
    models/            # opt.py (Dense), llama.py (GQA), deepseek_v2.py (DeepSeek V2/V3 MoE+MLA)
  platforms/           # → vllm/platforms/  Platform interface, attn backend selection
  profiler/            # → vllm/profiler/  WorkerProfiler, TorchProfiler, CUDAGraphProfiler
  lora/ envs/ logger/ utils/
  forward_context.py   # → vllm/forward_context.py  metadata channel (attn_metadata/slot_mapping/layers)
  entrypoints/         # → vllm/entrypoints/  llm.py, openai/api_server.py
>>>>>>> Stashed changes
  v1/                  # → vllm/v1/
    engine/            # EngineCore, EngineCoreClient (InprocClient / BusyLoopClient), LLMEngine,
                      # AsyncLLM, InputProcessor, OutputProcessor, detokenizer, exceptions
    core/              # BlockPool, KVCacheManager, kv_cache_utils, kv_cache_metrics
      sched/           # Scheduler, SchedulerInterface, SchedulerOutput, request_queue
    worker/            # WorkerBase, Worker, GPUModelRunner
      gpu/             # BlockTables, InputBatch, RequestState, PPHandler (pp_utils.py), model_states/
    executor/          # Executor ABC, UniProcExecutor, MultiprocExecutor, RayDistributedExecutor, ray_utils
<<<<<<< Updated upstream
    attention/         # AttentionBackend ABC, CommonAttentionMetadata, selector, backends/flash_attn, ops/paged_attn, ops/triton_*
=======
    attention/         # AttentionBackend ABC, CommonAttentionMetadata, selector (→ Triton),
                       #   backends/triton_attn, backends/flash_attn (fallback),
                       #   ops/triton_unified_attention, ops/triton_reshape_and_cache_flash, ops/triton_slot_mapping
>>>>>>> Stashed changes
    sample/            # Sampler (nn.Module), LogitsProcessor, SamplingMetadata
    metrics/           # stats, loggers, prometheus
    spec_decode/       # metadata (stub)
    engine/__init__.py # FinishReason, EngineCoreOutput, EngineCoreOutputs
    request.py         # Request (12 statuses), SamplingParams
    outputs.py         # SamplerOutput, ModelRunnerOutput
    serial_utils.py    # MsgpackEncoder/Decoder (binary format)
    utils.py           # ConstantList
    kv_cache_interface.py  # KVCacheSpec hierarchy
  _custom_ops.py       # CUDA op dispatch (csrc/cache_kernels.cu → _C.reshape_and_cache)

# Repo root (outside my_vllm/ package)
run.py                 # CLI entry point
csrc/                  # C++/CUDA sources (cache_kernels.cu, quantization/)
setup.py               # build _C extension
examples/              # run.py, benchmark.py, server.py
LOOP-PLAN.md           # C124→C128 plan (busy loop + EngineCoreClient)
```


**Path mapping is 1:1**: `my_vllm/` → `vllm/` (e.g., `my_vllm/config/vllm.py` → `vllm/config/vllm.py`).

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`. **All new and modified code must be directly extracted from the real vLLM codebase** — class bodies, method logic, data structures, and internal algorithms are taken from vLLM source files. Simplifications (fewer fields, no quantization backends) are principled scope reductions — each field or feature is omitted because the functionality it supports is excluded from scope, not because the architecture is simplified for whimsiness. No invented patterns or APIs. Legacy code from C000–C098 is gradually being replaced under this standard.

The current concrete model is `facebook/opt-125m` (Dense Transformer). The architecture is designed to accommodate Dense, GQA/MQA, and MoE through reusable layers — the three model families are OPT (Dense), Llama (GQA), DeepSeek V2/V3 (MoE + MLA). Model count increases should add config/thin adapter layers, not duplicate Runtime.

The **engineering spec** (`Prompt/Engineer-Spec/README.md`) defines the alignment standard:
- **P0** (10 items): Core skeletons that must be reproduced — Unified Execution Model, Paged KV path, KV Cache hierarchy, Block identity, Scheduler model, Request/RequestState, InputBatch, Attention Backend, Executor/Worker/Runner boundary, Distributed abstraction
- **P1** (5 items): Should be gradually aligned — Dynamic KV init, Model layers, OPT model, Sampler boundary, Metrics
- **Must implement**: Speculative Decoding, EP, CP, KV Connector, at least one Quantization path — these alter GPU execution, KV Cache, batching, scheduler, communication, or model parallelism and cannot remain NoOp/ABC
- **Keep (career-relevant)**: LoRA — one working Dense-Linear path
- **Delete (no JD match)**: Mamba/SSM, Multimodal, Structured Output, multi-tenant, pooler/embedding/rerank, encoder-decoder, non-OPT models, non-CUDA platforms, non-FP8 quantization backends, entrypoints business surface — remove cleanly, no dangling imports
- **Excluded from core code**: K8s, Ray, Autoscaling — understand theoretically, but not in `my_vllm/`; may exist as a separate `deployment/` experiment

## 4-Level alignment framework

| Level | Standard |
|---|---|
| **Level 1: File tree** | `my-vLLM files ⊆ vLLM files`. Every file must correspond to a real vLLM file. Delete invented files with no counterpart. |
| **Level 2: Objects & interfaces** | Class names, method names, responsibility, data ownership match vLLM. Not 100% parameter identical, but design intent must match. |
| **Level 3: Core logic** | `Scheduler → KV allocation → Input preparation → Unified batch → Model forward → Paged attention → Sampler → Output` — this chain must be faithfully reproduced. |
| **Level 4: Implementation tech** | `GPU memory → CUDA/Triton → Paged KV → Batch execution → CUDA Graph / async → NCCL → Multi-GPU` |

## Before EVERY code modification:

<<<<<<< Updated upstream
1. Read `Engineer-Spec/README.md` for the target alignment standard
2. Check `docs/alignment-plan.md` (keep/delete matrix + reference `vllm/...` path) and the Implementation scope tiers — confirm the feature is in P0/P1/Must Implement/Keep scope, not Delete or Excluded
=======
1. Read `Prompt/Engineer-Spec/README.md` for the target alignment standard
2. Check the Implementation scope tiers — confirm the feature is in P0/P1/Must Implement scope, not Architecture-only or Excluded
>>>>>>> Stashed changes
3. Read the real vLLM source at `/home/kasw/文档/GitHub/GPU-Project/vllm` for API verification
4. Confirm the file exists in real vLLM's file tree
5. Match vLLM class names, method names, dataclass field names exactly
6. Never invent API names or file paths
7. Verify `python run.py` or `pytest` passes before committing

## KV Cache dtype abstractions

See `Prompt/Engineer-Spec/README.md` § KV Cache dtype — FP16/BF16 baseline, FP8 quantization/dequantization, KVCacheSpec abstraction chain.

## Model architecture priorities

See `Prompt/Engineer-Spec/README.md` § Model architecture — Dense/GQA/MoE are must-master; model count increases should add config, not duplicate Runtime.

## Model loading: checkpoint → GPU

See `Prompt/Engineer-Spec/README.md` § Model loading — init/load separation, meta device, weight mapping, parallel linear, TP-aware shard.  `my_vllm/model_executor/model_loader.py` now constructs the vLLM-native `OPTForCausalLM` and loads weights via `load_weights()` from an HF checkpoint (still `from_pretrained` for the weight source, not meta-device + streaming).

## Commit workflow (per `Prompt/Principle/Commit-Standard.md`)

1. One commit = one P0/P1 alignment item. Format: `Cxxx: Module: Description` (C110+)
2. Each commit must be runnable (`python run.py` or `pytest` must pass)
3. Create `docs/snapshots/Cxxx.md` with required sections:
   Goal, New Features, File Tree, Runtime Flow (mermaid),
   Current Capability, Current Limitation, Next Stage
4. Tag with `Cxxx-description` (e.g. `git tag C033-kernel`)
5. Never modify earlier stages' code; if a fix is needed, create a new fix commit
6. See `Prompt/Principle/Commit-Standard.md` for the full rules
7. After commit, output: commit message, file diff summary, file tree,
   architecture diagram (mermaid), capabilities, limitations,
   benchmark (skip if no performance code — see Commit-Standard rule 7)
8. Update AGENTS.md in the same commit if module or architecture facts changed
9. **Stop** — await human confirmation before proceeding to next stage

## Key architecture facts

- EngineCore `step()` = `schedule() → execute_model() → update_from_output()` → EngineCoreOutputs
- EngineCore `run_busy_loop()` = `while _handle_shutdown(): _process_input_queue() → _process_engine_step() → output_queue.put_nowait()` (C124); inputs come via `input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]]` with `{ADD, ABORT, WAKEUP}`; shutdown state machine `EngineShutdownState = {RUNNING, REQUESTED, SHUTTING_DOWN}`
<<<<<<< Updated upstream
- LLMEngine `step()` = `EngineCoreClient.get_output() → EngineCoreOutputs → OutputProcessor.process_outputs()`; LLMEngine holds an `EngineCoreClient` (`busy_loop=False` → `InprocClient` direct-step, `busy_loop=True` → `BusyLoopClient` daemon thread); `running_requests()` accessor replaces direct `scheduler.running` access
=======
- LLMEngine `step()` = EngineCore.step() → EngineCoreOutputs → OutputProcessor.process_outputs() (still caller-driven; EngineCoreClient wiring lands at C125)
>>>>>>> Stashed changes
- SchedulerInterface ABC: schedule() → SchedulerOutput, update_from_output(), finish_requests()
- Model loading chain: `Executor._init_executor() → Worker.init_device() → GPUModelRunner.load_model()`
- Unified `execute_model`: `update_states → prepare_inputs → prepare_attn → set_forward_context → model.forward → sample → postprocess` (single batch, no prefill/decode split)
- Model is vLLM-native `OPTForCausalLM` (custom layers), forward is `model(input_ids, positions)` — no HF `past_key_values`
- Attention: `set_forward_context(attn_dict)` → `Attention` layer → `TritonAttentionBackend` (`triton_reshape_and_cache_flash` + `unified_attention` + `triton_slot_mapping`)
- KV cache: HND-packed `(num_layers, num_blocks, num_kv_heads, block_size, 2*head_dim)`; block-major; BlockPool + FreeKVCacheBlockQueue O(1) alloc
- CUDA graph: `capture_cudagraph()` captures decode graphs per batch size; `_run_with_cudagraph()` replays decode-only steps
- Prefix caching: BlockHashToBlockMap (hash→list[block]), free_blocks() with ref_cnt cascading release, CoW
- Scheduler: FCFS + Priority scheduling + chunked-prefill + prefix-cache aware + resource model + Preemption
- Request PREEMPTED status supports resume/retry
- Chunked prefill splits long prompts by `token_budget` at the Scheduler level
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- TP=2: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear
- Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization

## Model dependency

Model is `facebook/opt-125m`, loaded from a local checkpoint (`my_vllm/model_executor/model_loader.py` + `entrypoints/llm.py` both default `_LOCAL_MODEL_PATH = "/home/kasw/文档/GitHub/cpu-vllm/opt-125m"`). The model must be available for any run or test.

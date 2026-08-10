# AGENTS.md

## Project identity

This is an **educational simulation** of the vLLM V1 serving runtime. Each Git commit represents one architectural evolution stage (C001→C098 planned), in sequence. The Git history itself is the curriculum — **never squash or rebase**. Every commit has a matching `docs/snapshots/Cxxx.md` and a Git tag.

**Current baseline**: C032 (`C032-cleanup` tag). C033 and C034 were implemented on an unmerged branch and are **abandoned (废案)**. 

**Current status**: All 24 P0 stages (C035→C098) implemented across 4 phases:
- Phase 1 (single-GPU production): C035, C036, C037, C038, C051, C052, C054c, C048, C054, C050, C069 — complete
- Phase 2 (distributed runtime): C083, C080, C081, C082, C084, C085, C086 — complete
- Phase 3 (cluster + KV transfer): C091, C092, C093, C102 — complete
- Phase 4 (convergence): C107, C098 — complete
- 13 P1 stages intentionally deferred; 61 P2 stages intentionally skipped.

**Logic alignment** (post-C098): Internal implementation logic has been audited and aligned with real vLLM across 3 commits:
1. VllmConfig propagation + output structures + types (config flows through all constructors, dict→list index alignment, FinishReason, MsgpackEncoder, directory migration)
2. step() pipeline restructure (EngineCoreOutputs list, Scheduler EOS/length internal, OutputProcessor detokenization-only, EngineCore/LLMEngine separation)
3. BlockPool fixes + unified batching + PagedAttention interface (touch() crash fix, hash eviction, free_blocks eviction ordering, __lt__ FCFS ordering, max_cache_hit_length, unified _update_states→prefill→decode flow, torch_attn paged KV interface)

**Next goal**: Make both code implementation logic and file tree architecture a proper subset of `/home/kasw/文档/GitHub/GPU-Project/vllm`. Every file in `my_vllm/` should correspond to a real file in `vllm/`, with matching API names, method signatures, data structures, and internal algorithm logic. Educational simplifications are permitted for reduced scope (fewer fields, single model, no quantization) but no invented patterns or APIs.

## Commands

```bash
# Activate the virtual environment first
source /home/kasw/workspace/gpu-infra/.venv/bin/activate

# Offline inference (V1 engine)
python run.py

# Only the V1 attention backend works — PyTorch SDPA (torch_attn.py)
# FlashAttention and CUDA graphs are stubs. enforce_eager=True always.

# Run all tests
pytest tests/

# Run a single test file or class
pytest tests/test_C031_runtime.py -v
pytest tests/test_C031_runtime.py::TestC031FileCompleteness -v
```

There is no CI, no lint, no formatter, no typecheck configured.

## Package layout

**Active code lives in `my_vllm/`.** Every file path maps to a real file in `/home/kasw/文档/GitHub/GPU-Project/vllm/`. Config, distributed, and model_executor are at top level (matching vLLM), not nested under v1/.

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

No legacy stub directories remain. All code goes under `my_vllm/` following vLLM's directory structure.

**Path mapping is now 1:1**: `my_vllm/` → `vllm/` (e.g., `my_vllm/config/vllm.py` → `vllm/config/vllm.py`).

## Configuration

`my_vllm/config/` — `VllmConfig` aggregates `ModelConfig`, `CacheConfig`, `SchedulerConfig`, `ParallelConfig`. Defaults:
- model: `facebook/opt-125m`, dtype: `bfloat16`, block_size: `16`, num_kv_blocks: `512`, max_num_seqs: `8`
- enforce_eager: `True`, max_num_batched_tokens: `2048`, eos_token_id: `2`
- `v1/config.py` re-exports `EngineConfig` (VllmConfig alias) for backward compat.

## Design documents (Principle/)

When implementing a new stage CXXX, **first read the corresponding Principle design doc**:

| Stage range | Design doc |
|---|---|
| C000–C014 | `Principle/C000-C014/C000-C014-Do-to-my-vllm.md`, `C014-Final.md` |
| C015–C031 | `Principle/C015-C031/C015-C031-Do-to-my-vllm.md`, `C031-Final.md` |
| C032 | `Principle/C032/C032.md` |
| C033–C050 | `Principle/C033-C050/C033-C050.md`, `C050-Final.md` |
| C051–C070 | `Principle/C051-C070/C051-C070.md`, `C070-Final.md` |
| C071–C100 | `Principle/C071-C100/C070-C100.md`, `C100-Final.md` |
| C101–C120 | `Principle/C101-C120/C101-C120.md` |
| C121–C140 | `Principle/C121-C140/C121-C140.md` |

Each design doc specifies: industrial motivation, existing code baseline, exact file changes, acceptance criteria, and what NOT to change. This is the authoritative spec for that stage.

**Career-focused implementation plan**: `Base-on-C032-Focus-Career/README.md` overrides the default ordering within each group. Stages are classified:

| Priority | Meaning | Count |
|---|---|---|
| **P0** | Must implement — full code, runnable commit | 24 stages |
| **P1** | Lightweight implementation or analysis doc | 13 stages |
| **P2** | Intentionally skipped (see plan for rationale) | 61 stages |

When implementing a new stage, first check `Base-on-C032-Focus-Career/README.md` for its priority and recommended execution order. P2 stages (including C033, C034, C043-A/B/C, C058-C062, C121-C140) are deferred indefinitely — do NOT implement them unless the career plan is revised.

**Implementation specs** (`Base-on-C032-Focus-Career/`) — self-contained, extracted from Principle/ for each P0/P1 stage:

| Phase | File | Stages | Content |
|---|---|---|---|
| 1: 单 GPU 生产级 | `Phase1-Single-GPU.md` | C035, C036, C037, C038, C051, C052, C054c, C048, C054, C050, C069 | 11 P0 stages: BlockHashToBlockMap, SchedulerInterface ABC, Priority scheduling, Preemption, KVCacheMetricsCollector, ColumnParallel/RowParallel, VllmConfig |
| 2: 分布式 Runtime + MultiprocExecutor | `Phase2-Distributed.md` | C083, C080, C081, C082, C084, C085, C086 | 7 P0 stages: initialize_model_parallel + GroupCoordinator, MultiprocExecutor, FusedMoE + FusedMoERouter, EP All-to-All, PPHandler, Ring Attention, Multi-GPU integration |
| 3: 集群 + KV 传输 | `Phase3-Cluster-KV.md` | C091, C092, C093, C102 | 4 P0 stages: RayDistributedExecutor, K8s deployment, AutoscalingController, KVConnectorBase_V1 (start_load_kv/save_kv_layer) |
| 4: 收敛验证 | `Phase4-Convergence.md` | C107, C098 | 2 P0 stages |
| 选修 | `P1-Optional.md` | C039-C042, C053, C054b, C068, C070, C100-C101, C103, C118, C120 | 13 P1 stages |

Each spec includes: goal, files to modify/create, core implementation skeleton, industrial motivation, dependencies, verification, and AGENTS.md update. **Use these as the primary implementation reference** — consult Principle/ only for deeper context.

## Reference codebase

The real vLLM V1 source lives at `/home/kasw/文档/GitHub/GPU-Project/vllm`.
The **Phase implementation spec** for each stage determines which vLLM APIs to extract:

- **P0 stages**: match vLLM class names, method names, dataclass fields, and file paths as closely as practical. Educational simplifications (fewer fields, smaller scope) are permitted, but **no invented API names**.
- **P1 stages**: implement simplified/educational versions or write analysis docs only.
- **P2 stages**: intentionally skipped — do NOT implement.

**Path mapping is 1:1**: `my_vllm/` directory structure mirrors `vllm/` directly (e.g., `my_vllm/config/` → `vllm/config/`, `my_vllm/v1/engine/` → `vllm/v1/engine/`).

**Implementation alignment**: P0 阶段的内部实现逻辑（方法体、调用链、事件通知模式、引用计数语义）已逐模块对照真实 vLLM 源码进行审计和对齐：
- Config 传播链：VllmConfig 作为单一对象贯穿所有构造函数
- step() 管道：EngineCore.step() → EngineCoreOutputs → LLMEngine → OutputProcessor
- KV Cache：BlockPool touch/get_new_blocks/free_blocks 已修复崩溃和静默数据损坏
- Scheduler：EOS/length 检测内聚在 update_from_output()
- OutputProcessor：仅做 detokenization，不再重复检测 EOS
- Unified batching：_update_states → _execute_prefill → _execute_decode 统一流程
- PagedAttention：torch_attn 接口接受 kv_cache + block_table + seq_lens

## Before EVERY code modification:

- Re-read `Principle/Commit-Standard.md`
- Read the matching phase implementation spec in `Base-on-C032-Focus-Career/` for the target stage
- Check `Base-on-C032-Focus-Career/README.md` for the stage's priority (P0/P1/P2)
- If P2: **do NOT implement** — skip to next P0 or P1 stage
- For deeper context, consult the matching Principle design doc
- Reference `/home/kasw/文档/GitHub/GPU-Project/vllm` at the depth specified
  in `## Reference codebase` for this stage's range
- If the doc contains an `## AGENTS.md update` section, apply those changes to AGENTS.md in the same commit
- Verify the planned change complies with both
- Only then modify code

## Commit workflow (when adding new stages)

1. One commit = one stage. Format: `Cxxx: Module: Description`
2. Each commit must be runnable (`python run.py` or `pytest` must pass)
3. Create `docs/snapshots/Cxxx.md` with required sections:
   Goal, New Features, File Tree, Runtime Flow (mermaid),
   Current Capability, Current Limitation, Next Stage
4. Tag with `Cxxx-description` (e.g. `git tag C033-kernel`)
5. Never modify earlier stages' code; if a fix is needed, create a new fix commit
6. See `Principle/Commit-Standard.md` for the full rules
7. Read the matching phase implementation spec in `Base-on-C032-Focus-Career/` and the corresponding `Principle/CXXX-CYYY/` design doc for stage-specific requirements
8. If the design doc has `## AGENTS.md update`, apply those changes and include AGENTS.md in the commit
9. After commit, output: commit message, file diff summary, file tree,
   architecture diagram (mermaid), capabilities, limitations,
   benchmark (skip if no performance code — see Commit-Standard rule 7)

## Implementation workflow (per-commit execution sequence)

When implementing a new stage, follow this exact sequence:

1. **Read specs (3 sources)**
   - `Base-on-C032-Focus-Career/PhaseX-*.md` — primary implementation spec
   - `Principle/CXXX-CYYY/CXXX-CYYY.md` — deeper industrial context
   - `/home/kasw/文档/GitHub/GPU-Project/vllm` — real vLLM source for API verification

2. **Implement**
   - Create/modify only the files listed in the Phase spec
   - Match vLLM class names, method names, dataclass field names exactly
   - Never invent API names (e.g. no `MoELayer`, use `FusedMoE`)

3. **Verify runnability**
   - `python run.py` or `pytest tests/` must pass before committing

4. **Commit**
   - `git add` only intended files — never commit secrets, .venv, or __pycache__
   - `git commit -m "Cxxx: Module: Description"`
   - Create `docs/snapshots/Cxxx.md` in the same commit

5. **Tag**
   - `git tag Cxxx-description`

6. **Output summary** (in terminal after each commit)
   - Commit message
   - Modified/added file list
   - Current file tree
   - Runtime flow (mermaid diagram)
   - Current capabilities and limitations
   - Benchmark results (skip if no performance code change)

7. **Stop** — await human confirmation before proceeding to next stage.
   The project never auto-advances to the next commit without explicit instruction.

## Key architecture facts

- EngineCore `step()` = `schedule() → execute_model() → update_from_output()` → EngineCoreOutputs
- LLMEngine `step()` = EngineCore.step() → EngineCoreOutputs → OutputProcessor.process_outputs()
- SchedulerInterface ABC: schedule() → SchedulerOutput, update_from_output(), finish_requests()
- Model loading chain: `Executor._init_executor() → Worker.init_device() → GPUModelRunner.load_model()` — matching vLLM's deep loading architecture (model is loaded inside the runner, not passed in from the entry point)
- MultiprocExecutor: Worker Process per GPU, 替代 UniProcExecutor 支持多进程并行
- MoE: FusedMoERouter (TopK gating) + FusedMoE (expert ensemble) + ExpertMLP, auxiliary load balancing loss
- KV cache is physical-block-based (BlockPool + FreeKVCacheBlockQueue, doubly-linked, O(1) alloc)
- Prefix caching uses BlockHashToBlockMap (hash→list[block]), free_blocks() with ref_cnt cascading release, GC 驱逐逻辑, CoW: shared blocks copied on write (ref_cnt > 1 → copy → decrement → private block), KVCacheBlockCopy via SchedulerOutput.kv_cache_block_copies
- Scheduler: FCFS + Priority scheduling + chunked-prefill + prefix-cache aware + resource model (GPU budget, KV budget, token budget) + Preemption
- Request PREEMPTED 状态支持 resume/retry
- Chunked prefill splits long prompts by `token_budget` at the Scheduler level
- PagedAttention ops are pure PyTorch (`reshape_and_cache_pytorch`, `gather_cached_kv_pytorch`)
- Sampler is an `nn.Module` subclass with penalties → temperature → top-k → top-p → sampling
- Tensor parallel: TP=2 industrial: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear 通过 GroupCoordinator 通信
- KVCacheSpec hierarchy: FullAttentionSpec (packed K/V, head_size_v), SlidingWindowSpec (swa window), MLAAttentionSpec (cache_dtype_str), 聚合类型 KVCacheGroupSpec 将同 typespec 层分组
- Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization
- KV Metrics: KVCacheMetricsCollector, BlockMetricsState — reuse interval, residency, eviction count, prefix cache hit rate

## Model dependency

Hardcoded to `facebook/opt-125m`. Tests load from the HuggingFace cache or the local path above. The model must be available for any run or test.

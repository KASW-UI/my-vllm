# Engineer-Spec: my-vllm 工程对齐规范

## Project re-positioning

**my-vllm** 不是一个"教学版 vLLM"，也不是一个"功能够用的简化 vLLM"。

项目定位：

> 以真实 vLLM 为教材，尽可能还原其工程组织、抽象边界、核心数据结构、执行路径和 GPU Runtime 接口。删除的只限于与当前职业目标无关的业务/生态功能。

职业定位：

> **AI Systems / AI Infrastructure Engineer → LLM Inference / Distributed Inference**
> 重点：GPU Performance + Runtime + Distributed Systems

my-vllm 的最终形态：

```
真实 vLLM
    │
    ├── 与职业方向强相关 → 尽可能还原
    ├── 与 Runtime 工程强相关 → 尽可能还原
    └── 与当前职业方向无关 → 可以删除
             ↓
        my-vLLM（vLLM 的严格工程子集）
```

---

## Verification standard

项目完成的标志不是"90 个文件都写完了"，而是：

> 可以拿真实 vLLM 的任意一个核心请求，从 `EngineCore.step()` 一直追到 GPU kernel，
> 并且在 my-vLLM 中存在对应的对象、数据结构、接口和执行路径。
> 当删除一个真实 vLLM 功能时，能明确解释它为什么被删，以及它原本位于架构的什么位置。

---

## 4-Level alignment framework

### Level 1：文件树

```
my-vLLM files ⊆ vLLM files
```

每个 my-vllm 文件必须对应真实 vLLM 中的一个文件。不存在于真实 vLLM、又没有独立架构价值的自创文件应删除。

例外：`tests/`、`docs/`、`run.py`、`Engineer-Spec/` 等学习辅助/配置/项目根目录文件。

### Level 2：对象和接口

保留的模块：class name、method name、responsibility、data ownership 尽量一致。

不要求参数 100% 一样，但 API 的设计思想必须一致。

### Level 3：核心逻辑

以下执行路径必须高度还原：

```
Scheduler → KV allocation → Input preparation → Unified batch
    → Model forward → Paged attention → Sampler → Output
```

### Level 4：实现技术

```
GPU memory → CUDA/Triton → Paged KV → Batch execution
    → CUDA Graph / async → NCCL → Multi-GPU
```

---

## Implementation priority

### P0-1：Unified Execution Model

**文件**：`v1/worker/gpu_model_runner.py`

**目标**：消除分离的 `_execute_prefill()` / `_execute_decode()`，改为统一批次

```
GPUModelRunner
├── _prepare_inputs()     # 构建统一输入张量
├── _model_forward()      # 单次 GPU forward
└── execute_model()       # 编排入口
```

**执行路径**：
```
Scheduler → scheduled tokens → 统一批次 → Model Forward
    (prefill + decode 在同一批次中)
```

**关键差异**：
- 不再使用 HuggingFace 的 `past_key_values`
- KV cache 读写通过 attention backend 的 paged attention 接口
- 位置的统一计算：`num_computed_tokens + local_offset`

### P0-2：Paged KV Cache 的真实执行路径

**文件**：`v1/worker/gpu_model_runner.py`, `v1/attention/backends/`

**目标**：删除 `gather_cached_kv` → contiguous tensor → HF Attention 路径

**正确路径**：
```
Scheduler → Block Table → Attention Metadata → Attention Backend
    → Paged Attention Kernel → Physical KV Cache
```

即：
```
logical block → block_table → physical block → GPU KV cache
```

不应该是：
```
physical blocks → gather → large contiguous tensor
```

- `gather_cached_kv_pytorch` 是教育性中间步骤，最终应消失
- `slot_mapping` 是连接 scheduler 和 attention backend 的关键数据结构

### P0-3：KV Cache 的对象层级

**文件**：`v1/core/kv_cache_manager.py`, `v1/core/block_pool.py`

**目标**：建立清晰的职责层次

```
KVCacheManager      # 对外接口，调度层可见
    ↓
KVCacheCoordinator  # 多 group 协调（可先实现单 group，但保留 coordinator 概念）
    ↓
KVCacheGroup        # 单组 KV cache（full-attn, sliding-window, etc.）
    ↓
BlockPool           # 物理块池
    ↓
KVCacheBlock        # 单个物理块
```

不要求实现所有 group 类型，但 Manager → Coordinator → Group → Pool → Block 的职责层次必须建立。

### P0-4：Block identity / hash / caching

**文件**：`v1/core/block_pool.py`, `v1/core/kv_cache_utils.py`

**目标**：Block 不只是内存块，而是可识别、可复用、可缓存的数据对象

当前已有：`ref_cnt`、`free_blocks`、`touch()`、eviction

需要加强：
```
BlockHash / BlockHashWithGroupId  # 块的唯一标识
cached block                       # 已被缓存的块
computed block                     # 本次计算的块
```

**核心流程**：
```
Request → Token prefix → Block hash → KV Cache lookup → reuse existing block
```

这直接关系到后续的 KV Transfer、LMCache、Mooncake 等分布式方案。

### P0-5：Scheduler 的真正数据模型

**文件**：`v1/core/sched/scheduler.py`

**目标**：对齐核心调度逻辑

```
waiting queue → num_computed_tokens → num_scheduled_tokens
    → KV allocation → SchedulerOutput
```

**核心关系**：
```
num_computed_tokens    # 已经算了多少
    ↓
remaining tokens        # 还剩多少
    ↓
num_scheduled_tokens   # 这一步算多少
    ↓
required KV blocks     # 需要多少 blocks
    ↓
admission decision     # 是否允许继续
```

这个逻辑比"把 Scheduler 写长"重要得多。

### P0-6：Request / RequestState 分离

**文件**：`v1/request.py`, `v1/worker/gpu/states.py`

**目标**：区分请求标识和运行时状态

```
Request              # 请求级信息
├── request identity  (req_id)
├── prompt            (prompt_token_ids)
├── sampling params   (temperature, top_k, top_p, max_tokens)
└── priority / arrival_time
```

```
RequestState          # 运行时状态（每 step 变化）
├── output_token_ids
├── num_computed_tokens
├── num_generated_tokens
├── finished state
├── logprobs
└── block_table (当前持有的 KV blocks)
```

**工程意义**：状态所有权、修改权限、生命周期的分离。这是大型 Runtime 系统的典型设计模式。

### P0-7：InputBatch

**文件**：`v1/worker/gpu/input_batch.py`

**目标**：GPUModelRunner 不直接操作 Python Request 对象，而是接收 GPU 友好的批次数据结构

```
Python Requests → Scheduler → InputBatch → GPU-friendly representation → ModelRunner
```

**关键概念**（不需要全量复制 630 行，但需要复制思想）：
- `idx_mapping`：batch 槽位 → 持久化 request state 槽位的间接映射
- 预分配的 GPU 缓冲区（避免 per-step 分配）
- `block_table_tensor`、`slot_mapping` 等 GPU 侧 tensor

### P0-8：Attention Backend 抽象

**文件**：`v1/attention/backend.py`, `v1/attention/metadata.py`, `v1/attention/selector.py`

**目标**：建立完整的 backend 抽象层次

```
AttentionBackend (ABC)
├── FlashAttentionBackend   # SDPA-based (operational since C110)
└── [future] other backends

AttentionMetadata            # 传给 backend 的元数据
AttentionMetadataBuilder     # 从 batch 构建 metadata
```

**抽象边界**：
```
ModelRunner → Attention interface → Backend → Kernel
```

不要求实现 FlashAttention CUDA 内核，但抽象边界必须正确。

### P0-9：Model Executor / Worker / Executor 边界

**文件**：`v1/executor/abstract.py`, `v1/worker/gpu_worker.py`, `v1/worker/worker_base.py`

**目标**：对齐职责边界

```
EngineCore      # orchestration（调度 + 执行）
    ↓
Executor        # distributed execution abstraction（单机/多机/多进程）
    ↓
Worker          # device-specific execution（GPU/CPU 初始化、模型加载）
    ↓
GPUModelRunner  # GPU execution（批次构建、模型前向、采样）
    ↓
Attention / Model / Sampler
```

不要求实现 `collective_rpc` 全部方法，但 executor → worker → runner 的调用链和职责分配必须正确。

### P0-10：Distributed abstraction

**文件**：`distributed/parallel_state.py`, `config/parallel.py`

**目标**：TP=1、PP=1 只是当前配置，不是系统不支持

```
ParallelConfig → ParallelState → Process Groups → Executor → Worker
```

NCCL 基础设施（`GroupCoordinator`、`initialize_model_parallel`）已存在，但需要打通：
- `run.py` 支持切换 `MultiprocExecutor`
- 模型加载支持 TP 权重分片
- attention 支持 TP 通信

最终：
```
TP=1, PP=1 → 单 GPU 推理
TP=2, PP=1 → 两 GPU 张量并行
TP=1, PP=2 → 两 GPU 流水线并行
```

---

## P1：应该逐渐还原

### P1-1：Dynamic KV Cache Initialization

`num_gpu_blocks = 512` → GPU memory profiling → available memory → KV cache capacity → `num_gpu_blocks`

这属于：Runtime 如何根据硬件资源自适应。

### P1-2：真实 Model Layer 抽象 ★★★★★

**文件**：`model_executor/layers/linear.py`

**目标**：`nn.Linear` → `LinearBase` → `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`

这里是 **Model → Parallelism → Communication** 的交汇点。

**必须掌握的 6 个概念**（详见本文 `## Model loading: checkpoint → GPU`）：
1. `initialize_model()` / `load_weights()` 分离 — 控制 GPU 显存峰值，enable TP sharding
2. Meta device — `torch.device("meta")` 先建结构，再加载权重
3. Weight mapping / `weight_loader` — checkpoint weight names ≠ runtime parameter layout
4. Parallel Linear — `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`：本地计算 + NCCL 通信 = 全局结果
5. Quantization at load time — checkpoint 和 GPU Runtime representation 可能不同
6. Memory lifecycle — 从 safetensors 到 GPU HBM 的内存分配时序

### P1-3：OPT 模型实现 ★★★★★

**文件**：`model_executor/models/opt.py`

**目标**：`from transformers import OPTForCausalLM` → 用 vLLM 的 layer 重新实现 OPT

当前 `AutoModelForCausalLM.from_pretrained()` + `model.to("cuda")` 绕过了整个 Runtime 边界——没有 vLLM-style layers、没有 TP-aware loading、没有 weight mapping。

修复后应形成：

```
Checkpoint (safetensors)
    ↓
ModelLoader (BaseModelLoader → DefaultModelLoader)
    ↓ initialize_model() on meta device
    ↓ load_weights() via safetensors iterator + weight_mapping
    ↓ TP-aware shard
vLLM Model (OPTForCausalLM)
    ├── OPTAttention (vLLM Attention backend, QKVParallelLinear)
    ├── OPTDecoderLayer
    └── VocabParallelEmbedding / ParallelLMHead / LogitsProcessor
    ↓
GPU-resident model
```

否则会一直停留在 "vLLM = HuggingFace + Scheduler" 的错误理解上。

### P1-4：Sampler 工程边界

```
Sampler → SamplerOutput → LogitsProcessor → ModelRunnerOutput
```

功能可简化，但边界要对齐。

### P1-5：Metrics

保留一个真正属于 vLLM 的 metrics 层：TTFT、ITL、throughput、KV cache usage、request latency。

这是后续性能实验的必需品。

---

## Implementation scope tiers

**Rule**: if a feature alters GPU execution, KV Cache, batching, scheduler, communication, or model parallelism, it cannot remain NoOp/ABC. It must have at least one real end-to-end implementation path.

### Must implement

```
Speculative Decoding     → 改变执行/采样/RequestState，属于 Runtime 核心
EP (Expert Parallelism)  → MoE + All-to-All + NCCL，属于 Distributed Inference
CP (Context Parallelism) → 改变 Attention/Context 执行和通信
KV Connector             → 至少一个真实 backend（local/remote KV transfer）
Quantization             → 至少一种真实实现（如 FP8）
```

### Architecture + limited implementation

```
LoRA                → 接口 + 一个真实路径，证明理解 Runtime 扩展
Mamba / SSM         → 理解 Runtime impact，实现优先级低
Multimodal          → 理解 encoder/input processing 边界，有限实现
Structured Output   → logits → constraint → sampler 基本路径
```

### Excluded from core code (study separately)

```
K8s ecosystem        → 必须理解，但不进入 my_vllm/ 核心代码
Ray ecosystem        → 必须理解 Distributed Runtime 角色，但不进入核心
Autoscaling          → 必须理解 serving 层调度闭环，但不进入核心
Encoder-Decoder      → 理解另一种 KV/执行模式，不实现
Mamba / SSM / Hybrid → 理解非 Transformer state cache，不实现
非 OPT 模型架构         → 通过架构覆盖 Dense/GQA/MoE，不追求模型数量
非 CUDA 平台 backend   → 不实现
```


---

## KV Cache dtype abstractions

The task is not "calling different precision KV cache libraries" — it is understanding vLLM's implementation abstraction for different KV Cache dtypes at three levels:

### FP16 / BF16 — must fully understand (baseline)

K/V tensor → block layout → write KV cache → paged attention → read K/V. Must be able to explain and implement: physical layout, block table, logical→physical block mapping, K/V write path, how attention kernel reads by block, how different sequences share/access different blocks in a batch.

### FP8 — must understand the implementation, not necessarily write the full kernel

FP16/BF16 K,V → quantization → FP8 KV cache → attention read → dequantization/scale. Must understand: E4M3 vs E5M2 differences, why KV is suitable for low-precision storage, scale provenance (per-tensor / per-token / per-channel), when quantization happens on write, how scale is consumed on read, and GPU/backend FP8 support differences. Can call mature FP8 libraries, but must know what problem they solve internally.

### Other dtypes — understand the abstraction, not every format

The goal is establishing the abstraction chain:

```
KVCacheSpec → dtype → scale/quantization metadata → cache storage layout → attention backend
```

Individual CUDA kernels for each dtype are not required.

---

## Model architecture priorities

The goal is NOT "support many HF models" — it is understanding **how architecture differences force the Runtime to change**. Priority by Runtime impact:

| Category | Examples | Why |
|---|---|---|
| **Must master** | Dense Transformer, GQA/MQA, MoE | Baseline execution; GQA directly changes KV Cache shape/metadata/kernel; MoE introduces Router→Top-K→Dispatch→Expert→Combine and connects to All-to-All/NCCL/EP |
| **Must understand Runtime impact** | Long Context (RoPE), Multimodal (encoder), Encoder-Decoder, Mamba/SSM | Changes attention/KV cache; introduces encoder/input processing; alternative execution/KV modes; non-Transformer state cache |
| **Awareness** | Other architectures | Know why vLLM has abstractions, can read code, can judge which Runtime module is affected |

The implementation approach: **model count increases should add config/thin adapter layers, not duplicate Runtime**. Reusable layers (Linear, QKVParallelLinear, MLP, Attention, FusedMoE) are shared across architectures:

```
Model → Model Architecture → Reusable Layers
                                 ├── Linear / ColumnParallelLinear / RowParallelLinear
                                 ├── QKVParallelLinear (GQA/MQA)
                                 ├── Attention (backend-swappable)
                                 └── FusedMoE (Router + Expert + EP)
```

---

## Model loading: checkpoint → GPU

Model loading is not about supporting many checkpoint formats (HF, GGUF, safetensors...) — it is the **Runtime → GPU Memory → Distributed Execution** interface. The goal is understanding how a checkpoint becomes a GPU-resident, TP-shardable, executable model.

### Why vLLM does not use `from_pretrained()` + `.to("cuda")`

```
AutoModelForCausalLM.from_pretrained()   ← loads HF class, all weights monolithically
model.to("cuda")                         ← moves everything to GPU at once
```

This bypasses: TP sharding, meta-device memory efficiency, weight-to-runtime-parameter mismatch, online quantization. The real chain is:

```
ModelConfig → architecture selection → initialize_model() on meta device
    → load_weights() via safetensors iterator → weight_mapping → TP-aware shard
    → post-load processing → GPU-resident model
```

### 6 concepts to master

| Concept | Why |
|---|---|
| **init/load separation** | `initialize_model()` creates structure; `load_weights()` fills weights. Controls GPU memory peak, enables TP sharding, online quantization |
| **Meta device** | `torch.device("meta")` creates parameter structure without GPU storage — critical for large-model loading memory management |
| **Weight mapping / weight_loader** | Checkpoint weight names ≠ runtime parameter layout. TP fuses QKV in checkpoint → QKVParallelLinear in runtime. The `weight_loader` bridges this mismatch |
| **Parallel Linear** | `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `VocabParallelEmbedding` — local computation + NCCL communication = global result. This connects Model Loading → TP → NCCL → Distributed Inference |
| **Quantization at load time** | Checkpoint representation ≠ GPU Runtime representation. Weight loader may quantize during loading (`finalize_layerwise_processing`) |
| **Memory lifecycle** | Checkpoint (disk/mmap) → model construction (meta) → GPU allocation → weight loading (TP shard) → GPU-resident weights. Must answer: when is what memory allocated? What does each GPU get under TP=4? |

### Current gap

`model_executor/model_loader.py` calls `AutoModelForCausalLM.from_pretrained()` + `model.to("cuda")`, returning HF `OPTForCausalLM` with `nn.Linear` and HF attention. `models/opt.py` is a `from transformers import OPTForCausalLM` placeholder.

This bypasses the entire Runtime boundary — no vLLM-style layers, no TP-aware loading, no weight mapping. P1-2 (Model layers) and P1-3 (OPT model) are the high-priority P1 items that close this gap.

---

## Files to delete

以下文件不存在于真实 vLLM 目标版本，属于自创文件，应删除/重构：

| 文件 | 状态 | 原因 |
|---|---|---|---|
| `v1/attention/backends/torch_attn.py` | ✅ C110 已删除 | 真实 vLLM 无此文件；代码已合并入 flash_attn.py |
| `v1/attention/ops/ring_attention.py` | ✅ C110 已删除 | 真实 vLLM 无此文件；DCP attention 内联于 FlashAttention backend |
| `distributed/elastic_ep/dispatch.py` | ✅ C110 已删除 | 真实 vLLM 无此文件；EP dispatch 在 `GroupCoordinator` 内 |
| `v1/metrics/collector.py` | ✅ C110 已删除 | 真实 vLLM 无此文件；metrics 在 `stats.py` + `loggers.py` |
| `v1/config.py` | ✅ C110 已删除 | 真实 vLLM 无此文件；仅为 backward-compat shim |
| `runtime/` 目录 | ⏳ 待迁移 | 真实 vLLM 无此目录；入口点应在 `entrypoints/` |

---
 
## Implementation order

```
Phase 1: 清理 + 骨架建立
  1. [DONE C110] 删除自创文件（torch_attn.py, ring_attention.py, dispatch.py, collector.py, v1/config.py）
  2. [NEXT] 重构 file tree 使 runtime/ → entrypoints/ 路径对齐
  3. P0-9: Executor / Worker / Runner 边界对齐
  4. P0-10: Distributed abstraction 打通（TP=1 可运行，TP=2 可配置）

Phase 2: 核心执行路径
  5. P0-1: Unified Execution Model（最关键，改动最大）
  6. P0-2: Paged KV Cache 的真实执行路径
  7. P0-8: Attention Backend 抽象
  8. P0-7: InputBatch

Phase 3: 数据模型
  9. P0-4: Block identity / hash / caching
  10. P0-3: KV Cache 对象层级
  11. P0-6: Request / RequestState 分离
  12. P0-5: Scheduler 数据模型

Phase 4: 辅助系统
  13. P1-1: Dynamic KV Cache Initialization
  14. P1-2: Model Layer 抽象
  15. P1-3: OPT 模型实现
  16. P1-4: Sampler 边界
  17. P1-5: Metrics

Phase 5: 分布式打通 + Must Implement
  18. NCCL 多卡执行验证
  19. TP=2 / PP=2 可运行
  20. Speculative Decoding：至少一条完整执行路径（draft → verify → accept → KV update）
  21. EP：FusedMoE + All-to-All + NCCL dispatch/combine（至少 2 GPU）

Phase 6: Must Implement（续）
  22. CP：至少一条 Context Parallelism 执行路径（partition → local attn → communication → combine）
  23. KV Connector：至少一个真实 backend（local shared-memory 或 remote transfer）
  24. Quantization：至少 FP8 量化路径（KV cache 或 weight quantization）
```

每个 Phase 内的顺序可按依赖关系调整，但 Phase 之间的顺序不应跨越。

---

## Commit convention

继续沿用 `Cxxx: Module: Description` 格式。但 Cxxx 编号不再使用原有 24 个 P0 的编号体系，而是从 `C110` 开始新序列（避免与已完成的 C098 及之前的编号冲突）。

每完成一个 P0/P1 项，产生一个 commit + 对应的 `docs/snapshots/Cxxx.md` + Git tag。

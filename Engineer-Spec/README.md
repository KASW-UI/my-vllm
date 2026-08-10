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
├── TorchAttentionBackend   # PyTorch SDPA 后端
└── FlashAttentionBackend   # FlashAttention 后端（stub → 可运行）

AttentionMetadata             # 传给 backend 的元数据
AttentionMetadataBuilder      # 从 batch 构建 metadata
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

### P1-2：真实 Model Layer 抽象

`nn.Linear` → `LinearBase` → `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear`

这里是 Model → Parallelism → Communication 的交汇点。

### P1-3：OPT 模型实现

`from transformers import OPTForCausalLM` → 用 vLLM 的 layer 重新实现 OPT

否则会一直停留在"vLLM = HuggingFace + Scheduler"的错误理解上。

### P1-4：Sampler 工程边界

```
Sampler → SamplerOutput → LogitsProcessor → ModelRunnerOutput
```

功能可简化，但边界要对齐。

### P1-5：Metrics

保留一个真正属于 vLLM 的 metrics 层：TTFT、ITL、throughput、KV cache usage、request latency。

这是后续性能实验的必需品。

---

## P1：可作为接口存在，但不实现

以下功能作为抽象接口/NoOp 实现保留，保持架构完整性：

```
KV Connector        → KVConnectorBase_V1 + NoopConnector
LoRA                → 接口 stub，不实现适配器管理
Quantization        → 接口存在，默认 UnquantizedLinearMethod
Spec Decode         → 接口 stub
Structured Output   → 接口 stub
Multimodal          → 接口 stub
Mamba / SSM         → 不实现
EP / EPLB           → 接口 stub（Expert Parallel 调度）
CP / DCP            → 接口 stub（Context Parallelism）
```

原则：如果某个功能会改变核心架构，就算不实现，也要保留正确的抽象位置。

---

## Files to delete

以下文件不存在于真实 vLLM 目标版本，属于自创文件，应删除/重构：

| 文件 | 原因 |
|---|---|
| `v1/attention/backends/torch_attn.py` | 真实 vLLM 无此文件；PyTorch SDPA backend 应放在正确位置 |
| `v1/attention/ops/ring_attention.py` | 真实 vLLM 无此文件；DCP attention 内联于 FlashAttention backend |
| `distributed/elastic_ep/dispatch.py` | 真实 vLLM 无此文件；EP dispatch 在 `GroupCoordinator` 内 |
| `v1/metrics/collector.py` | 真实 vLLM 无此文件；metrics 在 `stats.py` + `loggers.py` |
| `v1/config.py` | 真实 vLLM 无此文件；仅为 backward-compat shim |
| `runtime/` 目录 | 真实 vLLM 无此目录；入口点应在 `entrypoints/` |

---

## Scope exclusion（明确不实现）

```
LoRA（全功能）
Multimodal（图像/音频/视频）
Encoder-Decoder 模型
Mamba / SSM / Hybrid
Structured Output（全功能）
Speculative Decoding（全功能）
各种 Quantization（GPTQ, AWQ, FP8, INT4...）
各种 Model Architecture（非 OPT）
各种 Sampling Backend（非核心 sampling）
各种 Platform Backend（ROCm, XPU, TPU...）
硬件兼容性代码
K8s 部署
Ray（全功能，但保留 actor 接口 stub）
Autoscaling
```

---

## Implementation order

```
Phase 1: 清理 + 骨架建立
  1. 删除自创文件（torch_attn.py, ring_attention.py, dispatch.py, collector.py, v1/config.py）
  2. 重构 file tree 使 runtime/ → entrypoints/ 路径对齐
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

Phase 5: 分布式打通
  18. NCCL 多卡执行验证
  19. TP=2 / PP=2 可运行
```

每个 Phase 内的顺序可按依赖关系调整，但 Phase 之间的顺序不应跨越。

---

## Commit convention

继续沿用 `Cxxx: Module: Description` 格式。但 Cxxx 编号不再使用原有 24 个 P0 的编号体系，而是从 `C110` 开始新序列（避免与已完成的 C098 及之前的编号冲突）。

每完成一个 P0/P1 项，产生一个 commit + 对应的 `docs/snapshots/Cxxx.md` + Git tag。

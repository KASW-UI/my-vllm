# Base-on-C032 Career-Focused Implementation Plan

> 基于 C032 基线，按 **Distributed LLM Inference / KV Cache / Disaggregated Inference Engineer**
> 职业目标重新切片后的 Git 实现方案。

---

## 一、总原则

| 原则 | 说明 |
|---|---|
| Cxxx 编号不变 | 尊重已有 Principle/ 设计文档的编号体系 |
| 每组内重排顺序 | 按优先级 + 依赖链，不按原编号顺序 |
| P0 = 完整实现 | 24 个必修 stage，每 commit 可运行 |
| P1 = 精简实现/分析 | 13 个选修，有精力再补 |
| P2 = 明确跳过 | 61 个，标明但不实现 |
| 线性 Git history | 不 squash，每个 Cxxx commit 独立可 checkout |
| 每 commit 必须可运行 | `python run.py` 或 `pytest tests/` |

---

## 二、96 个未实现 stage → 三档分类

### P0 · 必须实现（24 commits）

| 原编号 | Stage | 原属组 | 理由 |
|---|---|---|---|
| C035 | BlockPool GC | C033-C050 | KV Cache 生命周期基础 |
| C036 | Prefix Cache CoW | C033-C050 | 共享块写入语义——多机 KV 共享前序 |
| C037 | SchedulerInterface ABC | C033-C050 | 调度器标准化接口 |
| C038 | Scheduler Resource Model | C033-C050 | GPU/KV/token 预算模型 |
| C048 | Metrics TTFT/TPOT/ITL | C033-C050 | 性能瓶颈定位的基础 |
| C050 | TP=2 真实 NCCL | C033-C050 | 第一个真正的多 GPU 通信 |
| C051 | Priority Scheduler | C051-C070 | C052 的前置依赖 |
| C052 | Preemption + KV reclaim | C051-C070 | KV 驱逐/回收——KV 迁移的前序 |
| C054c | ref_cnt 级联释放 | C051-C070 | CoW 路径的引用计数完整性 |
| C054 | KV Cache Metrics | C051-C070 | KV 利用率、命中率 |
| C081 | MoE fused kernel | C071-C100 | Expert 并行基础 |
| C082 | EP All-to-All dispatch | C071-C100 | MoE 的分布式扩展 |
| C083 | TP=2 工业级 NCCL | C071-C100 | initialize_model_parallel + GroupCoordinator |
| **C080** | **MultiprocExecutor** | **C071-C100** | **多进程执行器 — C084/C086/C102 的前置依赖** |
| C084 | PP Pipeline Parallel | C071-C100 | 1F1B + bubble 分析 |
| C085 | CP Ring Attention | C071-C100 | 长序列分布式 |
| C086 | Multi-GPU Integration | C071-C100 | TP×PP×EP×CP 拓扑+成本模型 |
| C091 | Ray 部署 | C071-C100 | 分布式集群编排 |
| C092 | K8s 部署 | C071-C100 | 生产环境管理 |
| C093 | Autoscaling | C071-C100 | 集群扩缩容逻辑 |
| **C102** | **KV Connector** | C101-C120 | **跨进程 KV 传输——PD 分离基础** |
| C107 | Chunked Prefill 调度分析 | C101-C120 | hybrid batch 策略深度 |
| C098 | Benchmark Suite | C071-C100 | 全链路性能建模 |
| C069 | Config 体系 | C051-C070 | 配置结构标准化 |

### P1 · 精简实现/分析文档（13 commits，可选）

| 原编号 | Stage | 原属组 | 方式 |
|---|---|---|---|
| C039 | InputBatch 工业化 | C033-C050 | 精简实现 |
| C040 | BlockTable Triton | C033-C050 | 分析文档 |
| C041 | ForwardContext | C033-C050 | 精简实现 |
| C042 | AttentionBackend Registry | C033-C050 | 精简实现 |
| C053 | Async Scheduler | C051-C070 | 精简实现 |
| C054b | EngineCore Lifecycle | C051-C070 | 精简实现 |
| C068 | KVCacheManager 扩展 | C051-C070 | 分析文档 |
| C101 | MoE DP | C101-C120 | 精简实现 |
| C103 | DP×EP×TP 3D Topology | C101-C120 | 分析文档 |
| C070 | C051-C070 集成 | C051-C070 | 只跑已有测试 |
| C100 | C071-C100 集成 | C071-C100 | 只跑已有测试 |
| C118 | Extended Benchmark Suite | C101-C120 | 轻量 benchmark |
| C120 | C101-C120 Integration | C101-C120 | 文档+测试 |

### P2 · 明确跳过（61 commits）

| 原编号范围 | 内容 | 跳过理由 |
|---|---|---|
| **C033, C034** | **KV Cache Spec + Coordinator** | **废案——未合并分支，从 C032 重新开始** |
| C040 Triton impl, C043-A/B/C, C044-A/B | Attention backend 深度 + CUDA Graph | kernel author 工作 |
| C045-C047 | EngineClient / Multiproc / Async 完善 | 工程细节 |
| C049 | OpenAI Server | API 层 |
| C050b, C062b, C063b | GEMM / MLA / FlashAttention kernel | kernel author 工作 |
| C056-C057 | Advanced Sampler / Logprobs | 模型层 |
| C058-C062 | Spec Decode 全家桶 (5 stages) | 投机解码 |
| C063-C064 | Grammar + XGrammar | 结构化输出 |
| C066 | Embedding Runtime | Pooling |
| C087-C090 | LoRA / FP8 / MultiModal / MLA Analysis | 模型优化/多模态 |
| C094-C095 | Engine Client 完善 / LLM SDK | 用户层 |
| C104-C106, C108-C111 | 分析专题 | 过于深度 |
| C112-C117 | Demo 脚本 | 验证用 |
| C121-C140 | 17 个 Triton kernel | **你不是 kernel author** |
| C119 | C100 vs C120 comparison | 文档 |

---

## 三、Git 实现序列

### Phase 1: 单 GPU 生产级 Runtime（11 commits）

```
C032 (基线)
  │
  ├─ C035  BlockPool GC                     ← KV 生命周期基础
  │   └─ C036  Prefix Cache CoW             ← 需要 C035
  │       └─ C054c  ref_cnt 级联释放         ← 需要 C036
  │           └─ C054  KV Cache Metrics      ← 需要 C035
  │
  ├─ C037  SchedulerInterface ABC           ← 调度器标准化基础
  │   └─ C038  Scheduler Resource Model     ← 需要 C037
  │       └─ C051  Priority Scheduler        ← 需要 C037
  │           └─ C052  Preemption + KV reclaim  ← 需要 C051
  │
  ├─ C048  Metrics TTFT/TPOT/ITL            ← 独立，尽早接入
  │
  ├─ C050  TP=2 weight sharding (最小多 GPU)  ← 分布式网关
  │
  └─ C069  Config 体系                       ← 独立，最后收尾
```

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 1 | **C035: KVCache: BlockPool lifecycle upgrade + GC** | C032 | `pytest tests/` |
| 2 | **C036: KVCache: Prefix Cache Copy-on-Write** | C035 | `pytest tests/` |
| 3 | **C037: Scheduler: SchedulerInterface ABC** | C032 | `python run.py` |
| 4 | **C038: Scheduler: Resource model (GPU/KV/token budget + preemption)** | C037 | `python run.py` |
| 5 | **C051: Scheduler: Priority scheduling (FCFS/Priority)** | C037 | `python run.py` |
| 6 | **C052: Scheduler: Preemption + KV reclaim (RECOMPUTE/SWAP)** | C051 | `python run.py` |
| 7 | **C054c: KVCache: KVCacheBlock ref_cnt cascading release** | C036 | `pytest tests/` |
| 8 | **C048: Metrics: TTFT, TPOT, ITL, throughput, KV utilization** | — | `python run.py` |
| 9 | **C054: KVCache: Block metrics state (reuse/residency/eviction/hit rate)** | C035 | `python run.py` |
| 10 | **C050: Distributed: TP=2 weight sharding (内联 torch.distributed)** | — | `torchrun --nproc_per_node=2 run.py` |
| 11 | **C069: Config: VllmConfig / ModelConfig / CacheConfig** | — | `python run.py` |

---

### Phase 2: 分布式 Runtime（7 commits）

```
C050 (TP=2 weight sharding, 内联通信)
  │
  ├─ C083  TP=2 工业级 NCCL          ← 需要 C050: initialize_model_parallel + GroupCoordinator
  │   ├─ C080  MultiprocExecutor     ← 需要 C083: 多进程执行器基础设施
  │   ├─ C084  PPHandler + PP        ← 需要 C083 + C080
  │   └─ C085  CP Ring Attention     ← 需要 C083
  │
  ├─ C081  FusedMoE + FusedMoERouter ← 独立模块
  │   └─ C082  EP All-to-All         ← 需要 C081
  │
  └─ C086  Multi-GPU Integration     ← 需要 C080 + C082 + C084 + C085
```

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 12 | **C083: Distributed: TP=2 industrial (initialize_model_parallel + GroupCoordinator)** | C050 | `torchrun --nproc_per_node=2 run.py` |
| 13 | **C080: Executor: MultiprocExecutor** | C083 | `python run.py` |
| 14 | **C081: MoE: FusedMoE (factory) + FusedMoERouter + fused_moe_kernel** | — | `python run.py` |
| 15 | **C082: Distributed: EP All-to-All** | C081 | `python run.py` |
| 16 | **C084: Distributed: PPHandler + 1F1B-lite** | C083, C080 | `python run.py` |
| 17 | **C085: Distributed: CP Ring Attention** | C083 | `python run.py` |
| 18 | **C086: Distributed: TP×PP×EP×CP multi-GPU integration** | C080, C082, C084, C085 | `python run.py` |

---

### Phase 3: 集群 + KV 传输（4 commits）

```
C086 (分布式全景)
  │
  ├─ C091  RayDistributedExecutor  ← 独立
  ├─ C092  K8s 部署               ← 独立
  └─ C093  Autoscaling             ← 需要 C048 (Metrics)
  │
C080 + C086 + 完整 KV 理解
  └─ C102  KVConnectorBase_V1       ← 跨进程 KV 传输，PD 分离基础
```

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 19 | **C091: Cluster: RayDistributedExecutor (actor + initialize_ray_cluster)** | — | ray job |
| 20 | **C092: Cluster: K8s deployment** | — | `kubectl apply` |
| 21 | **C093: Cluster: Autoscaling** | C048 | 单元测试 |
| 22 | **C102: KVCache: KVConnectorBase_V1 — cross-process KV block transfer (start_load_kv/save_kv_layer)** | C080, C086 | `python run.py` |

---

### Phase 4: 收敛与验证（2 commits）

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 23 | **C107: Scheduler: Chunked prefill + paged decode analysis** | C030, C052 | `python run.py` |
| 24 | **C098: Benchmark: throughput, TTFT/TPOT/ITL, prefill, decode, MoE, EP, TP, PP, CP** | 全部 | `python benchmark/run.py` |

---

## 四、最终 Git log

```
C032  Cleanup: complete C031 (基线)
  │
  │  Phase 1: 单 GPU 生产级 (11 commits)
  ├─ C035  KVCache: BlockPool lifecycle upgrade + GC
  ├─ C036  KVCache: Prefix Cache Copy-on-Write
  ├─ C037  Scheduler: SchedulerInterface ABC
  ├─ C038  Scheduler: Resource model (GPU/KV/token budget)
  ├─ C051  Scheduler: Priority scheduling (FCFS/Priority)
  ├─ C052  Scheduler: Preemption + KV reclaim (RECOMPUTE/SWAP)
  ├─ C054c KVCache: ref_cnt cascading release
  ├─ C048  Metrics: TTFT, TPOT, ITL, throughput, KV utilization
  ├─ C054  KVCache: Block metrics (reuse/residency/eviction/hit rate)
  ├─ C050  Distributed: TP=2 weight sharding (内联 torch.distributed)
  ├─ C069  Config: VllmConfig / ModelConfig / CacheConfig
  │
  │  Phase 2: 分布式 Runtime (7 commits)
  ├─ C083  Distributed: TP=2 industrial NCCL (initialize_model_parallel)
  ├─ C080  Executor: MultiprocExecutor (WorkerProcess lifecycle)
  ├─ C081  MoE: FusedMoE (factory) + FusedMoERouter + fused_moe_kernel
  ├─ C082  Distributed: EP All-to-All dispatch/combine
  ├─ C084  Distributed: PPHandler + pipeline parallel (1F1B-lite)
  ├─ C085  Distributed: CP Ring Attention
  ├─ C086  Distributed: TP×PP×EP×CP multi-GPU integration
  │
  │  Phase 3: 集群 + KV 传输 (4 commits)
  ├─ C091  Cluster: RayDistributedExecutor (RayWorkerWrapper + initialize_ray_cluster)
  ├─ C092  Cluster: K8s deployment (Deployment + HPA)
  ├─ C093  Cluster: Autoscaling (queue_depth decision loop)
  ├─ C102  KVCache: KVConnectorBase_V1 — cross-process KV block transfer (start_load_kv/save_kv_layer)
  │
  │  Phase 4: 收敛 (2 commits)
  ├─ C107  Scheduler: Chunked prefill + paged decode analysis
  └─ C098  Benchmark: throughput, latency, KV scaling suite
```

**总计 24 P0 commits。** P0 (24) + P1 (13) = 37 commits，原计划 96 个的 38.5%。

---

## 五、与已有规划的差异

| 维度 | 原有规划 | 本方案 |
|---|---|---|
| 实现总数 | 96 commits | 24 P0 + 13 P1 = 37 |
| C043-A/B/C attention 三件套 | 全做 | **跳过** |
| C051 Priority / C052 Preemption | C051-C070 组后期 | **提至 Phase 1** |
| C058-C062 Spec Decode 5 个 | 全做 | **跳过** |
| C121-C140 17 个 kernel | 全做 | **跳过** |
| C102 KV Connector | C101-C120 组 | **提至 Phase 3** |
| C048/C054 Metrics | 原位置靠后 | **提前至 Phase 1** |
| C069 Config | C051-C070 组 | **提至 Phase 1 末尾** |

---

## 六、执行路线

```text
现在 C032 基线
  ↓
Step 1: Phase 1 (11 commits) → 单 GPU 生产级引擎
  ↓
Step 2: Phase 2 (6 commits) → 4 种并行策略
  ↓
Step 3: Phase 3 (4 commits) → KV 跨机传输 MVP
  ↓
Step 4: Phase 4 (2 commits) → 全链路 benchmark
  ↓
Step 5: Level 4 实战 → 改 Scheduler / 改 KV 驱逐 / Nsight profiling / KV 迁移实验
```

---

## 七、不做的部分

| 放弃的能力 | 理由 |
|---|---|
| FlashAttention/TritonAttention backend 实现 | kernel author 工作 |
| CUDA Graph 录制与回放 | 底层 infra |
| Spec Decode (Eagle/Medusa/NGram) | 与分布式推理正交 |
| Advanced Sampler / Grammar / Logprobs / Embedding | 模型层 |
| LoRA / FP8 / MultiModal / MLA | 模型优化/多模态 |
| C121-C140 全部 17 个 Triton kernel | kernel author 专项 |
| OpenAI Server / LLM SDK | API/用户层 |

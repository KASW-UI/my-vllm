# P1 选修阶段

> Phase 1 完成后、有余力时选择性实现。13 stages，每个精简实现或写分析文档。
>
> **真实 vLLM 路径映射**: `v1/model_executor/` → `vllm/model_executor/`（顶层）；`v1/distributed/` → `vllm/distributed/`（顶层）。阅读真实源码时需做路径翻译。

---

## 总览

| Stage | 内容 | 方式 | 原属组 |
|---|---|---|---|
| C039 | InputBatch 工业化 | 精简实现 | C033-C050 |
| C040 | BlockTable Triton | 分析文档 | C033-C050 |
| C041 | ForwardContext | 精简实现 | C033-C050 |
| C042 | AttentionBackend Registry | 精简实现 | C033-C050 |
| C053 | Async Scheduler | 精简实现 | C051-C070 |
| C054b | EngineCore Lifecycle | 精简实现 | C051-C070 |
| C068 | KVCacheManager 扩展 | 分析文档 | C051-C070 |
| C101 | MoE DP | 精简实现 | C101-C120 |
| C103 | DP×EP×TP 3D Topology | 分析文档 | C101-C120 |
| C070 | C051-C070 集成 | 只跑已有测试 | C051-C070 |
| C100 | C071-C100 集成 | 只跑已有测试 | C071-C100 |
| C118 | Extended Benchmark Suite | 轻量 benchmark | C101-C120 |
| C120 | C101-C120 Integration | 文档+测试 | C101-C120 |

---

## C039 — InputBatch 工业化

**来源**: `Principle/C033-C050/C033-C050.md` #C039 节

C032 `RequestState` 字段少（仅 4 个 GPU 字段）。C039 扩展为 `CachedRequestState`:
```python
prompt_len           # prompt 长度
prefill_len          # 实际 prefill 处理的 token 数
draft_tokens         # spec decode 的 draft token IDs (stub)
positions            # per-token 位置张量
```

**方式**: 精简实现，只加 `prompt_len` 和 `prefill_len`（chunked prefill 场景使用）。

---

## C040 — BlockTable Triton

**来源**: `Principle/C033-C050/C033-C050.md` #C040 节

C032 的 `compute_slot_mappings()` 用 Python for 循环。C040 分析 Triton kernel 替代方式。

**方式**: 分析文档 `docs/analysis/block_table_triton.md`，不实现代码。理解：
- sloat_mapping 的计算逻辑
- 为什么生产环境用 Triton/CUDA kernel 替代 Python 循环
- 参考 vLLM `copy_blocks()` kernel 的接口和数据流

---

## C041 — ForwardContext

**来源**: `Principle/C033-C050/C033-C050.md` #C041 节

解决 attention 层不知道 cache/metadata/backend 的问题。通过 thread-local context 注入。

**方式**: 精简实现
```python
# v1/forward_context.py
@dataclass
class ForwardContext:
    attn_metadata: dict
    kv_cache_config: object
    batch_descriptor: BatchDescriptor

def set_forward_context(ctx): ...
def get_forward_context() -> ForwardContext: ...
```

---

## C042 — AttentionBackend Registry

**来源**: `Principle/C033-C050/C033-C050.md` #C042 节

C032 的 backend 固定。C042 增加动态选择能力，支持 Torch/Flash/Triton 三种 backend。

**方式**: 精简实现，只加 registry 抽象层和枚举，不实现 Flash/Triton backend（属于 P2）。
```python
# v1/attention/registry.py
class AttentionBackendEnum(Enum):
    TORCH_SDPA = "torch_sdpa"
    FLASH_ATTN = "flash_attn"    # stub
    TRITON_ATTN = "triton_attn"  # stub
```

---

## C053 — Async Scheduler

**来源**: `Principle/C051-C070/C051-C070.md` #C053 节

Scheduler 支持异步调度模式。`schedule()` 在 background task 持续运行，EngineCore.step() 非阻塞等待。

**方式**: 精简实现
```python
# v1/core/sched/async_scheduler.py
class AsyncScheduler(SchedulerInterface):
    def schedule(self): ...
    
class BackgroundScheduleLoop:
    """asyncio task 持续调用 schedule()"""
```

---

## C054b — EngineCore Lifecycle

**来源**: `Principle/C051-C070/C051-C070.md` #C054b 节

补齐 EngineCore 核心执行循环的必要方法。

**方式**: 精简实现
```python
# v1/engine/core.py
def profile(self):                # KV cache sizing
def abort_requests(self, req_ids): # 外部 abort
def reset_prefix_cache(self):      # reset prefix cache
```

---

## C068 — KVCacheManager 扩展

**来源**: `Principle/C051-C070/C051-C070.md` #C068 节

C034 已规划 3 种 manager (被放弃)。C068 补齐 3 种关键类型的分析：CrossAttention、Mamba、ChunkedLocal。

**方式**: 分析文档 `docs/analysis/kv_cache_manager_types.md`，不实现代码。

---

## C101 — MoE DP (Data Parallel)

**来源**: `Principle/C101-C120/C101-C120.md` #C101 节

将 MoE + EP 扩展到 Data Parallel——token batch 维度切分，多 GPU 各自处理自己的 batch，通过 wave 协调同步。

**方式**: 精简实现
```python
# v1/distributed/moe_dp.py
class MoEDPGroup:
    """Data Parallel for MoE: batch split + wave sync"""
    def wave_sync(self, tokens, router_indices): ...
```

---

## C103 — DP×EP×TP 3D Topology

**来源**: `Principle/C101-C120/C101-C120.md` #C103 节

分析 3D 并行（DP×EP×TP）的拓扑布局、通信开销矩阵和 tensor sharding 兼容性。

**方式**: 分析文档 `docs/analysis/3d_topology.md`，包含：
- DP×EP×TP 的 GPU 布局示例
- 通信成本矩阵（all-reduce vs all-to-all vs all-gather）
- 实际生产案例（DeepSeek-V3 的 DP×EP×TP 部署拓扑）

---

## C070, C100, C118, C120 — 集成收敛

这些是集成测试和文档收敛阶段：

- **C070**: C051-C070 集成测试 — 验证 Priority/Preemption/Async/KV Metrics 全链路
- **C100**: C071-C100 集成测试 — 验证 TP/PP/EP/CP/MoE/Ray/K8s 全链路
- **C118**: Extended Benchmark Suite — 扩展 C098 的 benchmark 覆盖
- **C120**: C101-C120 集成 — DP + KV Connector 的完整测试

**方式**: 只跑已有测试，不新增功能代码。

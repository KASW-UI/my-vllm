# C120 最终架构 — C100 补充集

> C120 是 C100 的**补充集**，不是独立阶段。C100 已完成 Runtime 主干 + 分布式 + 现代模型 + 集群调度。C120 补齐 2 个深度实现 + 8 个深度分析（每个附带最小验证 demo）。
> 
> C120 每个类名、方法名、dataclass 字段必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。

## 一、C120 文件树（C100 增量部分）

> 完整 C100 文件树见 `Principle/C071-C100/C100-Final.md`。以下仅列出 C101-C120 新增文件。

```
my_vllm/                                      # C100 已有 ~90 个文件
│
├── v1/                                       # C100 已有
│   ├── engine/
│   │   ├── coordinator.py                 # C101: DPGroup, WaveScheduler
│   │   │                                    #   对齐 vLLM DPEngineCoreProc
│   │   └── weight_transfer_analysis.md       # C110: RL Training 权重同步分析
│   │
│   ├── core/
│   │   └── kv_connector.py                   # C102: KVConnectorBase,
│   │                                        #   SharedMemoryKVConnector
│   │                                        #   对齐 vLLM v1/core/kv_connector.py
│   │
│   ├── worker/
│   │   └── chunked_prefill_analysis.md       # C107: prefill chunking + hybrid batch
│   │
│   ├── attention/backends/
│   │   └── flashinfer_analysis.md            # C105: FlashInfer 性能分析
│   │
│   └── spec_decode/
│       └── proposer_comparison.md            # C111: 5 种 proposer 对比
│
├── mla/                                       # C100 已有 mla_layer.py
│   ├── backends_analysis.md                  # C104: 6 backends 对比分析
│   └── demo/
│       └── mla_kernel_comparison.py           # C104 demo: PyTorch vs Triton MLA
│
├── distributed/                               # C100 已有
│   ├── custom_all_reduce_analysis.md         # C106: custom all-reduce 拓扑 + 源码
│   ├── elastic_ep_analysis.md                # C109: Elastic EP 状态机 + 流程
│   └── demo/
│       ├── all_reduce_topology.py             # C106 demo: GPU 通信拓扑 + 带宽
│       └── elastic_ep_state_machine.py        # C109 demo: 扩缩容状态转换
│
├── model_executor/layers/quantization/
│   ├── methods_analysis.md                   # C108: FP8/AWQ/GPTQ/Marlin/NVFP4 全景
│   └── demo/
│       └── weight_packing_demo.py             # C108 demo: AWQ/GPTQ/Marlin packing
│
├── v1/attention/backends/demo/
│   └── flashinfer_paged_demo.py              # C105 demo: FlashInfer paged decode
│
├── v1/worker/demo/
│   └── chunked_prefill_demo.py               # C107 demo: prefill chunk scheduling
│
├── v1/engine/demo/
│   └── weight_transfer_demo.py               # C110 demo: save/load cycle
│
├── v1/spec_decode/demo/
│   └── draft_quality_comparison.py           # C111 demo: acceptance rate 对比
│
└── analysis/
    ├── dp_ep_tp_topology.md                  # C103: 3D 并行拓扑 + 决策树
    └── c100_vs_c120.md                       # C119: C100→C120 综合能力对比
```

**数量对照**

| 来源 | my_vllm/ 文件数 |
|------|---------------------|
| C100 | ~90 |
| C120 (增量) | +22（8 分析.md + 8 demo.py + 2 实现文件 + 4 工程文件） |
| C120 (总计) | ~112 |
| vLLM V1 `v1/` 目录 | ~220 |

---

## 二、C120 新增组件与职责边界

```
C100 完整引擎
  │
  │  ┌─ C120 新增: 分布式补齐 ─────────────────────────────────────────┐
  │  │                                                                  │
  │  ├── DP Coordinator (v1/engine/coordinator.py)                   │
  │  │   DPGroup: wave sync + batch distribution                       │
  │  │   WaveScheduler: DP-aware request routing                       │
  │  │   对齐 vLLM DPEngineCoreProc                                     │
  │  │                                                                  │
  │  └── KV Connector (v1/core/kv_connector.py)                        │
  │      KVConnectorBase: producer/consumer abstract                   │
  │      SharedMemoryKVConnector: same-node shared memory transfer    │
  │      KVConnectorMetadata: block_id + group_id + seq_num            │
  │      对齐 vLLM v1/core/kv_connector.py                              │
  │  └──────────────────────────────────────────────────────────────────┘
  │
  │  ┌─ C120 新增: 深度分析 (Analysis + Demo) ─────────────────────────┐
  │  │                                                                  │
  │  ├── C104 MLA 6 backends: why 6, per-backend diff, demo comparison │
  │  ├── C105 FlashInfer: paged KV cache API, production standard      │
  │  ├── C106 Custom all-reduce: topology + bandwidth, TP scaling      │
  │  ├── C107 Chunked prefill: hybrid batch strategy, TTFT tradeoff    │
  │  ├── C108 量化全景: FP8/AWQ/GPTQ/Marlin/NVFP4, weight packing      │
  │  ├── C109 Elastic EP: state machine, scale without downtime        │
  │  ├── C110 Weight Transfer: RL training sync, save/load cycle       │
  │  └── C111 dFlash/Gemma4: 5 proposer comparison, draft quality      │
  │  └──────────────────────────────────────────────────────────────────┘
```

---

## 三、C120 新增核心数据结构（对齐 vLLM）

### 1. DPGroup（`v1/engine/coordinator.py`）

```python
class DPGroup:
    """管理 DP rank 间的 batch 分配与 wave 同步。

    对齐 vLLM: DPEngineCoreProc._has_global_unfinished_reqs()
    """
    def __init__(self, dp_size, dp_rank):
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.sync_interval = 32

    def has_global_unfinished_reqs(self, local_unfinished: int) -> bool:
        """All-reduce 所有 DP rank 的 unfinished count。"""

class WaveScheduler:
    """将 request batch 分配到各 DP rank。
    对齐 vLLM: request.data_parallel_rank 路由。
    """
    def __init__(self, dp_size, dp_rank): ...
    def select_requests(self, all_requests: list, dp_rank: int) -> list: ...
```

### 2. KVConnector（`v1/core/kv_connector.py`）

```python
@dataclass
class KVConnectorMetadata:
    block_ids: list[int]
    kv_cache_group_id: int
    num_tokens_per_block: int
    sequence_id: int

class KVConnectorBase(ABC):
    """分布式 KV cache transfer 抽象。
    对齐 vLLM: v1/core/kv_connector.py — KVConnectorBase_V1
    """
    def start_transfer(self, metadata: KVConnectorMetadata) -> None: ...
    def check_transfer_done(self, block_id: int) -> bool: ...
    def recv_blocks(self) -> list[KVConnectorMetadata]: ...

class SharedMemoryKVConnector(KVConnectorBase):
    """同 node 内共享内存传输。
    对齐 vLLM: KVConnectorOutput / KVConnectorInput
    """
    def __init__(self, kv_cache_config): ...
    def start_transfer(self, metadata): ...
    def recv_blocks(self): ...
```

### 3. DP × EP × TP 组合决策树（`analysis/dp_ep_tp_topology.md`）

```
│─ 模型大小 > 70B? ──→ TP first (weight 切分必须)
│
├─ MoE 模型? ──→ EP + DP (expert 分布 + batch 分布)
│
├─ 序列 > 128K? ──→ CP first (序列切分)
│
├─ 吞吐 SLA > 延迟 SLA? ──→ DP first (独立 step loop, no bubble)
│
└─ 延迟 SLA > 吞吐 SLA? ──→ PP (pipeline hide per-token latency)
```

---

## 四、C120 新增数据流

```
C100 完整数据流
  │
  ┌─ DP-aware Engine Loop ──────────────────────────────────────────────┐
  │                                                                      │
  │ DPEngineCore.step_with_batch_queue():                                │
  │   for each step:                                                     │
  │     1. schedule() → local scheduler_output each DP rank             │
  │     2. execute_model(scheduler_output)                                │
  │     3. every 32 steps: dp_all_gather_reqs(local_unfinished)          │
  │        → global_unfinished → continue or idle                        │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  ┌─ KV Connector Flow ─────────────────────────────────────────────────┐
  │                                                                      │
  │ Producer (source GPU):                                               │
  │   1. Scheduler marks block as PENDING_TRANSFER                       │
  │   2. KVConnector.start_transfer(metadata)                            │
  │      → serialize K,V tensors → push to shared memory queue           │
  │   3. BlockPool.free_blocks() notifies connector done                  │
  │                                                                      │
  │ Consumer (target GPU):                                               │
  │   1. KVConnector.recv_blocks() pulls from shared memory queue         │
  │   2. deserialize K,V → write to local KV cache via BlockPool         │
  │   3. Mark block as READY → scheduler can schedule request            │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘

  ┌─ Elastic EP Scale Flow ─────────────────────────────────────────────┐
  │                                                                      │
  │ Scale up: EP=2 → EP=4                                               │
  │   1. NEW_CORE_ENGINES_INIT_READY: new GPU loads model weights        │
  │   2. WEIGHTS_TRANSFERRING: expert weights migrate to new GPU         │
  │      (via KV Connector for KV cache, collective broadcast for W)     │
  │   3. RECONFIGURE_FINISHED: DP client routing table updated           │
  │   4. 新 request 使用 EP=4 routing                                    │
  │                                                                      │
  │ Scale down: EP=4 → EP=2                                             │
  │   1. RECONFIGURE: target GPUs marked for shutdown                    │
  │   2. DRAINING: 不分配新 request 到 target GPUs                       │
  │   3. WAIT_FOR_DRAIN: 等待已在 target GPU 上的 request 完成            │
  │   4. SHUTDOWN_COMPLETE: 释放 GPU 资源                                │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 五、逐文件对比（C100 → C120 → vLLM）

| C120 文件 | C100 状态 | C120 vs vLLM |
|-----------|-----------|----------------|
| `v1/engine/coordinator.py` | **不存在** | DPGroup + WaveScheduler（对齐 vLLM DPEngineCoreProc） |
| `v1/core/kv_connector.py` | **不存在** | KVConnectorBase + SharedMemoryKVConnector（对齐 vLLM） |
| `mla/backends_analysis.md` | 不存在 | 6 backends 对比 + demo（对齐 vLLM `backends/mla/` 目录） |
| `mla/demo/mla_kernel_comparison.py` | 不存在 | PyTorch vs Triton MLA demo |
| `v1/attention/backends/flashinfer_analysis.md` | 不存在 | FlashInfer 性能分析（对齐 vLLM flashinfer backend） |
| `v1/attention/backends/demo/flashinfer_paged_demo.py` | 不存在 | FlashInfer paged decode demo |
| `distributed/custom_all_reduce_analysis.md` | 不存在 | Custom all-reduce 拓扑 + 源码分析（对齐 vLLM `device_communicators/`） |
| `distributed/demo/all_reduce_topology.py` | 不存在 | GPU 通信拓扑 + 带宽 demo |
| `v1/worker/chunked_prefill_analysis.md` | 不存在 | Chunked prefill + hybrid batch（对齐 vLLM scheduler.py prefill scheduling） |
| `v1/worker/demo/chunked_prefill_demo.py` | 不存在 | Prefill chunk scheduling demo |
| `model_executor/layers/quantization/methods_analysis.md` | 不存在 | FP8/AWQ/GPTQ/Marlin/NVFP4 全景 |
| `model_executor/layers/quantization/demo/weight_packing_demo.py` | 不存在 | Weight packing visualization demo |
| `distributed/elastic_ep_analysis.md` | 不存在 | Elastic EP 状态机 + 流程（对齐 vLLM `engine/core.py` EEP） |
| `distributed/demo/elastic_ep_state_machine.py` | 不存在 | EP 扩缩容状态转换 demo |
| `v1/engine/weight_transfer_analysis.md` | 不存在 | RL Training 权重同步（对齐 vLLM `save_sharded_state`） |
| `v1/engine/demo/weight_transfer_demo.py` | 不存在 | Save/load cycle demo |
| `v1/spec_decode/proposer_comparison.md` | 不存在 | 5 种 proposer 对比（对齐 vLLM `spec_decode/`） |
| `v1/spec_decode/demo/draft_quality_comparison.py` | 不存在 | Acceptance rate 对比 demo |
| `analysis/dp_ep_tp_topology.md` | 不存在 | 3D 并行拓扑 + 决策树 |

---

## 六、C120 架构差距总结

| 维度 | C100 | C120 | vLLM | 差距率 |
|------|------|------|------|--------|
| **引擎层** | Engine step/async 闭环 | + DP wave coordination | DPEngineCoreProc + DP coordinator | 10% |
| **KV Cache** | 5 种 Manager + CoW | + KV connector (分布式传输) | KV/EC connector 完整 | 10% |
| **MoE DP** | 单 GPU MoE + EP=2 | + DP=2 wave sync + DP×EP composition | MoE DP + elastic EP 完整 | 25% |
| **MLA** | 1 个分析文件 | + 6 backends 源码对比 + kernel demo | 20 个文件, full backends | 40% |
| **FlashInfer** | 无 | + 性能分析 + paged decode demo | Full FlashInfer backend | 35% |
| **Custom All-Reduce** | 无 | + 拓扑分析 + 带宽计算 demo | custom_all_reduce CUDA kernel | 30% |
| **Chunked Prefill** | 隐式在 ModelRunner | + 调度策略 + TTFT 分析 | vLLM scheduler chunked prefill complete | 25% |
| **量化** | FP8 + GPTQ stub | + AWQ/GPTQ/Marlin/NVFP4 全景分析 | Full kernel implementation | 25% |
| **Elastic EP** | 无 | + 状态机 + 扩缩容流程分析 | EEP 完整实现 | 30% |
| **Weight Transfer** | 无 | + RL training sync 分析 | save/load + collective broadcast 完整 | 30% |
| **Spec Decode** | Eagle/Medusa/NGram | + dFlash/Gemma4 对比 + draft quality | 完整 5 proposer | 15% |

**加权差距**（按 AI Infra 工程师重要性加权）：

```
C100: ~37% total gap → C120: ~20% total gap
差距缩小 ~17 个百分点
```

---

## 七、C120 完成后判断标准

1. **DP=2 + EP=2**: wave sync 正确，DP rank 各自独立 step loop
2. **KV Connector**: two workers, block transfer round-trip 正确，K,V tensors 数值一致
3. **DP×EP×TP 3D 分析**: 决策树正确，通信开销模型验证
4. **MLA 6 backends**: 对比分析完整，demo kernel 数值误差 < 1e-3
5. **FlashInfer**: 性能分析完整，paged decode demo 与 C050 reference 对比
6. **Custom all-reduce**: 拓扑分析 + 带宽计算正确，理解何时需要 custom
7. **Chunked prefill**: 调度策略分析完整，TTFT 对比正确
8. **量化全景**: 6 种方法原理 + weight packing demo
9. **Elastic EP**: 状态机分析 + 扩缩容流程正确
10. **Weight Transfer**: save/load cycle demo 正确
11. **dFlash/Gemma4**: 5 proposer 对比 + draft quality demo
12. **8 个 demo.py 全部可运行**: `python demo/xxx.py` 独立运行，有输出
13. **C119 综合对比文档**: C100 vs C120 能力矩阵
14. **pytest**: C120 新增测试全绿

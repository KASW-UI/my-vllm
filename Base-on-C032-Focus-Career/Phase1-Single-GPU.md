# Phase 1: 单 GPU 生产级 Runtime

> 基线: C032 | 目标: 单 GPU 引擎达到生产级质量，Scheduler/KV/Metrics 齐全
> 包含 11 个 P0 commits: C035, C036, C037, C038, C051, C052, C054c, C048, C054, C050, C069
>
> **真实 vLLM 路径映射**: 教育项目 `my_vllm/v1/distributed/` → 真实 vLLM `vllm/distributed/`（顶层）；`my_vllm/v1/model_executor/` → `vllm/model_executor/`（顶层）。阅读真实源码时需做路径翻译。

---

## 依赖拓扑

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
  ├─ C050  TP=2 真实 NCCL                   ← 分布式网关
  │
  └─ C069  Config 体系                       ← 独立，最后收尾
```

## 执行顺序

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 1 | C035: KVCache: BlockPool lifecycle upgrade + GC | C032 | `pytest tests/` |
| 2 | C036: KVCache: Prefix Cache Copy-on-Write | C035 | `pytest tests/` |
| 3 | C037: Scheduler: SchedulerInterface ABC | C032 | `python run.py` |
| 4 | C038: Scheduler: Resource model (GPU/KV/token budget) | C037 | `python run.py` |
| 5 | C051: Scheduler: Priority scheduling (FCFS/Priority) | C037 | `python run.py` |
| 6 | C052: Scheduler: Preemption + KV reclaim (RECOMPUTE/SWAP) | C051 | `python run.py` |
| 7 | C054c: KVCache: ref_cnt cascading release | C036 | `pytest tests/` |
| 8 | C048: Metrics: TTFT, TPOT, ITL, throughput, KV utilization | — | `python run.py` |
| 9 | C054: KVCache: Block metrics (reuse/residency/eviction/hit rate) | C035 | `python run.py` |
| 10 | C050: Distributed: TP=2 real NCCL | — | `python run.py` |
| 11 | C069: Config: VllmConfig / ModelConfig / CacheConfig | — | `python run.py` |

---

## C035 — BlockPool 生命周期升级

**来源**: `Principle/C033-C050/C033-C050.md` #C035 节

### 目标

C032 已有 `block_pool.py`（ref_cnt, free queue, LRU, hash cache）。C035 补齐生产级 block 池的两个缺失：
- **BlockHashToBlockMap**: hash → list[block]（处理哈希碰撞下的多 block 共享）
- **GC free_blocks()**: ref_cnt cascading release（引用计数级联释放）

### 修改文件

```
v1/core/block_pool.py       # BlockHashToBlockMap + free_blocks() + GC
v1/core/kv_cache_utils.py   # KVCacheBlockCopy(NamedTuple), BlockHashWithGroupId
```

### 核心实现

**BlockHashToBlockMap**:
```python
# C032: hash → block (1:1, 丢失哈希碰撞下的多 block)
# C035: hash → list[block] (1:N, 生产级)
# 对齐 vLLM: v1/core/block_pool.py BlockHashToBlockMap
class BlockHashToBlockMap:
    def get_one_block(self, block_hash):   # → KVCacheBlock | None (对齐 vLLM)
    def contain(self, block_hash):         # → bool
    def insert(self, block_hash, block):
    def pop(self, block_hash, block_id):   # → KVCacheBlock | None (对齐 vLLM)
```

**GC free_blocks()**:
```python
# ref_cnt=0 → FREE → reuse
# 生产环境需要处理级联释放: 释放父 block 时递归释放子 block
def free_blocks(self, blocks: list[KVCacheBlock]):
    for block in blocks:
        if block.ref_cnt > 0:
            block.ref_cnt -= 1
        if block.ref_cnt == 0:
            block.is_free = True
            # 从 hash map 中移除
            # 加入 free queue
```

**KVCacheBlockCopy** (新增):
```python
class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int
```

### 依赖

C032 的 BlockPool 基线（ref_cnt, hash cache, free queue）。

### 验证

`pytest tests/` 全部通过。

### AGENTS.md update

- `Package layout` section: 无新增文件
- `Key architecture facts` section: "Prefix caching" 行更新为 "BlockHashToBlockMap (hash→list[block]), free_blocks() with ref_cnt cascading release, GC 驱逐逻辑"

---

## C036 — Prefix Cache Copy-on-Write

**来源**: `Principle/C033-C050/C033-C050.md` #C036 节

### 目标

两个 request 共享 prefix block 时，B 写入不能污染 A 的 KV cache。C029 已有 `cached_block_hash_to_block`，但缺少 CoW 语义。

### 文件

```
v1/core/prefix_cache.py      (新增)
v1/core/block_pool.py         (修改: CoW 感知的 allocate)
v1/core/kv_cache_manager.py   (修改: CoW 感知的 KV 写入)
```

### 核心实现

```python
# prefix_cache.py
class PrefixCache:
    def get_block_for_write(self, block_hash, block_id):
        """共享块写时复制: ref_cnt > 1 → copy → decrement → private block"""
        if block.ref_cnt > 1:
            new_block = self.block_pool.get_new_block()
            copy_kv_block(block, new_block)  # 复制 KV 数据
            block.ref_cnt -= 1
            return new_block
        return block
```

流程:
```
shared block (ref_cnt=2)
       │
      CoW 触发 (写入检测)
       │
      copy block (ref_cnt=1)
       │
  private block (可安全写入)
```

### 依赖

C035 的 BlockHashToBlockMap 和 KVCacheBlockCopy。

### 验证

`pytest tests/` — 验证两个 request 共享 prefix 后各自写入不互相污染。

### AGENTS.md update

- `Package layout` section: `core/` 注释更新为 `BlockPool, KVCacheManager, kv_cache_utils, kv_cache_coordinator, prefix_cache`
- `Key architecture facts` section: "Prefix caching" 行更新为 "Prefix caching with CoW: shared blocks copied on write (ref_cnt > 1 → copy → decrement → private block)"

---

## C037 — SchedulerInterface ABC

**来源**: `Principle/C033-C050/C033-C050.md` #C037 节

### 目标

C032 的 scheduler 是单体 `scheduler.py`（FCFS + chunked prefill + update_from_output）。未来要加入 priority/preemption/async，需要先建立标准接口。

### 文件

```
v1/core/sched/interface.py    (新增)
v1/core/sched/scheduler.py    (修改: 实现 SchedulerInterface)
v1/core/sched/output.py       (不改, 已有)
```

### 核心实现

```python
# v1/core/sched/interface.py
class SchedulerInterface(ABC):
    @abstractmethod
    def schedule(self) -> SchedulerOutput: ...
    
    @abstractmethod
    def update_from_output(
        self, scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]: ...
    
    @abstractmethod
    def add_request(self, request: Request) -> None: ...
    
    @abstractmethod
    def finish_requests(
        self, request_ids, finished_status
    ) -> list[Request]: ...
    
    @abstractmethod
    def get_num_unfinished_requests(self) -> int: ...
    
    @abstractmethod
    def has_unfinished_requests(self) -> bool: ...
```

### 依赖

C032 的 scheduler.py 和 scheduler output。

### 验证

`python run.py` — 引擎执行链不变（Scheduler 继承 ABC 后行为一致）。

### AGENTS.md update

- `Package layout` section: `core/sched/` 注释更新为 `Scheduler, SchedulerInterface, SchedulerOutput, FCFSRequestQueue`
- `Key architecture facts` section: 新增 "SchedulerInterface ABC: schedule() → SchedulerOutput, update_from_output(), finish_requests()"

---

## C038 — Scheduler Resource Model

**来源**: `Principle/C033-C050/C033-C050.md` #C038 节

### 目标

C032 的 SchedulerOutput 已经有 `scheduled_new_reqs`、`num_scheduled_tokens` 等字段，但调度决策未显式建模 GPU/KV/token 三类资源预算。C038 补齐资源感知调度。

### 修改文件

```
v1/core/sched/scheduler.py    # schedule() 内增加资源预算检查
v1/core/sched/output.py       # SchedulerOutput 增加字段
```

### 核心实现

调度决策不再只看 `max_running`，还要检查：

```python
def schedule(self):
    for request in self.waiting_queue:
        # GPU budget: max_num_seqs 是否已满
        # KV budget: 是否有足够的 free blocks
        # token budget: num_scheduled_tokens <= max_num_batched_tokens
        
        if (gpu_ok and kv_ok and token_ok):
            schedule_request(request)
        else:
            break  # 任一资源不足，停止调度
```

**SchedulerOutput 扩展**:
```python
@dataclass
class SchedulerOutput:
    # C032 已有字段
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    
    # C038 新增
    num_common_prefix_blocks: list[int]       # 每个 req 的共享 prefix block 数
    finished_req_ids: set[str]
    preempted_req_ids: set[str] | None        # 预留给 C052 实现
```

### 依赖

C037 的 SchedulerInterface。

### 验证

`python run.py` — 资源限制生效（max_num_seqs/max_num_batched_tokens 边界正确）。

### AGENTS.md update

- `Key architecture facts` section: "Scheduler" 描述更新为 "FCFS + chunked-prefill + prefix-cache aware + resource model (GPU budget, KV budget, token budget) + preemption"

---

## C051 — Priority Scheduler

**来源**: `Principle/C051-C070/C051-C070.md` #C051 节

### 目标

Scheduler 从 FCFS 演化为多策略调度器。这是 C052 (Preemption) 的前置依赖——需要知道谁可被抢占。

### 文件

```
v1/core/sched/request_queue.py  (新增)
v1/core/sched/scheduler.py      (修改: 注入 policy，schedule() 委托 policy.select())
v1/core/sched/output.py         (修改: SchedulerOutput 传递 priority metadata)
```

### 核心实现

```python
# v1/core/sched/request_queue.py
# 对齐 vLLM: v1/core/sched/request_queue.py SchedulingPolicy, FCFSRequestQueue, PriorityRequestQueue

class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"

class FCFSRequestQueue(deque, RequestQueue):
    """C032 已有, C051 提取为独立队列类"""

class PriorityRequestQueue:
    """基于 heap 的优先级队列 (对齐 vLLM PriorityRequestQueue)"""
    def push(self, request: Request): ...
    def pop_request(self) -> Request:
        """弹出优先级最高的单个请求 (对齐 vLLM pop_request), 高 priority 优先"""
    def is_empty(self) -> bool: ...

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """工厂函数 (对齐 vLLM create_request_queue)"""
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
```

### 依赖

C037 的 SchedulerInterface（提供可替换的调度契约）。

### 验证

`python run.py` — FCFS/Priority 两种模式可切换，优先级调度结果正确。

### AGENTS.md update

- `Package layout` section: `core/sched/` 注释更新提及 `request_queue.py (SchedulingPolicy, FCFSRequestQueue, PriorityRequestQueue, create_request_queue)`
- `Key architecture facts` section: "Scheduler" 描述更新为 "FCFS + Priority scheduling"

---

## C052 — Preemption + KV Reclaim

**来源**: `Principle/C051-C070/C051-C070.md` #C052 节

### 目标

当 GPU/KV 资源不足时，Scheduler 抢占运行中的低优先级请求，释放 KV block 供新请求使用。这是 KV 跨机迁移的前序知识。

### 文件

```
v1/core/sched/scheduler.py       # schedule() 内 preempt 逻辑
v1/core/block_pool.py            # evict_blocks() 按 priority 选择 victim
v1/core/kv_cache_manager.py      # free_blocks() 支持 SWAP
v1/request.py                    # PREEMPTED 状态的 resume/retry 逻辑
```

### 核心实现

```python
class PreemptionMode(Enum):
    RECOMPUTE = "recompute"   # 释放 KV block，重新 prefill
    SWAP = "swap"             # 将 KV block 搬到 CPU，resume 时搬回

# scheduler.py
def schedule(self):
    if not enough_resources(new_request):
        victims = self._select_preemption_victims()
        for victim in victims:
            if self.preemption_mode == PreemptionMode.RECOMPUTE:
                self.kv_cache_manager.free_blocks(victim.block_ids)
                victim.status = RequestStatus.PREEMPTED
                victim.num_computed_tokens = 0  # 需要重新 prefill
            elif self.preemption_mode == PreemptionMode.SWAP:
                self.kv_cache_manager.swap_out(victim.block_ids)
                victim.status = RequestStatus.PREEMPTED
```

**Victim 选择**: 按 priority 从低到高选择 `preempted_req_ids`。

### 依赖

C051 的 Priority Scheduler（决定抢谁）。

### 验证

`python run.py` — 发送超过 KV 容量的请求，低优先级请求被抢占，高优先级继续执行。

### AGENTS.md update

- `Key architecture facts` section: "Scheduler" 描述更新为 "...+ Preemption (RECOMPUTE/SWAP mode)", 新增 "Request PREEMPTED 状态支持 resume/retry"

---

## C054c — KVCacheBlock ref_cnt 级联释放

**来源**: `Principle/C051-C070/C051-C070.md` #C054c 节

### 目标

完善 block 池的引用计数体系。prefix cache 共享 block 时，B 写不能污染 A（C036 建立 CoW 语义），但没有 ref_cnt 级联释放和 allocate_slots 中的 CoW 感知分配，共享 block 可能被意外释放或漏释放。

### 文件

```
v1/core/block_pool.py          # free_blocks() 增加级联递减
v1/core/kv_cache_manager.py    # allocate_slots() 中 CoW 感知分配
```

### 核心实现

```python
# block_pool.py
def free_blocks(self, blocks: list[KVCacheBlock]):
    """级联递减: 释放时递归递减 ref_cnt"""
    for block in blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            self._really_free(block)
            # 递归处理子 block (cascading)
            for child in self._get_children(block):
                self.free_blocks([child])

# kv_cache_manager.py
def allocate_slots(self, request, num_new_tokens):
    """CoW 感知分配: block.ref_cnt > 1 → copy → allocate new → use copy"""
    for block in self.get_cached_blocks(request):
        if block.ref_cnt > 1:
            copy = self.block_pool.get_new_block()
            KVCacheBlockCopy(src=block.block_id, dst=copy.block_id)
            block.ref_cnt -= 1
            yield copy
        else:
            yield block
```

### 依赖

C036 的 CoW 基础。

### 验证

`pytest tests/` — 多 request 共享 prefix + 独立写入 + 释放，验证 ref_cnt 不掉到负数、block 不泄漏。

### AGENTS.md update

- `Key architecture facts` section: "Prefix caching" 描述更新为 "...+ free_blocks() 级联递减 ref_cnt, allocate_slots() CoW 感知分配"

---

## C048 — Metrics

**来源**: `Principle/C033-C050/C033-C050.md` #C048 节

### 目标

补齐 LLM Serving 关键性能指标。C032 只有基础 metrics/collector.py。C048 增加：

- **TTFT** (Time To First Token)
- **TPOT** (Time Per Output Token)
- **ITL** (Inter-Token Latency)
- **Throughput** (tokens/s)
- **KV Cache Utilization** (used_blocks / total_blocks)

### 文件

```
v1/metrics/stats.py        (新增)
v1/metrics/loggers.py      (新增)
v1/metrics/prometheus.py   (新增, stub)
```

### 核心实现

```python
# v1/metrics/stats.py
@dataclass
class SchedulerStats:
    num_running_reqs: int
    num_waiting_reqs: int
    kv_cache_usage: float       # 0.0 ~ 1.0
    prefix_cache_hit_rate: float

@dataclass
class IterationStats:
    ttft_ms: float | None       # 本次 step 首次生成 token 的请求的 TTFT
    tpot_ms: float              # 本次 step 的 per-token 延迟
    itl_ms: float               # 连续两次 token 之间的间隔
    prompt_throughput: float    # prefill tokens/s
    generation_throughput: float # decode tokens/s

# 采集点: EngineCore.step() 的 schedule / update_from_output 边界
```

### 依赖

无（独立模块）。

### 验证

`python run.py` — 日志输出包含 TTFT/TPOT/ITL/throughput/KV util。

### AGENTS.md update

- `Key architecture facts` section: 新增 "Metrics: TTFT, TPOT, ITL, throughput, KV cache utilization"

---

## C054 — KV Cache Metrics

**来源**: `Principle/C051-C070/C051-C070.md` #C054 节

### 目标

block 生命周期可观测。在 BlockPool.allocate/free/touch/evict 事件上挂载 metrics 统计。

### 文件

```
v1/core/kv_cache_metrics.py  (新增)
v1/core/block_pool.py         (修改: 事件通知)
```

### 核心实现

```python
# v1/core/kv_cache_metrics.py
# 对齐 vLLM: v1/core/kv_cache_metrics.py BlockMetricsState + KVCacheMetricsCollector

@dataclass
class BlockMetricsState:
    """每个 block 的生存周期指标 (对齐 vLLM BlockMetricsState)"""
    reuse_interval: dict[int, float]     # 对齐 KVCacheEvictionEvent.reuse_gaps_seconds
    residency: dict[int, float]          # 对齐 KVCacheEvictionEvent.lifetime_seconds
    eviction_count: int
    hit_rate: float
    
class KVCacheMetricsCollector:
    """收集器 (对齐 vLLM KVCacheMetricsCollector)"""
    def on_block_allocated(self, block: KVCacheBlock):
    def on_block_accessed(self, block: KVCacheBlock):
    def on_block_evicted(self, block: KVCacheBlock):
```

### 依赖

C035 的 BlockPool 增强。

### 验证

`python run.py` — 日志输出 reuse interval / residency / eviction count / hit rate。

### AGENTS.md update

- `Key architecture facts` section: 新增 "KV Metrics: reuse interval, residency, eviction count, prefix cache hit rate"

---

## C050 — TP=2 Weight Sharding (Minimal)

**来源**: `Principle/C033-C050/C033-C050.md` #C050 节 + `C050-Final.md`

### 目标

C032 的 TP 是 stub（world_size=1 时 identity）。C050 实现 ColumnParallelLinear 和 RowParallelLinear 的**最低可行多 GPU 版本**：直接在 layer 代码中内联 `torch.distributed.all_reduce/all_gather`，不做 init_process_group、不做 GroupCoordinator 抽象（留给 C083）。

### 文件

```
v1/model_executor/layers/linear.py    # 修改: ColumnParallelLinear/RowParallelLinear 内联 torch.distributed 调用
```

### 核心实现

```python
# v1/model_executor/layers/linear.py
# C032: 所有通信都是 identity (world_size=1 时直接返回)
# C050: 内联真实的 torch.distributed 调用

class ColumnParallelLinear(nn.Module):
    def forward(self, x):
        # 每个 GPU 持有权重的 1/tp_size 列
        local_output = F.linear(x, self.weight)  # [batch, out_dim // tp_size]
        if self.world_size == 1:
            return local_output
        # all_gather 拼接各 GPU 的输出
        gathered = [torch.empty_like(local_output) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, local_output)
        return torch.cat(gathered, dim=-1)

class RowParallelLinear(nn.Module):
    def forward(self, x):
        # 每个 GPU 持有权重的 1/tp_size 行
        local_output = F.linear(x, self.weight)  # [batch, out_dim] (partial sum)
        if self.world_size == 1:
            return local_output
        # all_reduce 聚合各 GPU 的 partial sum
        torch.distributed.all_reduce(local_output)
        return local_output
```

**与 C083 的分工**:
- C050: 最小实现——内联通信原语，假设进程组已外部初始化（launcher 设置）
- C083: 工业级——init_process_group(nccl) + GroupCoordinator 抽象 + broadcast + communication_op wrapper

### 依赖

C032 的 TP stub 和 model_executor/layers/linear.py。

### 验证

`torchrun --nproc_per_node=2 run.py` — TP=2 运行 OPT-125m，输出与单 GPU 一致。

### AGENTS.md update

- `Key architecture facts` section: "Tensor parallel" 行更新为 "TP=2: ColumnParallelLinear (all_gather) + RowParallelLinear (all_reduce), 内联 torch.distributed 调用"

---

## C069 — Config 体系

**来源**: `Principle/C051-C070/C051-C070.md` #C069 节

### 目标

C032 只有 42 行 `EngineConfig`。C069 拆出 3 个最小必要配置类，为 Phase 2 分布式配置提供基础。

### 文件

```
v1/config.py → v1/config/
  __init__.py      # 顶层导出
  vllm.py          # VllmConfig (聚合 — 对齐 vLLM vllm/config/vllm.py)
  model.py         # ModelConfig (对齐 vLLM vllm/config/model.py)
  cache.py         # CacheConfig (对齐 vLLM vllm/config/cache.py)
  scheduler.py     # SchedulerConfig (对齐 vLLM vllm/config/scheduler.py)
```

### 核心实现

```python
# v1/config/model.py  — 对齐 vLLM vllm/config/model.py (精简版)
@dataclass
class ModelConfig:
    model_name: str = "facebook/opt-125m"
    dtype: str = "bfloat16"
    max_model_len: int = 2048
    seed: int = 0

# v1/config/cache.py  — 对齐 vLLM vllm/config/cache.py (精简版)
@dataclass
class CacheConfig:
    block_size: int = 16
    num_kv_blocks: int = 512
    enable_prefix_caching: bool = True

# v1/config/scheduler.py  — 对齐 vLLM vllm/config/scheduler.py (精简版)
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    enable_chunked_prefill: bool = True
    policy: str = "fcfs"

# v1/config/vllm.py  — 对齐 vLLM vllm/config/vllm.py (精简版)
@dataclass
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    scheduler_config: SchedulerConfig   # 对齐 vLLM: 独立配置类, 非内联
    tensor_parallel_size: int = 1
```

### 依赖

无（独立模块）。

### 验证

`python run.py` — 所有现有路径使用新 VllmConfig 运行正常。

### AGENTS.md update

- `Package layout` section: `v1/config/` 行更新，注释 `VllmConfig, ModelConfig, CacheConfig`
- `Configuration` section: 更新 config 路径为 `my_vllm/v1/config/`

---

## Phase 1 完成后的能力

- ✅ 生产级 BlockPool: GC + CoW + 级联释放
- ✅ 标准化 Scheduler: ABC + 资源模型 + Priority + Preemption + KV reclaim
- ✅ Metrics: TTFT/TPOT/ITL/throughput/KV utilization + BlockMetrics
- ✅ 真实 TP=2 多 GPU 通信
- ✅ 标准化 Config 体系
- ✅ 所有修改 `python run.py` 可运行

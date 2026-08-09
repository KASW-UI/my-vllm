# Phase 2: 分布式 Runtime

> 基线: C050 (TP=2 内联通信) | 目标: 4 种并行策略全部跑通，含组合拓扑分析
> 包含 7 个 P0 commits: C083, C080, C081, C082, C084, C085, C086
> 
> **真实 vLLM 路径映射**: 教育项目 `my_vllm/v1/distributed/` → 真实 vLLM `vllm/distributed/`（顶层，不在 v1 内）；`my_vllm/v1/model_executor/` → `vllm/model_executor/`（顶层）。阅读真实源码时需做路径翻译。

---

## 依赖拓扑

```
C050 (TP=2 内联通信, 无 NCCL init)
  │
  ├─ C083  TP=2 工业级 NCCL         ← 需要 C050: init_process_group + GroupCoordinator
  │   │
  │   ├─ C080  MultiprocExecutor     ← 需要 C083: 多进程 executor 基础设施
  │   │
  │   ├─ C084  PP Pipeline Parallel   ← 需要 C083 + C080
  │   └─ C085  CP Ring Attention      ← 需要 C083
  │
  ├─ C081  MoE fused kernel           ← 独立模块
  │   └─ C082  EP All-to-All          ← 需要 C081
  │
  └─ C086  Multi-GPU Integration      ← 需要 C080 + C082 + C084 + C085
```

## 执行顺序

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 12 | C083: Distributed: TP=2 industrial NCCL (init_process_group + GroupCoordinator) | C050 | `torchrun --nproc_per_node=2 run.py` |
| 13 | C080: Executor: MultiprocExecutor (WorkerProcess lifecycle) | C083 | `python run.py` |
| 14 | C081: MoE: TopK router + fused Triton expert kernel | — | `python run.py` |
| 15 | C082: Distributed: EP All-to-All dispatch/combine | C081 | `python run.py` |
| 16 | C084: Distributed: PP Pipeline Parallel (1F1B-lite) | C083, C080 | `python run.py` |
| 17 | C085: Distributed: CP Ring Attention | C083 | `python run.py` |
| 18 | C086: Distributed: TP×PP×EP×CP multi-GPU integration | C080, C082, C084, C085 | `python run.py` |

---

## C083 — TP=2 Industrial NCCL

**来源**: `Principle/C071-C100/C070-C100.md` #C083 节 + 真实 vLLM `vllm/distributed/parallel_state.py`

### 目标

C050 已在 ColumnParallelLinear/RowParallelLinear 中内联 `torch.distributed` 调用（all_reduce/all_gather），但缺少：
- `init_process_group(nccl)` — 进程组初始化
- `GroupCoordinator` 抽象 — 封装 world_size/rank/group，提供统一的通信接口
- `broadcast` — TP 中权重初始同步
- 多 process group 管理 — 为 PP/EP/CP 预留

C083 补齐这些基础设施，不对 C050 的 layer 代码做结构性修改——只是将内联的 `torch.distributed` 调用替换为 `GroupCoordinator` 方法调用。

### 文件

```
v1/distributed/
  parallel_state.py       # 新增: initialize_model_parallel(nccl), GroupCoordinator 类
                           #   all_reduce → GroupCoordinator.all_reduce
                           #   all_gather → GroupCoordinator.all_gather
                           #   broadcast  → GroupCoordinator.broadcast
                           #   get_tp_group().world_size / .rank

v1/model_executor/layers/linear.py  # 可选修改: 将内联 torch.distributed 调用替换为 GroupCoordinator 方法
```

### 核心实现

```python
# v1/distributed/parallel_state.py
import torch.distributed as dist

_TP: GroupCoordinator | None = None

class GroupCoordinator:
    """封装一个 process group 的通信操作。
    
    与 C050 的区别:
    - C050: 内联 torch.distributed.all_reduce(all_gather(...)) 在 layer 代码中
    - C083: 提取为 GroupCoordinator，提供统一的 world_size/rank/group 管理
    """
    def __init__(self, group, device, world_size, rank):
        self.group = group
        self.device = device
        self.world_size = world_size
        self.rank = rank

    def all_reduce(self, tensor):
        if self.world_size == 1: return tensor
        dist.all_reduce(tensor, group=self.group)
        return tensor

    def all_gather(self, tensor, dim=-1):
        if self.world_size == 1: return tensor
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor, group=self.group)
        return torch.cat(gathered, dim=dim)

    def broadcast(self, tensor, src=0):
        if self.world_size == 1: return tensor
        dist.broadcast(tensor, src=src, group=self.group)
        return tensor

def initialize_model_parallel(tp_size, pp_size=1, backend="nccl"):
    """初始化分布式模型并行 (对齐 vLLM initialize_model_parallel)
    
    C083 新增: C050 假设 launcher 已设置分布式环境，
    C083 显式调用 init_process_group。
    """
    dist.init_process_group(backend=backend, rank=rank, world_size=tp_size * pp_size)
    group = dist.new_group(ranks=list(range(tp_size)))
    _TP = GroupCoordinator(group, f"cuda:{local_rank}", tp_size, rank)
    _TP = GroupCoordinator(group, f"cuda:{local_rank}", tp_size, rank)

def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "TP group not initialized"
    return _TP
```

### 与 C050 的分工

| 职责 | C050 | C083 |
|---|---|---|
| ColumnParallelLinear/RowParallelLinear | ✅ 内联 torch.distributed | 可选替换为 GroupCoordinator |
| torch.distributed.all_reduce/all_gather | ✅ 直接调用 | 封装到 GroupCoordinator |
| init_process_group(nccl) | ❌ 假设 launcher 已设 | ✅ 显式初始化 |
| GroupCoordinator 抽象 | ❌ 无 | ✅ world_size/rank/group 封装 |
| broadcast (权重同步) | ❌ 无 | ✅ GroupCoordinator.broadcast |
| 多 group 管理 (PP/EP/CP) | ❌ 无 | ✅ 预留扩展点 |

### 工业问题

大模型 > 单 GPU 显存。TP 将权重矩阵沿 hidden dimension 切分到多 GPU，每 GPU 只存 1/tp_size 的权重。column-split（output 需要 all-gather）和 row-split（output 需要 all-reduce）是分布式推理的基础原语。C083 补充了 C050 缺失的进程组初始化和抽象层。

### 依赖

C050 的内联 torch.distributed 通信层。

### 验证

`torchrun --nproc_per_node=2 run.py` — C083 重构后用 GroupCoordinator 替代内联调用，输出与 C050 一致（回归测试）。

### AGENTS.md update

- `Package layout` section: `v1/distributed/` 注释更新为 `parallel_state.py (GroupCoordinator + initialize_model_parallel)`
- `Key architecture facts` section: "Tensor parallel" 行更新为 "TP=2 industrial: initialize_model_parallel(nccl) → GroupCoordinator(all_reduce/all_gather/broadcast), ColumnParallelLinear/RowParallelLinear 通过 GroupCoordinator 通信"

---

## C080 — MultiprocExecutor

**来源**: 真实 vLLM `v1/executor/multiproc_executor.py` (1089 lines, 教育版提取 ~150 lines)

### 目标

C032 只有 `UniProcExecutor`（单进程）。C084 (PP)、C086 (Multi-GPU)、C102 (KV Connector) 都需要多 Worker 进程管理。C080 在 C083 的 NCCL 基础设施之上，创建最小可行的 MultiprocExecutor。

### 文件

```
v1/executor/multiproc_executor.py  (新增)
```

### 核心实现

```python
# v1/executor/multiproc_executor.py
import multiprocessing as mp
from v1.executor.abstract import Executor

class MultiprocExecutor(Executor):
    """多进程 Executor: 每个 GPU 一个 Worker 进程。
    
    与 UniProcExecutor 的区别:
    - UniProcExecutor: 主进程直接调用 Worker.execute_model()
    - MultiprocExecutor: 通过 multiprocessing + shared memory 调度多个 Worker 进程
    """
    supports_pp: bool = True
    
    def __init__(self, vllm_config, monitor_workers=True):
        self.vllm_config = vllm_config
        self.world_size = vllm_config.tensor_parallel_size
        self.workers: list[WorkerProcHandle] = []
    
    def _init_executor(self):
        """为每个 GPU rank 启动 Worker 子进程"""
        for rank in range(self.world_size):
            worker = WorkerProc(
                vllm_config=self.vllm_config,
                local_rank=rank,
                rank=rank,
            )
            handle = worker.start()  # 启动子进程 + 等待 ready
            self.workers.append(handle)
    
    def execute_model(self, scheduler_output):
        """分发 SchedulerOutput 到所有 Worker，收集结果"""
        outputs = []
        for worker in self.workers:
            outputs.append(worker.recv_output())
        return outputs[0]
    
    def shutdown(self):
        for worker in self.workers:
            worker.terminate()

class WorkerProc:
    """管理一个 Worker 子进程的生命周期"""
    def __init__(self, vllm_config, local_rank, rank):
        self.process: mp.Process | None = None
    
    def start(self):
        self.process = mp.Process(target=self._run_worker)
        self.process.start()
        return self
    
    def _run_worker(self):
        """子进程入口: 初始化 CUDA → 加载模型 → 循环执行 execute_model"""
        torch.cuda.set_device(self.local_rank)
        worker = Worker(self.vllm_config, self.rank, self.world_size)
        worker.init_device()
        worker.load_model()
```

### 为什么放在 C083 之后

C083 提供 `initialize_model_parallel(nccl)`。MultiprocExecutor 需要多进程 + NCCL process group 组合——每个 Worker 子进程调用 `initialize_model_parallel` 加入同一个 group。

### 工业问题

单进程只能管理一张 GPU。多 GPU 需要多进程——python `multiprocessing` 是最小依赖方案。`MultiprocExecutor` 实现 `Executor` ABC——EngineCore 不需要知道后面是单进程、多进程还是 Ray actor。

### 依赖

C083 的 `init_process_group(nccl)` 和 `GroupCoordinator`。

### 验证

`python run.py` — MultiprocExecutor 替代 UniProcExecutor，单 worker 运行正常（world_size=1 兼容路径）。

### AGENTS.md update

- `Package layout` section: `executor/` 注释更新为 `ABC, UniProcExecutor, MultiprocExecutor`
- `Key architecture facts` section: 新增 "MultiprocExecutor: WorkerProcess per GPU, 替代 UniProcExecutor 支持多进程并行"

---

## C081 — MoE (Mixture of Experts)

**来源**: `Principle/C071-C100/C070-C100.md` #C081 节

### 目标

从零实现 MoE layer，理解 gating + expert dispatch + fused kernel 全流程。为 C082 EP 打基础。

### 文件

```
v1/model_executor/layers/fused_moe/
  __init__.py
  fused_moe.py       # fused_moe_kernel (Triton): parallel expert computation
  layer.py           # FusedMoE factory → MoERunner (对齐 vLLM)
  router/
    __init__.py
    fused_moe_router.py  # FusedMoERouter: TopK routing + load balancing loss

v1/model_executor/models/opt.py  # 修改: OPT decoder layer 中替换 MLP 为 MoE
```

### 核心实现

```python
# router/fused_moe_router.py
# 对齐 vLLM: model_executor/layers/fused_moe/router/fused_moe_router.py FusedMoERouter
class FusedMoERouter(nn.Module):
    def __init__(self, hidden_dim, num_experts=8, top_k=2):
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        router_logits = self.gate(x)         # [num_tokens, num_experts]
        routing_weights, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        return routing_weights, expert_indices

# layer.py
# 对齐 vLLM: model_executor/layers/fused_moe/layer.py FusedMoE() → MoERunner
def FusedMoE(hidden_dim, intermediate_dim, num_experts=8, top_k=2):
    """工厂函数: 创建 MoE layer (对齐 vLLM FusedMoE)"""
    router = FusedMoERouter(hidden_dim, num_experts, top_k)
    experts = nn.ModuleList([
        ExpertMLP(hidden_dim, intermediate_dim) for _ in range(num_experts)
    ])
    return MoERunner(router, experts, top_k)

class MoERunner(nn.Module):
    """执行 MoE 推理 (对齐 vLLM MoERunner)"""
    def __init__(self, router, experts, top_k):
        self.router = router
        self.experts = experts
        self.top_k = top_k

    def forward(self, x):
        routing_weights, expert_indices = self.router(x)
        output = fused_moe(x, self.experts, routing_weights, expert_indices, self.top_k)
        return output

# fused_moe.py
@triton.jit
def fused_moe_kernel(
    a_ptr, w1_ptr, w2_ptr, output_ptr,
    routing_map, routing_weights,
    num_experts, BLOCK_SIZE_M: tl.constexpr,
):
    """每个 Triton block 处理一个 expert 的 token batch。
    所有 expert 并行执行，无需 for 循环。"""
```

### 工业问题

参数量爆炸——LLaMA-7B → DeepSeek-V3 671B。MoE 通过条件计算（每个 token 只激活 top-k 个 expert）将每 token FLOPs 从 O(d²) 降到 O(d²/E × k)，同时维持总参数量。这是 DeepSeek-V3 / Mixtral / Qwen-MoE 的核心架构。

### 依赖

无（独立模块，为 C082 EP 打基础）。

### 验证

`python run.py` — `FusedMoE()` 返回的 MoERunner 替代标准 MLP 后模型可正常 forward，输出 shape 正确。

### AGENTS.md update

- `Package layout` section: `v1/model_executor/` 下新增 `layers/fused_moe/` 目录，注释 `FusedMoE (factory → MoERunner), FusedMoERouter, fused_moe_kernel`
- `Key architecture facts` section: 新增 "MoE: FusedMoERouter (TopK gating) + fused_moe_kernel (Triton parallel), auxiliary load balancing loss"

---

## C082 — EP (Expert Parallel)

**来源**: `Principle/C071-C100/C070-C100.md` #C082 节

### 目标

将 C081 的 MoE 扩展到多 GPU，实现 All-to-All expert dispatch/combine。

### 文件

```
v1/distributed/
  elastic_ep/
    __init__.py
    dispatch.py         # EPGroup, all_to_all_dispatch, all_to_all_combine

v1/model_executor/layers/fused_moe/layer.py  # 修改: 集成 EP dispatch/combine
v1/distributed/parallel_state.py             # 修改: 增加 EP group 管理
```

### 核心实现

```python
# v1/distributed/elastic_ep/dispatch.py
def expert_parallel_dispatch(
    hidden_states: torch.Tensor,    # [num_tokens, hidden_dim]
    router_indices: torch.Tensor,   # [num_tokens, top_k]
    router_weights: torch.Tensor,   # [num_tokens, top_k]
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 token dispatch 到对应 expert 所在的 GPU。
    
    通信模式:
    1. 每个 token 根据 router_indices 计算目标 GPU:
       target_gpu = expert_id // num_local_experts
    2. All-to-All: 每个 GPU 发送 tokens 到持有对应 expert 的 GPU
    3. 接收端: 聚合同一 expert 的所有 tokens
    """

def expert_parallel_combine(
    expert_output: torch.Tensor,    # expert computation results
    sorted_indices: torch.Tensor,   # for restoring original order
) -> torch.Tensor:
    """All-to-All 将 expert 输出送回原始 token 所在的 GPU。"""
```

### 工业问题

MoE 模型有 8-256 个 experts，每个 expert 完整 MLP 单 GPU 装不下。All-to-All 是 EP 的最大瓶颈：通信量与 `top_k × hidden_dim × num_tokens` 成正比，与 TP（高频小量 all-reduce）和 PP（低频大量 send/recv）的通信模式完全不同。

### 依赖

C081 的 MoE router + expert 实现。

### 验证

`python run.py` — EP=2 运行 MoE 模型，所有 tokens 正确路由到对应 expert GPU 并返回结果。

### AGENTS.md update

- `Package layout` section: `v1/distributed/` 注释更新提及 `elastic_ep/dispatch.py`
- `Key architecture facts` section: 新增 "EP: All-to-All dispatch (token→expert GPU), All-to-All combine (expert output→original GPU)"

---

## C084 — PP (Pipeline Parallel)

**来源**: `Principle/C071-C100/C070-C100.md` #C084 节

### 目标

将模型 layers 切分到多个 GPU，通过 micro-batch 填充 pipeline bubble。推理使用 1F1B-lite schedule。

### 文件

```
v1/worker/gpu/pp_utils.py                # 新增: PPHandler (对齐 vLLM v1/worker/gpu/pp_utils.py)
                                           #   sampled-token broadcast/recv on side stream

v1/executor/multiproc_executor.py        # 修改: 支持 PP layout
v1/worker/gpu_model_runner.py            # 修改: 仅在当前 stage 的 layers 上执行
v1/worker/gpu_worker.py                  # 修改: execute_model 处理 send/recv
v1/distributed/parallel_state.py         # 修改: get_pp_group() + 增加 PP group 管理
```

### 核心实现

```python
# v1/worker/gpu/pp_utils.py
# 对齐 vLLM: v1/worker/gpu/pp_utils.py PPHandler
@dataclass
class PendingRecv:
    """Per-step slot data for deferred postprocess (对齐 vLLM PendingRecv)"""
    event: torch.cuda.Event
    sampled_tokens: torch.Tensor       # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor          # [num_reqs]
    idx_mapping: torch.Tensor          # [num_reqs]

class PPHandler:
    """Pipeline parallelism sampled-token broadcast/recv on a side stream (对齐 vLLM PPHandler).
    
    非 last rank: receive sampled tokens from last rank
    Last rank: broadcast sampled tokens to all other ranks
    
    PP stage 间 activation 传递:
        stage_i → send/recv activation → stage_{i+1}
        通过 NCCL p2p send/recv (vllm.distributed.parallel_state)
    
    1F1B-lite schedule (概念):
        warmup: 依次发送 micro-batch 0,1,...,N-1
        steady: 每个 micro-batch 完成 → 发送下一个
        cooldown: 没有更多 micro-batch → 只处理剩余
    
    bubble_time = (pp_size - 1) * micro_batch_time
    efficiency = N / (N + pp_size - 1)
    """
    def receive(self, input_batch) -> bool:
        """非 last rank: recv sampled tokens from last rank"""
    def broadcast(self, sampled_token_ids, ...) -> None:
        """Last rank: broadcast sampled tokens to all other ranks"""
```

### 工业问题

TP 的通信量随层数线性增长。PP 按层切分，只在 stage 边界传 activation（send/recv），通信量大幅降低。但 PP 引入 pipeline bubble——第一个 micro-batch 走完所有 stage 之前，后面的 stage 等待。

### 依赖

C083 的 NCCL 通信基础设施 + C080 的 MultiprocExecutor（管理多个 PP stage 的 Worker 进程）。

### 验证

`python run.py` — PP=2 运行 OPT-350m，micro-batch 填充 pipeline bubble。

### AGENTS.md update

- `Key architecture facts` section: 新增 "PP (Pipeline Parallel): PPHandler (sampled-token side-stream broadcast), NCCL p2p send/recv for activation, 1F1B-lite schedule, bubble_time = (pp_size-1)×t_per_micro_batch"

---

## C085 — CP (Context Parallel) / Ring Attention

**来源**: `Principle/C071-C100/C070-C100.md` #C085 节

### 目标

将长序列的 Q/K/V 切分到多 GPU，通过 ring send/recv KV tiles 完成分布式 attention。

### 文件

```
v1/attention/ops/
  ring_attention.py     # 新增: Ring Attention kernel (Triton) + CPGroup

v1/distributed/parallel_state.py    # 修改: 增加 CP group 管理
```

### 核心实现

```python
# v1/attention/ops/ring_attention.py
class CPGroup:
    """Context Parallel group: 序列切分 + ring 通信。
    
    每个 GPU 持有: Q_local, K_local, V_local [batch, seqlen//cp_size, heads, head_dim]
    
    Ring 循环 (cp_size 步):
        step 0: 用本地 K,V 做 attention → partial O
        step 1: send K_local,V_local → next rank
                recv K_remote,V_remote → attention → accumulate to O
        ...
        step cp_size-1: 最后一步 → 完整 O
    """
    def __init__(self, cp_size, cp_rank):
        self.next_rank = (cp_rank + 1) % cp_size
        self.prev_rank = (cp_rank - 1) % cp_size

@triton.jit
def ring_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    Lse_ptr,          # log-sum-exp for online rescale
    seqlen_local: int, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """每个 ring step 对当前 K,V tile 做 FlashAttention 增量计算。
    通过 LSE 实现跨 GPU 的 online softmax 正确性。"""

def ring_attention_forward(Q, K, V, cp_group: CPGroup):
    """Ring Attention 前向。
    for step in range(cp_size):
        O_partial, Lse_partial = flash_attention_local(Q, K, V)
        O = online_softmax_merge(O, Lse, O_partial, Lse_partial)
        send K,V to next; recv from prev
    """
```

### 工业问题

序列长度从 4K 到 128K 到 1M tokens，attention O(n²) 成为瓶颈。TP/PP/EP 都不解决序列维度问题。CP 沿序列维度切分 Q/K/V，ring attention 通过 send/recv KV tiles 分布式计算 attention。核心难点是 online softmax 跨 GPU 的 LSE merge 正确性。

### 依赖

C083 的 NCCL 通信基础设施。

### 验证

`python run.py` — CP=2 运行长序列，与单 GPU 数值一致。

### AGENTS.md update

- `Key architecture facts` section: 新增 "CP: Ring Attention — Q/K/V split across GPUs, ring send/recv KV tiles, LSE online softmax merge across GPUs"

---

## C086 — Multi-GPU Integration

**来源**: `Principle/C071-C100/C070-C100.md` #C086 节

### 目标

TP × PP × EP × CP 组合验证，理解四种并行的拓扑布局和通信开销模型。

### 文件

```
v1/config/parallel.py              # 新增: ParallelConfig (tp/pp/ep/cp_size)

v1/executor/multiproc_executor.py  # 修改: TP×PP×EP×CP worker layout
v1/distributed/parallel_state.py   # 修改: 多 group 管理
```

### 核心理解（分析级，不做全配置覆盖）

**拓扑布局示例 (8 GPU)**:
```
TP=2, PP=2, EP=2:
  GPU 0,1: stage 0, expert_group 0 (TP=2)
  GPU 2,3: stage 1, expert_group 0 (TP=2)
  GPU 4,5: stage 0, expert_group 1 (TP=2)
  GPU 6,7: stage 1, expert_group 1 (TP=2)
```

**通信开销模型**:
```
TP: O(hidden_dim × seqlen) per layer          → 高频小量
PP: O(hidden_dim × seqlen) per stage          → 低频大量
EP: O(top_k × hidden_dim × seqlen) per MoE    → 偶发
CP: O(num_heads × head_dim × seqlen/cp_size)  → 批量中等
```

**兼容矩阵**:
```
TP × PP: ✅ (每个 PP stage 内部做 TP)
TP × EP: ✅ (每个 EP rank 内部做 TP)
TP × CP: ✅ (每个 CP rank 内部做 TP)
PP × CP: ⚠️ (需要跨 stage 同步 ring state)
EP × CP: ⚠️ (All-to-All 与 ring 通信重叠复杂)
```

### 工业问题

在实践中没有一种并行能单独解决所有问题。DeepSeek-V3 用 TP×EP×DP 3D 并行部署 671B 参数。理解四种并行的通信开销模型和兼容矩阵是设计分布式推理方案的前提。

### 依赖

C080 (MultiprocExecutor) + C082 (EP) + C084 (PP) + C085 (CP)。

### 验证

`python run.py` — TP=2 + PP=2 端到端运行 OPT-350m。

### AGENTS.md update

- `Key architecture facts` section: 新增 "Multi-GPU: TP×PP×EP×CP 拓扑布局 + 通信开销模型 + 兼容矩阵分析"

---

## Phase 2 完成后的能力

- ✅ TP=2 工业级 NCCL（init_process_group, GroupCoordinator, all_reduce/all_gather/broadcast）
- ✅ MultiprocExecutor: 多 Worker 进程管理
- ✅ MoE: TopK routing + fused Triton expert kernel
- ✅ EP: All-to-All expert dispatch/combine
- ✅ PP: 1F1B-lite pipeline parallel + bubble 分析
- ✅ CP: Ring Attention + LSE 跨 GPU merge
- ✅ Multi-GPU: TP×PP×EP×CP 组合拓扑 + 通信成本模型
- ✅ 面试天花板题：4 种并行策略的通信模式和兼容矩阵

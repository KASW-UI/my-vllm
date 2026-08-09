# Phase 3: 集群 + KV 传输

> 基线: C086 (分布式全景) | 目标: 集群部署能力 + KV Connector（PD 分离 MVP）
> 包含 4 个 P0 commits: C091, C092, C093, C102
>
> **真实 vLLM 路径映射**: KV Connector 在真实 vLLM 有两层——worker 侧 `v1/worker/gpu/kv_connector.py`，scheduler 侧 `distributed/kv_transfer/kv_connector/v1/base.py`（顶层）。教育项目统一放在 `v1/core/kv_connector.py`。

---

## 依赖拓扑

```
C086 (分布式全景)
  │
  ├─ C091  Ray 部署           ← 独立
  ├─ C092  K8s 部署           ← 独立
  └─ C093  Autoscaling         ← 需要 C048 (Metrics)
  │
C086 + 完整 KV 理解
  └─ C102  KV Connector        ← 跨进程 KV 传输，PD 分离基础
```

## 执行顺序

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 18 | C091: Cluster: Ray deployment | — | ray job |
| 19 | C092: Cluster: K8s deployment | — | `kubectl apply` |
| 20 | C093: Cluster: Autoscaling | C048 | 单元测试 |
| 21 | C102: KVCache: KV Connector — cross-process KV block transfer | C086 | `python run.py` |

---

## C091 — Ray 部署

**来源**: `Principle/C071-C100/C070-C100.md` #C091 节 + 真实 vLLM `v1/executor/ray_executor.py`

### 目标

理解 Ray actor 生命周期和 PlacementGroup——如何用 Ray 做推理引擎的分布式部署。

### 文件

```
v1/executor/ray_executor.py       # 新增: RayDistributedExecutor, RayWorkerWrapper (对齐 vLLM)
v1/executor/ray_utils.py          # 新增: initialize_ray_cluster, FutureWrapper (对齐 vLLM)
```

### 核心实现

```python
# v1/executor/ray_executor.py
# 对齐 vLLM v1/executor/ray_executor.py RayDistributedExecutor
import ray
from v1.executor.abstract import Executor

@ray.remote(num_gpus=1)
class RayWorkerWrapper:
    """Ray actor: 封装一个 Worker（GPU 进程）(对齐 vLLM RayWorkerWrapper).
    生命周期:
    - __init__: 加载模型到 GPU
    - execute_model: 接收 SchedulerOutput，返回 ModelRunnerOutput
    - shutdown: 释放 GPU 资源
    """
    def __init__(self, vllm_config, rank, world_size):
        self.worker = Worker(vllm_config, rank, world_size)
        self.worker.init_device()
        self.worker.load_model()

    def execute_model(self, scheduler_output):
        return self.worker.execute_model(scheduler_output)

class RayDistributedExecutor(Executor):
    """Ray 版 Executor (对齐 vLLM RayDistributedExecutor).
    Executor(ABC) → UniProcExecutor | MultiprocExecutor | RayDistributedExecutor
    """
    supports_pp: bool = True
    
    def __init__(self, vllm_config, placement_group=None):
        self.workers: list[ray.actor.ActorHandle] = [
            RayWorkerWrapper.remote(vllm_config, rank, world_size)
            for rank in range(world_size)
        ]

    def execute_model(self, scheduler_output):
        futures = [w.execute_model.remote(scheduler_output) for w in self.workers]
        return ray.get(futures)

# v1/executor/ray_utils.py
# 对齐 vLLM v1/executor/ray_utils.py
def initialize_ray_cluster(placement_strategy="STRICT_SPREAD"):
    """初始化 Ray 集群 (对齐 vLLM initialize_ray_cluster).
    
    PlacementGroup 策略:
    - STRICT_SPREAD: 每个 node 最多 1 个 actor → 跨 node 分布
    - STRICT_PACK: 所有 actor 在同一 node → 低延迟通信
    - PACK: 尽可能在同一 node，超出则 spill → 默认
    """
    ray.init()
    pg = ray.util.placement_group(...)
    return pg
```

### 核心理解点

- **Ray actor vs multiprocessing.Process**: Ray 提供跨 node、自动故障恢复、资源调度
- **PlacementGroup**: GPU 拓扑感知——控制 actor 在哪些 GPU/node 上运行
- **何时用 Ray 何时用 multiprocess**: 单 node 用 multiprocess（低延迟），多 node 用 Ray（自动调度）
- **Ray 与 Runtime 的解耦**: RayDistributedExecutor 实现 Executor ABC，EngineCore 不感知

### 工业问题

单 node 4-8 GPU 够跑 7B-70B 模型，但 405B+ 或高吞吐场景需要跨 node。Ray 提供跨 node 的 actor 生命周期管理、自动故障恢复、GPU topology-aware placement。RayDistributedExecutor 实现 Executor ABC——EngineCore 不需要知道后面是单进程、多进程还是 Ray actor。

### 验证

ray job 提交 — RayDistributedExecutor 替代 MultiprocExecutor 运行推理。

### AGENTS.md update

- `Package layout` section: `executor/` 注释更新提及 `ray_executor.py (RayDistributedExecutor, RayWorkerWrapper), ray_utils.py (initialize_ray_cluster)`
- `Key architecture facts` section: 新增 "RayDistributedExecutor: Ray actor per Worker, initialize_ray_cluster (STRICT_SPREAD/PACK/STRICT_PACK)"

---

## C092 — Kubernetes 部署

**来源**: `Principle/C071-C100/C070-C100.md` #C092 节

### 目标

理解 K8s 如何管理推理引擎的生命周期——deployment spec、HPA、readiness probe。

### 文件

```
deploy/k8s/
  deployment.yaml    # K8s Deployment spec
  service.yaml       # K8s Service spec
  hpa.yaml           # Horizontal Pod Autoscaler spec
```

### 核心实现

```yaml
# deploy/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-vllm-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: my-vllm
  template:
    spec:
      nodeSelector:
        accelerator: nvidia-a100    # GPU node affinity
      containers:
      - name: my-vllm
        image: my-vllm:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "opt-125m"
        - name: TP_SIZE
          value: "2"
        ports:
        - containerPort: 8000
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30   # 等待模型加载
          periodSeconds: 10

# deploy/k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: Pods
    pods:
      metric:
        name: request_queue_depth     # 自定义指标: 等待队列长度
      target:
        type: AverageValue
        averageValue: "5"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
    scaleDown:
      stabilizationWindowSeconds: 300 # 5min 才 scale down
```

### 核心理解点

- **GPU node affinity**: pod 必须调度到有 GPU 的 node（`nvidia.com/gpu: 1`）
- **Readiness probe**: 模型加载需要 30s-300s，pod 启动后不能立即接收流量
- **HPA 指标**: GPU utilization 对 LLM 不敏感（continuous batching 下天然接近 100%），用 `request_queue_depth` 或 `avg_ttft` 做扩缩容信号

### 工业问题

模型在企业环境部署后需要 K8s 管理生命周期。三个关键点：GPU node affinity、Readiness probe（模型加载延迟）、HPA 指标选择（queue_depth > GPU util）。

### 验证

`kubectl apply -f deploy/k8s/` — deployment/service/hpa 配置语义正确。

### AGENTS.md update

- `Package layout` section: 新增顶层 `deploy/k8s/` 目录
- `Key architecture facts` section: 新增 "K8s: Deployment (GPU nodeSelector), HPA (queue_depth metric), readiness probe"

---

## C093 — Autoscaling

**来源**: `Principle/C071-C100/C070-C100.md` #C093 节

### 目标

理解 autoscaling 的决策逻辑——何时 scale、如何避免冷启动、多模型调度。

### 文件

```
deploy/
  autoscaling.py   # AutoscalingController
```

### 核心实现

```python
# deploy/autoscaling.py
@dataclass
class ScalingDecision:
    action: str          # "scale_up" | "scale_down" | "noop"
    target_replicas: int
    reason: str

class AutoscalingController:
    """基于请求队列的自动扩缩容。

    决策逻辑:
    1. 监控指标:
       - request_queue_depth: 等待中的请求数
       - avg_ttft: 平均 Time To First Token
       - gpu_utilization: GPU 利用率（辅助参考）

    2. Scale up 条件 (OR):
       - queue_depth > threshold_high 持续 > 30s
       - avg_ttft > sla_target 持续 > 60s

    3. Scale down 条件 (AND):
       - queue_depth < threshold_low 持续 > 300s
       - all_gpu_utilization < target_util 持续 > 300s

    4. 冷启动处理:
       - 新 replica 需要加载模型 (~30s small, ~300s large)
       - scale_up 时增加容错余量 (over-provision)
       - 预热: 发送 dummy batch 触发 CUDA Graph capture

    5. 多模型场景:
       - 每个模型独立 autoscaling policy
       - GPU 总量约束: sum(model_i.replicas_i) <= total_gpus
       - 优先级 preemption: 高优模型可抢占低优模型的 GPU
    """
    def _should_scale_up(self, metrics) -> bool:
        """queue_depth > threshold_high 持续 > scale_up_window"""
    def _should_scale_down(self, metrics) -> bool:
        """queue_depth < threshold_low 持续 > scale_down_window"""
```

### 核心理解点

- **Autoscaling 是 Runtime 的"外层控制回路"**: Runtime 不变，外部根据指标调整 replica 数
- **LLM 特有的扩缩容挑战**: 模型加载时间长（冷启动）、GPU 量少且贵、TTFT SLA 敏感
- **为什么用 queue_depth 而不是 GPU util**: GPU 利用率在 LLM 推理中天然接近 100%，不是好的扩缩容信号

### 工业问题

LLM autoscaling 三个独特挑战：冷启动长达 30-300s（模型加载 + CUDA Graph capture）；GPU 数量有限且昂贵；TTFT SLA 敏感。关键设计：scale_up 窗口 30s（快速响应），scale_down 窗口 300s（防止频繁缩减浪费冷启动时间）；queue_depth 作为主指标。

### 依赖

C048 的 Metrics（TTFT/queue_depth 数据源）。

### 验证

单元测试 — 模拟 queue_depth 序列，验证 scale_up/scale_down 决策逻辑。

### AGENTS.md update

- `Package layout` section: `deploy/` 注释更新提及 `autoscaling.py`
- `Key architecture facts` section: 新增 "Autoscaling: queue_depth → scale_up/scale_down, cold_start_time 容错, 多模型 GPU 总量约束"

---

## C102 — KV Connector

**来源**: `Principle/C101-C120/C101-C120.md` #C102 节

### 目标

实现 KV cache 跨进程/跨机传输——这是 Prefill/Decode 分离（PD Disaggregation）的基础。允许 prefill worker 生成的 KV block 传输到 decode worker，实现 prefill 和 decode 在不同 GPU/机器上执行。

### 文件

```
v1/core/kv_connector.py           (新增)
v1/core/kv_cache_manager.py       (修改: 集成 connector)
v1/worker/gpu_model_runner.py     (修改: 支持接收外部 KV block)
v1/core/sched/scheduler.py        (修改: 支持跨 worker 调度)
```

### 核心实现

```python
# v1/core/kv_connector.py
# 对齐 vLLM 双层架构:
#   Worker 侧: v1/worker/gpu/kv_connector.py KVConnector / ActiveKVConnector
#   Scheduler 侧: distributed/kv_transfer/kv_connector/v1/base.py KVConnectorBase_V1
# 教育项目统一放在 v1/core/kv_connector.py
from enum import Enum

class KVTransferMethod(Enum):
    SHARED_MEMORY = "shared_memory"  # 同机多进程: shared memory
    NCCL = "nccl"                    # 同机多 GPU: NCCL send/recv
    RDMA = "rdma"                    # 跨机: RDMA (stub)

class KVConnectorBase_V1(ABC):
    """跨进程/跨机 KV block 传输 (对齐 vLLM KVConnectorBase_V1).
    
    使用场景 (PD Disaggregation):
      Prefill Worker (GPU A) → KVConnector.save_kv_layer(...)
        → Decode Worker (GPU B) → KVConnector.start_load_kv(...)
    
    传输粒度: KVCacheBlock (block_size tokens 的 K/V tensor)
    """
    
    # Worker-side API (对齐 vLLM KVConnector / ActiveKVConnector)
    def start_load_kv(self, forward_context, **kwargs) -> None:
        """开始从外部加载 KV blocks (对齐 vLLM start_load_kv)"""
    
    def wait_for_layer_load(self, layer_name: str) -> None:
        """等待指定 layer 的 KV 加载完成"""
    
    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        """保存一层的 KV cache (对齐 vLLM save_kv_layer)"""
    
    def wait_for_save(self):
        """等待所有 KV 保存完成"""
    
    # Scheduler-side API (对齐 vLLM KVConnectorBase_V1)
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        """计算可以从外部获取的 token 数"""
    
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        """分配 block 后更新状态"""
    
    def build_connector_meta(self, scheduler_output):
        """构建 scheduler→worker 的 metadata"""
    
    def request_finished(self, request, block_ids):
        """请求完成后清理"""
    
    # 磁盘操作 (对齐 vLLM offloading connector)
    def save_blocks(self, blocks: list[KVCacheBlock], path: str):
        """保存 KV blocks 到磁盘（用于 checkpoint/swap）"""
    
    def load_blocks(self, path: str) -> list[KVCacheBlock]:
        """从磁盘加载 KV blocks"""

# Scheduler 集成
class Scheduler:
    def schedule(self):
        """PD 分离调度:
        1. Prefill Worker: schedule new prompts → 生成 KV blocks
        2. KVConnector.wait_for_save() → KV 传输到 Decode Worker
        3. Decode Worker: start_load_kv() → 接收 blocks → 继续 decode
        """
        if self.is_prefill_worker:
            ...
        elif self.is_decode_worker:
            ...
```

### 为什么这是 PD 分离的基础

```
传统 vLLM (单 GPU):
  Prefill → KV in GPU memory → Decode
  (同一 GPU，KV 不需要移动)

PD Disaggregation:
  Prefill GPU:    Prompt → Prefill → KV blocks → save_kv_layer() + wait_for_save()
                                                               ↓
  Decode GPU:     start_load_kv() → KV blocks → Decode → tokens
  (不同 GPU/机器，KV blocks 通过网络传输)
```

### 依赖

C080 (MultiprocExecutor — 多 Worker 进程) + C083 (NCCL send/recv 基础设施) + C086 (完整的 KV Cache + 分布式通信理解)。

### 验证

`python run.py` — prefill 生成的 KV blocks 通过 shared memory/NCCL 传输到另一个 process，decode 继续正确生成 token。

### AGENTS.md update

- `Package layout` section: `core/` 注释更新提及 `kv_connector.py (KVConnectorBase_V1 — cross-process KV block transfer via start_load_kv/save_kv_layer)`
- `Key architecture facts` section: 新增 "KV Connector: KVConnectorBase_V1 (start_load_kv/save_kv_layer/wait_for_save), cross-process KV block transfer via shared memory/NCCL/RDMA, PD disaggregation foundation"

---

## Phase 3 完成后的能力

- ✅ Ray 部署: actor lifecycle + PlacementGroup
- ✅ K8s 部署: Deployment + HPA + readiness probe
- ✅ Autoscaling: queue_depth 决策循环 + 冷启动容错
- ✅ **KV Connector: 跨进程 KV 传输——PD 分离 MVP**
- ✅ 能从单机扩展到集群，能跨机传输 KV

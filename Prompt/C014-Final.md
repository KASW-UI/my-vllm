# C014 最终代码形态

## 一、文档约定

| 项目 | 值 |
|---|---|
| 项目地址 | `/home/kasw/文档/GitHub/GPU-Project/my-vllm` |
| 模型 | `facebook/opt-125m`（本地路径 `/home/kasw/文档/GitHub/cpu-vllm/opt-125m`） |
| 虚拟环境 | `/home/kasw/workspace/gpu-infra/.venv` |
| Python | 3.12 / torch 2.6.0+cu124 / transformers 4.49.0 |
| Git tag | `C014-production` (commit `8ec8f44`) |

---

## 二、实际文件树

```
my_vllm/
├── pyproject.toml
├── run.py                          # CLI 入口
├── .gitignore
│
├── examples/
│   ├── run.py                      # 离线推理
│   └── server.py                   # 启动 API 服务
│
├── tests/
│   ├── test_forward.py
│   ├── test_generate.py
│   ├── test_kv_cache.py
│   ├── test_request.py
│   ├── test_batch.py
│   ├── test_dynamic.py
│   ├── test_scheduler.py
│   ├── test_kv_manager.py
│   ├── test_model_runner.py
│   ├── test_tp.py
│   ├── test_llm_engine.py
│   ├── benchmark_c003.py
│   └── benchmark_c005.py
│
├── docs/snapshots/                 # C001..C014 阶段文档
│
└── my_vllm/
    ├── __init__.py
    │
    ├── runtime/                    # 基础运行时（C001-C003）
    │   ├── tokenizer.py            # HuggingFace AutoTokenizer 封装
    │   ├── model_loader.py         # HuggingFace AutoModelForCausalLM 加载
    │   ├── model.py                # LLM 类（forward + generate + generate_batch）
    │   └── main.py                 # CLI 入口逻辑
    │
    ├── generation/                 # 采样（C002）
    │   └── sampler.py              # greedy_sample()
    │
    ├── cache/                      # KV Cache 层（C003, C008, C009）
    │   ├── kv_cache.py             # KVCache 类：prefill() / decode()
    │   ├── block.py                # KVCacheBlock 数据类
    │   ├── block_pool.py           # BlockPool：固定大小 block 分配器
    │   ├── block_table.py          # BlockTable：per-request 逻辑→物理映射
    │   ├── allocator.py            # KVCacheAllocator（C008，已被 BlockPool 替代）
    │   └── kv_cache_manager.py     # KVCacheManager：集中 block 生命周期管理
    │
    ├── engine/                     # 引擎层（C004, C006, C014）
    │   ├── request.py              # Request + RequestStatus
    │   ├── waiting_queue.py        # WaitingQueue
    │   ├── running_queue.py        # RunningQueue
    │   ├── state_manager.py        # StateManager：核心调度-执行循环
    │   └── llm_engine.py           # LLMEngine：对外公 API
    │
    ├── scheduler/                  # 调度层（C007）
    │   └── scheduler.py            # Scheduler：FCFS + max_running + KV cache 感知
    │
    ├── worker/                     # 执行层（C010, C011）
    │   ├── worker_base.py          # WorkerBase：抽象接口
    │   ├── gpu_worker.py           # Worker：GPU 实现
    │   └── model_runner.py         # ModelRunner：forward + sampling 封装
    │
    ├── executor/                   # 执行器层（C012）
    │   └── uniproc_executor.py     # Executor 抽象 + UniProcExecutor
    │
    ├── distributed/                # 分布式（C013）
    │   └── parallel_state.py       # init_distributed / all_gather / all_reduce stub
    │
    ├── model_executor/             # 模型层（C013）
    │   └── layers/
    │       └── linear.py           # ColumnParallelLinear / RowParallelLinear
    │
    ├── server/                     # 服务层（C014）
    │   ├── api_server.py           # OpenAI 兼容 /v1/completions（stdlib）
    │   └── async_llm.py            # AsyncLLM：asyncio 异步包装
    │
    └── metrics/                    # 指标（C014）
        └── collector.py            # MetricsCollector

```

---

## 三、组件层次与职责边界

### 3.1 层次架构

```
┌────────────────────────────────────────────────────────────┐
│  server/api_server.py     HTTP server（OpenAI 兼容）        │  ← 外部接口
│  server/async_llm.py      Async 包装                       │
├────────────────────────────────────────────────────────────┤
│  engine/llm_engine.py     LLMEngine（公 API）               │  ← 外部 API
├────────────────────────────────────────────────────────────┤
│  engine/state_manager.py  StateManager（核心循环）           │  ← 编排层
│    ├─ scheduler/scheduler.py   调度决策（who）               │
│    ├─ executor/uniproc_executor.py 执行管理                 │
│    │    └─ worker/gpu_worker.py    GPU 执行环境             │
│    │         └─ worker/model_runner.py forward+sampling    │
│    └─ cache/kv_cache_manager.py  KV cache 生命周期          │
│         └─ cache/block_pool.py    Block 分配/回收           │
├────────────────────────────────────────────────────────────┤
│  distributed/parallel_state.py  TP 状态                     │  ← 分布式
│  model_executor/layers/linear.py  TP Linear 层             │
└────────────────────────────────────────────────────────────┘
```

### 3.2 关键约束

- **LLMEngine** 是唯一对外 API，外部不得直接接触 StateManager
- **StateManager** 是唯一协调者，外部不得绕过
- **Scheduler** 负责"谁运行"，不负责"怎么运行"
- **KVCacheManager** 是 Scheduler 与 BlockPool 之间唯一的中间层
- **Worker** 完全封装 ModelRunner（不暴露 `get_model_runner()` 给外部）
- **Executor** 统一管理 Worker 访问（当前为单 Worker）
- **Request** 不持有 block 引用（仅 `prompt_token_ids` + `output_token_ids` + `status`）

---

## 四、各组件逻辑详解

### 4.1 数据模型层

#### Request + RequestStatus (`engine/request.py`)

```python
class RequestStatus(IntEnum):
    WAITING  = 0
    RUNNING  = 1
    FINISHED_STOPPED       = 2
    FINISHED_LENGTH_CAPPED = 3

@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]  # 输入 token 序列
    output_token_ids: list[int]  # 累积生成的 token
    status: RequestStatus        # 当前状态
```

**注意**：Request 不持有 `block_table`、`block_ids`。Block 映射仅存在于 KVCacheManager 和后续的 CachedRequestState 中。`max_tokens` 通过 `getattr(req, "max_tokens", default)` 动态注入。

#### WaitingQueue (`engine/waiting_queue.py`)

```python
class WaitingQueue:
    push(request)    # 设置 status=WAITING，追加到队尾
    pop() → Request  # FCFS：弹出队首
    __len__ / __bool__
```

#### RunningQueue (`engine/running_queue.py`)

```python
class RunningQueue:
    add(request)      # 添加运行中请求
    remove(request)   # 移除请求
    __iter__          # 返回副本迭代（安全遍历 + 删除）
```

---

### 4.2 调度层

#### Scheduler (`scheduler/scheduler.py`)

```python
class Scheduler:
    # 核心状态
    waiting: list[Request]        # 等待队列（FCFS）
    running: list[Request]        # 运行队列
    max_running: int              # 最大并发数
    kv_cache_manager: KVCacheManager | None

    # 接口
    add_request(request)          # 入队 waiting
    schedule() → list[Request]    # FCFS 提升 waiting→running
    finish(request)               # 从 running 中移除
    has_work() → bool             # 是否有 work
```

**`schedule()` 逻辑**：
1. 遍历 waiting 队列（FCFS）
2. 检查 `max_running` 上限
3. 调用 `kv_cache_manager.allocate(request)` 预留 block
4. 若 allocate 失败（block 不足），将 request 放回 waiting，break
5. 设置 `status=RUNNING`，移入 running
6. 返回新提升的 request 列表

**关键行为**：`schedule()` 在提升前先通过 `kv_cache_manager.allocate()` 预留 block 资源。若 block 池已满，新的 request 不会提升，形成自然的 back-pressure。

**缺陷**：
- Scheduler 返回 `list[Request]`，非 vLLM 的 `SchedulerOutput` dataclass
- 无 token budget（`num_scheduled_tokens`）
- 无 `update_from_output()` 模式

---

### 4.3 KV Cache 层

#### KVCache (`cache/kv_cache.py`)

```python
class KVCache:
    _cached: tuple | None   # HF past_key_values

    prefill(model, input_ids) → logits
        # forward full prompt，use_cache=True
        # 存储 past_key_values

    decode(model, input_id) → logits
        # forward 单个 token，past_key_values 复用
        # 追加新 K,V 到 cache
```

**缺陷**：使用 HF 连续 tensor 而非物理 block 分页（注意力仍用 `past_key_values`，非 PagedAttention kernel）。

#### KVCacheBlock (`cache/block.py`)

```python
@dataclass
class KVCacheBlock:
    block_id: int
    is_free: bool = True
```

#### BlockPool (`cache/block_pool.py`)

```python
class BlockPool:
    blocks: list[KVCacheBlock]      # 全部 block
    _free_ids: list[int]            # 自由 block ID 队列
    block_size: int                 # 每 block 多少 token（默认 16）

    get_new_blocks(n) → list[KVCacheBlock] | None
        # 从 free_ids 弹 n 个，标记 !is_free
        # 不足返回 None

    free_blocks(blocks)
        # 标记 is_free，block_id 入 free_ids

    num_free / num_used: int
```

**消除碎片**：`free_blocks` 后 block_id 直接入自由队列，任意后续请求均可复用。

#### BlockTable (`cache/block_table.py`)

```python
class BlockTable:
    block_size: int
    blocks: list[KVCacheBlock]     # 逻辑位置 → 物理 block

    append(blocks)                  # 追加 block
    block_ids: list[int]            # 当前 block ID 列表
    num_tokens: int                 # = len(blocks) * block_size
```

#### KVCacheManager (`cache/kv_cache_manager.py`)

```python
class KVCacheManager:
    block_pool: BlockPool
    _caches: dict[str, KVCache]         # request_id → KVCache
    _block_tables: dict[str, BlockTable] # request_id → BlockTable

    allocate(request) → bool
        # ceil(prompt_tokens / block_size) 块
        # BlockPool.get_new_blocks() 分配
        # 创建 BlockTable + KVCache
        # 返回成功/失败

    get_cache(request_id) → KVCache | None
    get_block_ids(request_id) → list[int]
    free(request_id)
        # 归还 block 到 BlockPool，清理 cache 和 block_table
```

**`allocate()`** 被 Scheduler.schedule() 调用——在提升 request 之前预留 block。失败时 Scheduler 停止提升，形成 back-pressure。

---

### 4.4 执行层

#### WorkerBase (`worker/worker_base.py`)

```python
class WorkerBase(ABC):
    rank: int = 0
    world_size: int = 1

    @abstractmethod init_device()
    @abstractmethod execute_prefill(request_id, input_ids, cache) → int
    @abstractmethod execute_decode(request_id, cache) → int
    @abstractmethod get_step(request_id) → int
    @abstractmethod cleanup(request_id)
```

#### Worker (`worker/gpu_worker.py`)

```python
class Worker(WorkerBase):
    model: nn.Module
    device: torch.device
    _model_runner: ModelRunner | None

    init_device()
        # init_distributed(rank, world_size)
        # model.eval()
        # 创建 ModelRunner

    execute_prefill(request_id, input_ids, cache) → int
        # → ModelRunner.prefill()

    execute_decode(request_id, cache) → int
        # → ModelRunner.decode()
```

**关键**：Worker 不对外暴露 `get_model_runner()`——ModelRunner 是 Worker 内部实现细节。

#### ModelRunner (`worker/model_runner.py`)

```python
class ModelRunner:
    model: nn.Module
    device: torch.device
    eos_token_id: int
    _steps: dict[str, int]          # request_id → 已执行步数
    _next_token: dict[str, int]     # request_id → 下一个输入 token

    prefill(request_id, input_ids, cache) → int
        # cache.prefill(model, input_ids) → logits
        # greedy_sample(logits) → token
        # _steps[rid]=1, _next_token[rid]=token

    decode(request_id, cache) → int
        # _next_token[rid] → 构造输入 tensor
        # cache.decode(model, next_input) → logits
        # greedy_sample(logits) → next_token
        # _steps[rid] += 1, _next_token[rid]=next_token
```

**缺陷**：`_steps` 和 `_next_token` 用裸 dict 而非 vLLM 的 `CachedRequestState` dataclass。

---

### 4.5 执行器层

#### Executor + UniProcExecutor (`executor/uniproc_executor.py`)

```python
class Executor(ABC):
    @abstractmethod get_worker() → WorkerBase

class UniProcExecutor(Executor):
    worker: WorkerBase
    get_worker() → WorkerBase     # 返回持有的 Worker
```

**缺陷**：`get_worker()` 将 Worker 暴露给调用方（C012 遗留）。vLLM 的 Executor 提供 `execute_model(SchedulerOutput)` 而非直接暴露 Worker。

---

### 4.6 编排层（核心）

#### StateManager (`engine/state_manager.py`)

```python
class StateManager:    # 相当于 vLLM 的 EngineCore
    executor: Executor
    worker: WorkerBase                          # = executor.get_worker()
    kv_cache_manager: KVCacheManager
    scheduler: Scheduler
    max_new_tokens: int

    add_request(request)
        # → scheduler.add_request()

    step() → list[Request]
        # 1. scheduler.schedule() → new_requests
        # 2. 对每个 new req: _prefill(req)
        #    - kv_cache_manager.get_cache(rid) 获取已分配的 KVCache
        #    - worker.execute_prefill(rid, input_ids, cache) → first token
        #    - 若 EOS → finish
        #    - 否则 append 到 output_token_ids
        # 3. 对每个 running req: _decode_step(req)
        #    - worker.execute_decode(rid, cache) → next_token
        #    - 若 EOS → finish + free
        #    - 否则 append 到 output_token_ids
        # 4. 检查 finish：step >= max_tokens → FINISHED_LENGTH_CAPPED
        # 5. 返回 finished request 列表

    has_work() → bool
        # → scheduler.has_work()
```

**执行流程**：
```
step()
  ├─ scheduler.schedule()                          # 调度决策
  │     ├─ 遍历 waiting (FCFS)
  │     ├─ kv_cache_manager.allocate(req)          # 预留 block
  │     └─ return [newly_promoted]
  │
  ├─ for each new: _prefill(req)                   # Prefill 新请求
  │     ├─ kv_cache_manager.get_cache(rid)
  │     ├─ worker.execute_prefill(rid, ids, cache) # → ModelRunner.prefill()
  │     │     └─ cache.prefill(model, ids)         # → HF model.forward()
  │     │     └─ greedy_sample(logits)
  │     └─ append to output_token_ids
  │
  ├─ for each running: _decode_step(req)           # Decode 一步
  │     ├─ worker.execute_decode(rid, cache)       # → ModelRunner.decode()
  │     │     └─ cache.decode(model, token)        # → HF model.forward()
  │     │     └─ greedy_sample(logits)
  │     └─ append to output_token_ids
  │
  └─ 检查：step >= max_tokens? → FINISHED + free
```

**缺陷**：
- Per-request 逐个执行（非 batch `execute_model`）
- Finish 检查内联在 StateManager（非 `scheduler.update_from_output()`）
- 命名为 `StateManager` 而非 vLLM 的 `EngineCore`

---

### 4.7 对外 API 层

#### LLMEngine (`engine/llm_engine.py`)

```python
class LLMEngine:                     # 对外唯一公 API
    _sm: StateManager
    _tokenizer
    _pending: dict[str, Request]

    add_request(prompt, max_new_tokens?) → str   # 创建 Request → StateManager
    step() → list[dict]                           # 执行一步，返回完成的输出
    has_work() → bool
    generate(prompt, max_new_tokens) → dict       # 阻塞式完整生成
```

`LLMEngine` 是外部用户和 API server 访问系统的唯一入口。它封装了 StateManager 的调度循环，将 Request 对象翻译为可序列化的 dict 输出。

#### AsyncLLM (`server/async_llm.py`)

```python
class AsyncLLM:
    _engine: LLMEngine
    _queue: Queue[GenerationRequest]
    _running: bool

    async generate(prompt, max_new_tokens) → dict
        # 创建 Future，入队 GenerationRequest
        # await future → 返回结果

    async run_loop()
        # 主循环：
        # 1. _drain_queue() → engine.add_request()
        # 2. if has_work(): engine.step()
        # 3. 通知完成的 future
```

AsyncLLM 将同步的 LLMEngine 包装为异步接口。API server 当前使用同步 `LLMEngine.generate()`（因 stdlib HTTP server 非 async），AsyncLLM 为未来 async server 预留。

#### API Server (`server/api_server.py`)

```python
# 零外部依赖：stdlib http.server + json
POST /v1/completions  {"prompt": "...", "max_tokens": 50, "temperature": 0}
  → engine.generate(prompt, max_new_tokens)
  → {"choices": [{"text": "...", "finish_reason": "stop/length"}], "usage": {...}}

GET /health  → {"status": "ok"}
```

OpenAI 兼容响应格式：`id`, `object`, `model`, `choices[].text/finish_reason`, `usage.prompt_tokens/completion_tokens/total_tokens`。

---

### 4.8 辅助模块

#### sampler (`generation/sampler.py`)

```python
def greedy_sample(logits: torch.Tensor) → torch.Tensor:
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
```

仅支持 greedy（temperature=0）。API server 拒绝 temperature>0 的请求。

#### MetricsCollector (`metrics/collector.py`)

```python
@dataclass
class MetricsCollector:
    num_requests, num_finished, total_prompt_tokens, total_output_tokens
    avg_latency_ms: float
    throughput_tok_per_s: float

    on_request_start(request_id)
    on_request_finish(request_id, prompt_tokens, output_tokens)
    report() → dict
```

#### parallel_state (`distributed/parallel_state.py`)

```python
def init_distributed(world_size=1, rank=0)  # 设置全局状态
def get_tp_world_size() → int               # 返回 tp 数量
def all_gather / all_reduce → Tensor        # world_size=1 → identity
def split_tensor(tensor, dim) → Tensor      # world_size=1 → return tensor
```

world_size=1 时所有 collective ops 退化为 identity——接口存在，多卡扩展时激活。

#### Column/RowParallelLinear (`model_executor/layers/linear.py`)

```python
class ColumnParallelLinear(nn.Module):
    # weight: (output/tp_size, input)
    # forward: F.linear(x, weight, bias)
    #   tp_size>1: all_gather(local_out)

class RowParallelLinear(nn.Module):
    # weight: (output, input/tp_size)
    # forward: F.linear(x, weight)
    #   tp_size>1: all_reduce(local_out) + bias
```

world_size=1 时退化为标准 `nn.Linear`。

---

## 五、完整控制流（一次 /v1/completions 请求）

```
Client: POST /v1/completions {"prompt": "Hello world", "max_tokens": 10}
  │
  ├─ [1] api_server.CompletionHandler.do_POST()
  │       ├─ 解析 JSON body
  │       └─ engine.generate(prompt, max_new_tokens=10)
  │
  ├─ [2] LLMEngine.generate()
  │       ├─ add_request(prompt, 10)
  │       │     ├─ tokenizer.encode(prompt) → prompt_token_ids
  │       │     ├─ Request(id, prompt_token_ids, WAITING)
  │       │     └─ state_manager.add_request(req)
  │       │           └─ scheduler.add_request(req) → waiting.append
  │       │
  │       └─ while has_work(): step()
  │
  ├─ [3] StateManager.step()
  │       ├─ scheduler.schedule()
  │       │     ├─ waiting.pop(0) → request (FCFS)
  │       │     ├─ kv_cache_manager.allocate(request)
  │       │     │     ├─ ceil(3 tokens / 16) = 1 block
  │       │     │     ├─ block_pool.get_new_blocks(1) → [KVCacheBlock(0)]
  │       │     │     ├─ BlockTable(16) + append(blocks)
  │       │     │     └─ create KVCache()
  │       │     ├─ status=RUNNING, running.append(request)
  │       │     └─ return [request]
  │       │
  │       ├─ _prefill(request)
  │       │     ├─ kv_cache_manager.get_cache(rid) → KVCache
  │       │     ├─ worker.execute_prefill(rid, tensor([[2,15043,3186]]), cache)
  │       │     │     └─ ModelRunner.prefill(rid, ids, cache)
  │       │     │           ├─ cache.prefill(model, ids) → HF use_cache=True → past_key_values
  │       │     │           ├─ greedy_sample(logits) → token_id (e.g., 3186)
  │       │     │           └─ _steps[rid]=1, _next_token[rid]=3186
  │       │     └─ request.output_token_ids.append(3186)
  │       │
  │       ├─ _decode_step(request) × 9  (decode loop)
  │       │     └─ worker.execute_decode(rid, cache)
  │       │           └─ ModelRunner.decode(rid, cache)
  │       │                 ├─ _next_token[rid] → next_input tensor
  │       │                 ├─ cache.decode(model, next_input) → past_key_values
  │       │                 ├─ greedy_sample(logits) → next_token
  │       │                 └─ _steps[rid] += 1
  │       │
  │       └─ step(10) ≥ max_new_tokens(10)? → FINISHED_LENGTH_CAPPED
  │             ├─ kv_cache_manager.free(rid)
  │             │     └─ block_pool.free_blocks([KVCacheBlock(0)])
  │             └─ return [request]
  │
  ├─ [4] LLMEngine.step() → result dict
  │       ├─ tokenizer.decode(output_token_ids) → text
  │       └─ return {"text": "Hello world! I'm...", "finish_reason": "...", ...}
  │
  └─ [5] 200 OK → OpenAI JSON response
```

---

## 六、性能参数（C014 运行配置）

| 参数 | 值 |
|---|---|
| 模型 | `facebook/opt-125m` |
| dtype | `bfloat16` |
| max-model-len | `2048`（OPT 默认） |
| max-num-seqs | `8`（默认 `max_running`） |
| block-size | `16` |
| num-kv-blocks | `512`（默认） |
| tensor-parallel-size | `1` |
| pipeline-parallel-size | `1` |
| attention-backend | HF `past_key_values`（内置 SDPA） |
| sampling | greedy only（temperature=0） |

---

## 七、与 vLLM V1 的架构差距

| 差距 | C014 现状 | vLLM V1 |
|---|---|---|
| 编排组件名 | `StateManager` | `EngineCore` |
| Scheduler 返回 | `list[Request]` | `SchedulerOutput` dataclass |
| Executor 接口 | `get_worker()` → Worker | `execute_model(SchedulerOutput)` → ModelRunnerOutput |
| 执行模式 | per-request 循环 | batch forward |
| 输出处理 | StateManager 内联 check | `scheduler.update_from_output()` |
| 执行状态 | `_steps/_next_token: dict` | `CachedRequestState` dataclass |
| FlashAttention | 无（HF SDPA） | FlashAttention / FlashInfer backend |
| CUDA Graph | 无 | 有 |
| Prefix Caching | 无 | BlockPool block hash 查找 |
| Chunked Prefill | 无 | `allocate_slots(num_new_tokens)` |
| Sampler | `greedy_sample()` 函数 | `Sampler(nn.Module)` 完整 pipeline |
| API Server | stdlib `http.server` | FastAPI + uvicorn |

---

## 八、C014 定位声明

C014 achieves **architectural parity** with the core execution path of vLLM V1.

The implementation intentionally removes production complexity while preserving:

- EngineCore (StateManager) driven execution loop
- Scheduler → allocate → execute → cleanup flow
- Worker → ModelRunner execution boundary
- KVCacheManager → BlockPool memory abstraction with paged allocation
- Request lifecycle management (WAITING → RUNNING → FINISHED)
- OpenAI-compatible API server
- Async serving wrapper
- Tensor Parallel infrastructure (world_size=1 stub)

差距不是架构缺失，而是：工程规模、Kernel 优化、分布式能力、调度策略丰富度、生产可靠性。

---

## 九、vllm V1 文件树

```
vllm/v1/
├── engine/          (14 files)
│   ├── core.py              ← EngineCore — 核心调度循环
│   ├── core_client.py       ← EngineCore 客户端抽象
│   ├── llm_engine.py        ← LLMEngine — 外部入口
│   ├── async_llm.py         ← AsyncLLM — async 包装
│   ├── coordinator.py       ← 协调多进程 / DP
│   ├── detokenizer.py       ← token→text
│   ├── output_processor.py  ← 输出后处理
│   ├── input_processor.py   ← 输入预处理
│   ├── logprobs.py / parallel_sampling.py / tensor_ipc.py
│   └── exceptions.py / utils.py
│
├── core/            (7 + sched/)
│   ├── block_pool.py               ← KVCacheBlock 池
│   ├── kv_cache_manager.py         ← KV cache 分配/释放
│   ├── kv_cache_coordinator.py     ← 异构 KV cache 协调
│   ├── kv_cache_utils.py / kv_cache_metrics.py
│   ├── single_type_kv_cache_manager.py
│   ├── encoder_cache_manager.py
│   └── sched/
│       ├── scheduler.py            ← 核心调度器
│       ├── async_scheduler.py      ← async variant
│       ├── interface.py / output.py / request_queue.py / utils.py
│
├── worker/          (30+ files)
│   ├── worker_base.py              ← Worker 抽象
│   ├── gpu_worker.py / cpu_worker.py / xpu_worker.py
│   ├── gpu_model_runner.py         ← GPU ModelRunner
│   ├── cpu_model_runner.py
│   ├── block_table.py              ← BlockTable (per-request)
│   ├── gpu/
│   │   ├── model_runner.py         ← 实际 GPU 推理执行
│   │   ├── input_batch.py          ← 批输入构建
│   │   ├── block_table.py / attn_utils.py / states.py
│   │   ├── model_states/           ← 模型状态管理
│   │   ├── sample/                 ← GPU 端采样
│   │   └── spec_decode/            ← GPU 端推测解码
│   └── workspace.py / startup_plan.py / utils.py
│
├── executor/        (9 files)
│   ├── abstract.py          ← Executor 接口
│   ├── uniproc_executor.py  ← 单进程 Executor
│   ├── multiproc_executor.py← 多进程 Executor
│   └── ray_executor.py / ray_executor_v2.py / ray_utils.py
│
├── attention/       (30+ files)
│   ├── backend.py / selector.py
│   ├── backends/            ← 各种 attention 后端
│   │   ├── flash_attn.py / flashinfer.py / flex_attention.py
│   │   ├── triton_attn.py / mamba_attn.py / ...
│   │   └── mla/             ← MLA 专用 (DeepSeek)
│   └── ops/                 ← 手写 CUDA/Triton kernel 操作
│       ├── paged_attn.py / triton_decode_attention.py
│       ├── chunked_prefill_paged_decode.py
│       └── ...
│
├── sample/          (sampler + logits_processor + ops)
├── spec_decode/     (推测解码 proposer)
├── kv_offload/      (KV cache CPU offload)
├── structured_output/
├── metrics/
├── pool/            (embedding pooling)
├── request.py
├── outputs.py
├── kv_cache_interface.py
└── serial_utils.py / utils.py
2. vllm/model_executor 完整树 (45 dirs, 464 files)
model_executor/
├── layers/
│   ├── attention/          ← Attention 层实现 (MLA, cross, encoder, etc.)
│   ├── linear.py           ← 各类 Linear (含 TP Column/RowParallelLinear)
│   ├── fused_moe/          ← MoE 融合 kernel
│   ├── mamba/              ← Mamba mixer
│   ├── quantization/       ← 量化方案 (fp8, gptq, awq, marlin, ...)
│   ├── rotary_embedding/   ← 所有 RoPE 变体
│   ├── layernorm.py / activation.py / conv.py / mla.py / mhc.py
│   └── vocab_parallel_embedding.py
│
├── models/                 ← 150+ 模型适配
│   ├── llama.py / qwen2.py / deepseek_v2.py / mistral.py / ...
│   └── transformers/       ← 通用 Transformers 模型实现
│
├── kernels/                ← 手写 CUDA/Triton kernels
│   ├── linear/             ← 各类 quantized linear kernels
│   └── mhc/                ← Multi-Head Convolution kernels
│
├── model_loader/           ← 模型加载 (sharded, tensorizer, etc.)
├── offloader/              ← GPU-CPU offload
├── warmup/                 ← 预热 (JIT, CUDA graphs)
└── parameter.py / utils.py / custom_op.py
```

---

## 十、启动命令

```bash
# 离线推理
python run.py

# API 服务
python examples/server.py

# 测试
pytest tests/ --ignore=tests/benchmark_c003.py --ignore=tests/benchmark_c005.py

# API 调用
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello world","max_tokens":20}'

curl http://localhost:8000/health
```

# C014 最终代码形态（修正版）

## 一、文档约定

| 项目 | 值 |
|---|---|
| 项目地址 | `/home/kasw/文档/GitHub/GPU-Project/my-vllm` |
| 模型 | `opt-125m` |
| 虚拟环境 | `/home/kasw/workspace/gpu-infra/.venv` |

## 二、最终文件树

```
my_vllm/
├── pyproject.toml
├── my_vllm/
│   ├── __init__.py
│   ├── config.py                      # EngineConfig（model_name, dtype, max_model_len, ...）
│   ├── v1/
│   │   ├── __init__.py
│   │   ├── engine/
│   │   │   ├── __init__.py
│   │   │   ├── core.py                # EngineCore
│   │   │   ├── llm_engine.py          # LLMEngine
│   │   │   ├── async_llm.py           # AsyncLLM
│   │   │   ├── input_processor.py     # 原始 prompt → EngineCoreRequest
│   │   │   ├── output_processor.py    # EngineCoreOutputs → RequestOutput（含 detokenize）
│   │   │   └── detokenizer.py         # token IDs → text
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── scheduler.py           # Scheduler
│   │   │   ├── block_pool.py          # KVCacheBlock + BlockPool
│   │   │   ├── kv_cache_manager.py    # KVCacheManager
│   │   │   ├── request_queue.py       # RequestQueue
│   │   │   └── output.py              # SchedulerOutput, NewRequestData, CachedRequestData
│   │   ├── worker/
│   │   │   ├── __init__.py
│   │   │   ├── worker_base.py         # WorkerBase（抽象接口）
│   │   │   ├── gpu_worker.py          # Worker（GPU实现）
│   │   │   ├── gpu_model_runner.py    # GPUModelRunner
│   │   │   └── cache_state.py         # CachedRequestState
│   │   ├── executor/
│   │   │   ├── __init__.py
│   │   │   └── uniproc_executor.py    # UniProcExecutor
│   │   ├── sample/
│   │   │   ├── __init__.py
│   │   │   └── sampler.py             # Sampler
│   │   ├── attention/
│   │   │   ├── __init__.py
│   │   │   └── backends/
│   │   │       └── flash_attn.py      # FlashAttentionWrapper
│   │   └── request.py                 # Request + RequestStatus
│   ├── model_executor/
│   │   ├── __init__.py
│   │   ├── model_loader.py            # 从 HF 加载权重
│   │   └── models/
│   │       └── opt.py                 # OPTModel 适配
│   └── server/
│       ├── __init__.py
│       └── api_server.py              # OpenAI /v1/completions
├── examples/
│   ├── offline.py                     # 离线推理
│   └── server.py                      # 启动 API
├── tests/
└── docs/
    └── snapshots/
        ├── C001.md ... C014.md
```

## 三、组件层次与职责边界

```
LLMEngine          ← 对外唯一 API，add_request() / step()
  │
  ▼
EngineCore         ← 唯一协调者，外部不知道 Scheduler/Executor 存在
  │
  ├──▶ Scheduler   ← schedule() → SchedulerOutput
  │                    update_from_output() → EngineCoreOutputs
  │
  └──▶ Executor    ← execute_model(SchedulerOutput) → ModelRunnerOutput
          │
          └──▶ Worker  ← execute_model()
                  │
                  ├─ 1. SchedulerOutput → CachedRequestState（在 Worker 端缓存/更新 request 投影）
                  ├─ 2. CachedRequestState → 构造 GPU 输入张量
                  └─ 3. GPUModelRunner.execute_model()
                          ├─ model.forward()
                          └─ sampler()
```

**关键约束**：
- `LLMEngine` 不直接调用 Scheduler，只通过 `EngineCore.step()`
- `EngineCore` 对外暴露 `add_request()` 和 `step()` 两个函数
- `Request` 是纯逻辑对象，不持有 block 资源引用
- `KVCacheManager` 是 `Scheduler` 与 `BlockPool` 之间唯一的中间层
- `CachedRequestState` 是 `SchedulerOutput` 到 `ModelRunner` 之间的显式中间层

## 四、核心数据结构

### 4.1 Request (`v1/request.py`)

```python
class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_LENGTH_CAPPED = 3
    FINISHED_ABORTED = 4

@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]         # 逐渐累积，只读视图
    status: RequestStatus
    arrival_time: float
    max_tokens: int                     # 最大输出长度
    num_computed_tokens: int            # 已计算 token 数（prefill 后 = len(prompt)，每步 decode +1）
    block_hashes: list[bytes]           # 前缀缓存 hash（简化版，C014 可选）
    sampling_params: SamplingParams     # temperature, top_p, max_tokens 等
```

**注意**：`Request` 不持有 `block_table`、`block_ids`。Block 映射只存在于 `KVCacheManager` 和 `CachedRequestState` 中。

对照 vLLM：`vllm/v1/request.py:59` — Request 只有 `block_hashes`，没有 block_table。

### 4.2 SchedulerOutput (`v1/core/output.py`)

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams  # 首次调度时传递给 Worker 缓存

@dataclass
class CachedRequestData:
    # 已经缓存在 Worker 中的 request，只传差异
    req_ids: list[str]
    # req_id → 本轮新增的 block_ids（追加到已有 block_table）
    new_block_ids: dict[str, list[int]]
    # req_id → 本轮计算的 token 数
    num_computed_tokens: dict[str, int]
    # req_id → 需要附加的 block_ids 列表（完整，用于 prefill）
    block_ids: dict[str, list[int]]

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]          # 首次调度的请求
    scheduled_cached_reqs: CachedRequestData          # 已缓存过的请求
    num_scheduled_tokens: dict[str, int]              # req_id → 本轮调度 token 数
    total_num_scheduled_tokens: int                   # 本轮总 token 数
    finished_req_ids: set[str]                        # 本轮结束的 request
    # C014 简化：暂不含 spec_decode, encoder_inputs 等
```

对照 vLLM：`vllm/v1/core/sched/output.py:193-237`

### 4.3 CachedRequestState (`v1/worker/cache_state.py`) — 新增

```python
@dataclass
class CachedRequestState:
    """Request 在 Worker 进程中的缓存投影。
    
    首次调度时由 NewRequestData 创建，后续只更新差异部分。
    """
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    generator: torch.Generator | None

    block_ids: tuple[list[int], ...]     # 每层 KV cache 的 block_ids（简化版单个 list）
    num_computed_tokens: int             # 已计算 token 数
    output_token_ids: list[int]          # 已输出 token

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
```

对照 vLLM：`vllm/v1/worker/gpu_input_batch.py:34-90`

### 4.4 ModelRunnerOutput

```python
@dataclass
class ModelRunnerOutput:
    sampled_token_ids: dict[str, list[int]]    # req_id → 本轮采样 token
    logprobs: dict[str, list[float]] | None    # 可选
```

### 4.5 EngineCoreOutput

```python
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]
    finish_reason: str | None    # "stop" | "length" | None
```

## 五、KVCacheManager — Block 映射关系的唯一持有者

```python
class KVCacheManager:
    def __init__(self, block_pool: BlockPool, block_size: int, num_layers: int):
        self.block_pool = block_pool
        self.block_size = block_size
        # request_id → BlockTable（每个 request 持有的 block 列表）
        self.request_blocks: dict[str, list[KVCacheBlock]] = {}

    def get_blocks(self, request_id: str) -> list[KVCacheBlock]:
        return self.request_blocks.get(request_id, [])

    def allocate_slots(
        self, request: Request, num_new_tokens: int
    ) -> list[KVCacheBlock] | None:
        """为 request 分配 num_new_tokens 所需的 block。返回 None 表示 GPU 内存不足。"""
        ...

    def free(self, request_id: str) -> None:
        """释放 request 占用的所有 block。"""
        ...

    def get_block_ids(self, request_id: str) -> list[int]:
        """返回 block_id 列表（供 Worker 使用）。"""
        ...
```

**注意**：`Request` 不知道 block 的存在。`KVCacheManager` 是唯一持有 `request_id → blocks` 映射的组件。

## 六、核心控制流（EngineCore.step()）

```python
class EngineCore:
    def __init__(self, vllm_config):
        self.scheduler = Scheduler(vllm_config)
        self.executor = UniProcExecutor(vllm_config)

    def add_request(self, request: Request):
        self.scheduler.add_request(request)

    def step(self) -> dict[str, EngineCoreOutput]:
        # 1. 调度决策
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.total_num_scheduled_tokens == 0:
            return {}

        # 2. 模型执行（SchedulerOutput → Worker → GPU）
        model_output = self.executor.execute_model(scheduler_output)

        # 3. 输出处理（更新 request 状态、释放 KV cache）
        engine_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_outputs
```

### Scheduler.schedule() 内部逻辑

```
1. 遍历 self.running（已调度 request）：
    每个 request 分配 1 个 decode token
    → kv_cache_manager.allocate_slots(req, num_tokens=1)

2. 遍历 self.waiting（新 request）：
    如果 token budget 和 block 足够：
        kv_cache_manager.allocate_slots(req, num_prompt_tokens)
        从 waiting 移到 running

3. 组装 SchedulerOutput：
    scheduled_new_reqs   ← 首次调度的 request
    scheduled_cached_reqs ← 已缓存的 request（含新增 block_ids）
    num_scheduled_tokens
    finished_req_ids
```

### GPUModelRunner 内部流程

```
1. 获取/更新 CachedRequestState 缓存
   - scheduled_new_reqs → 创建新的 CachedRequestState
   - scheduled_cached_reqs → 更新已有 CachedRequestState 的 block_ids / output_token_ids

2. 从所有 CachedRequestState 构造输入张量：
   - input_ids: [batch_size, seq_len]    → 各 request 的本轮 token
   - positions: [batch_size]             → 各 token 在序列中的位置
   - block_tables: [batch, max_blocks]   → 各 request 的 block 映射
   - seq_lens: [batch_size]              → 各 request 序列长度

3. model.forward(input_ids, positions, block_tables)
   内部使用 FlashAttention + Paged KV Cache

4. 取最后一个 token 的 hidden_states → lm_head → logits

5. sampler(logits, sampling_params) → sampled_token_ids

6. 返回 ModelRunnerOutput
```

## 七、与 vLLM V1 源码的对照

| my-vllm 文件 | vLLM V1 文件 | 对齐程度 |
|---|---|---|
| `v1/engine/core.py` → `EngineCore` | `v1/engine/core.py:98` → `EngineCore` | 核心控制流对齐 |
| `v1/engine/llm_engine.py` → `LLMEngine` | `v1/engine/llm_engine.py:48` | 简化版（去掉 OutputProcessor 等） |
| `v1/core/scheduler.py` → `Scheduler` | `v1/core/sched/scheduler.py:69` | 简化版（FCFS only） |
| `v1/core/output.py` → `SchedulerOutput` | `v1/core/sched/output.py:193` | 字段精简但结构对齐 |
| `v1/core/block_pool.py` → `BlockPool` | `v1/core/block_pool.py:143` | 核心结构对齐 |
| `v1/core/kv_cache_manager.py` → `KVCacheManager` | `v1/core/kv_cache_manager.py:117` | 核心接口对齐 |
| `v1/worker/worker_base.py` → `WorkerBase` | `v1/worker/worker_base.py:39` | 抽象接口对齐 |
| `v1/worker/gpu_worker.py` → `Worker` | `v1/worker/gpu_worker.py` | 简化版 |
| `v1/worker/gpu_model_runner.py` → `GPUModelRunner` | `v1/worker/gpu_model_runner.py:453` | 核心流程对齐 |
| `v1/worker/cache_state.py` → `CachedRequestState` | `v1/worker/gpu_input_batch.py:34` | 核心字段对齐 |
| `v1/executor/uniproc_executor.py` → `UniProcExecutor` | `v1/executor/uniproc_executor.py` | 结构对齐 |
| `v1/sample/sampler.py` → `Sampler` | `v1/sample/sampler.py` | 简化版（greedy + top-p） |
| `v1/request.py` → `Request` | `v1/request.py:59` | 核心字段对齐（无 block_table） |
| `v1/attention/` | `v1/attention/selector.py` + `v1/attention/backends/` | auto 选择后端，flash_attn / torch fallback |
| `model_executor/models/opt.py` | `model_executor/models/` | 模型适配 |
| `v1/engine/async_llm.py` | `v1/engine/async_llm.py` | 简化 async 包装 |
| `server/api_server.py` | — | OpenAI 兼容 API |

## 八、C014 故意不包含的内容（留给 C015+）

| 不可实现 | 原因 |
|---|---|
| Multi-process EngineCore | 单进程足够展示核心控制流 |
| Speculative decoding | C014 后独立优化阶段 |
| Prefix caching | C014 后内存优化阶段 |
| Chunked prefill | C014 后调度增强阶段 |
| Structured output / grammar | 采样增强阶段 |
| CUDA Graph | 性能优化阶段 |
| Multi-modal (vision) | 远超简化范围 |
| PP / DP（流水线/数据并行） | 仅在 C013 实现 TP |
| P/D (disaggregation) | 架构过于复杂 |
| LoRA | 模型增强阶段 |
| KV cache quantization (FP8) | 内存优化阶段 |

## 九、性能参数（C014 最终运行）

| 参数 | 值 |
|---|---|
| 模型 | `facebook/opt-125m` |
| dtype | `bfloat16` |
| max-model-len | `4096` |
| max-num-batched-tokens | `4096` |
| max-num-seqs | `32` |
| block-size | `16` |
| tensor-parallel-size | `1` |
| pipeline-parallel-size | `1` |
| attention-backend | `auto` |
| attention-fallback | `torch_attention`（无 flash-attn 时自动降级） |
| enforce-eager | `True`（开发模式） |

---

## 十、定位声明

C014 achieves architectural parity with the core execution path of vLLM V1.

The implementation intentionally removes production complexity while preserving:

- EngineCore driven execution loop
- Scheduler → SchedulerOutput → Executor data flow
- Worker → ModelRunner execution boundary
- KVCacheManager → BlockPool memory abstraction
- Request lifecycle management

C014 完成后核心判断标准：回头看 C001 到 C014 的 Git history，看到的应该是一个工程团队"遇到真实问题→重构解决"的自然演进，而不是"按模块搭建"的教学项目。

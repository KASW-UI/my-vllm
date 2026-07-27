# C031 最终架构 — 对照 vLLM V1 与 C014 现状

## 一、C031 最终文件树

```
my_vllm/
├── __init__.py
├── config.py                          # EngineConfig  ← C014 不存在
│
├── v1/                                # ← C014 不存在整个 v1/ 命名空间
│   ├── __init__.py
│   ├── request.py                     # Request + RequestStatus (11种状态)
│   ├── outputs.py                     # ModelRunnerOutput, SamplerOutput  ← C014 不存在
│   ├── utils.py                       # ConstantList, tensor helpers     ← C014 不存在
│   ├── serial_utils.py                # MsgpackEncoder/Decoder, IPC 支持 ← C014 不存在
│   │
│   ├── engine/                        # ← C014 是 engine/ 但内容完全不同
│   │   ├── __init__.py
│   │   ├── core.py                    # EngineCore — 核心调度循环
│   │   ├── llm_engine.py              # LLMEngine — 对外 API
│   │   ├── async_llm.py               # AsyncLLM — 真正异步（非轮询包装）
│   │   ├── output_processor.py        # OutputProcessor + RequestState
│   │   ├── input_processor.py         # InputProcessor — prompt→EngineCoreRequest
│   │   ├── detokenizer.py             # IncrementalDetokenizer
│   │   └── exceptions.py              # EngineError
│   │
│   ├── core/                          # ← C014 是 cache/ + scheduler/ 分散
│   │   ├── __init__.py
│   │   ├── scheduler.py               # Scheduler — schedule()→SchedulerOutput
│   │   ├── block_pool.py              # KVCacheBlock + BlockPool + FreeKVCacheBlockQueue
│   │   ├── kv_cache_manager.py        # KVCacheManager + KVCacheBlocks
│   │   ├── kv_cache_utils.py          # KVCacheBlock, FreeKVCacheBlockQueue, BlockHashWithGroupId
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── output.py              # SchedulerOutput, NewRequestData, CachedRequestData
│   │       └── request_queue.py       # FCFSRequestQueue
│   │
│   ├── kv_cache_interface.py          # KVCacheSpec — Attention ↔ KVCacheManager 边界
│   │
│   ├── worker/                        # ← C014 有 worker/ 但接口完全不同
│   │   ├── __init__.py
│   │   ├── worker_base.py             # WorkerBase — execute_model(scheduler_output)
│   │   ├── gpu_worker.py              # Worker — GPU 实现
│   │   ├── gpu_model_runner.py        # GPUModelRunner — 批处理执行
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── input_batch.py         # InputBatch + InputBuffers
│   │       ├── block_table.py         # BlockTables — GPU tensor block table
│   │       ├── states.py              # RequestState — GPU 端请求状态缓存
│   │       └── model_states/
│   │           ├── interface.py       # ModelState 初始化接口
│   │           └── default.py         # 默认模型状态管理
│   │
│   ├── executor/                      # ← C014 有 executor/ 但缺少 execute_model()
│   │   ├── __init__.py
│   │   ├── abstract.py                # Executor (ABC) — execute_model(SchedulerOutput)
│   │   └── uniproc_executor.py        # UniProcExecutor
│   │
│   ├── attention/                     # ← C014 整个目录不存在
│   │   ├── __init__.py
│   │   ├── backend.py                 # AttentionBackend + AttentionImpl + CommonAttentionMetadata
│   │   ├── metadata.py                # AttentionMetadata (基类)
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── flash_attn.py          # FlashAttentionBackend
│   │       └── ops/
│   │           ├── __init__.py
│   │           └── paged_attn.py      # PagedAttention (PyTorch 实现)
│   │
│   ├── sample/                        # ← C014 是 generation/sampler.py (6行)
│   │   ├── __init__.py
│   │   └── sampler.py                 # Sampler (nn.Module) — greedy + top-k/top-p
│   │
│   └── metrics/                       # ← C014 有 metrics/ 但需重构
│       ├── __init__.py
│       └── collector.py
│
├── model_executor/                    # ← C014 存在但需增强
│   ├── __init__.py
│   ├── model_loader.py
│   └── models/
│       └── opt.py
│
├── server/                            # ← C014 存在但需重构（基于 FastAPI/uvicorn）
│   ├── __init__.py
│   └── api_server.py
│
├── examples/
│   ├── offline.py
│   └── server.py
└── tests/
```

**数量对照**

| 来源 | 文件数 |
|------|--------|
| C014 现状 | 33 |
| C031 目标（不含 tests/examples/docs） | ~40 |
| vLLM V1 `v1/` 目录 | ~220 |

C031 不追求文件数量对齐，追求**控制流 / 数据流 / 组件边界**对齐。

---

## 二、组件层次与职责边界

### C031 目标

```
User (API Server / AsyncLLM / LLMEngine)
  │ add_request() / generate()
  ▼
LLMEngine  ← 对外唯一 API，不直接调用 Scheduler
  │
  ├─ InputProcessor    — 原始 prompt → EngineCoreRequest
  │
  ▼
EngineCore  ← 唯一协调者，step() 循环
  │
  ├─▶ Scheduler       — schedule() → SchedulerOutput
  │   │                 update_from_output() → EngineCoreOutputs
  │   ├─ RequestQueue  — FCFS/Priority
  │   └─ KVCacheManager
  │       └─ BlockPool — block 分配/释放/prefix cache
  │
  └─▶ Executor        — execute_model(SchedulerOutput) → ModelRunnerOutput
        │
        └─▶ Worker    — execute_model() / sample_tokens()
              │
              ├─ InputBatch     — SchedulerOutput → GPU 输入张量
              ├─ BlockTables    — GPU tensor block table
              ├─ RequestState   — GPU 端请求投影
              │                   (all_token_ids, num_computed_tokens...)
              └─ GPUModelRunner — execute_model()
                    ├─ Model (HF)
                    ├─ AttentionBackend (FlashAttention)
                    │     ├─ PagedAttention
                    │     └─ KV Cache (物理 GPU memory)
                    └─ Sampler → 采样结果

OutputProcessor  ← EngineCoreOutputs → RequestOutput（含 Detokenizer）
  └─ Detokenizer    — token IDs → text
```

关键约束：
- `LLMEngine` 不直接调用 Scheduler，只通过 `EngineCore.step()`
- `EngineCore` 对外暴露 `add_request()` 和 `step()` 两个函数
- `Scheduler` 和 `Executor` 外部不可见
- `Request` 是纯逻辑对象，不持有 block 资源引用
- `KVCacheManager` 是 `Scheduler` 与 `BlockPool` 之间唯一的中间层
- `RequestState`（GPU 端）是 `SchedulerOutput` → `GPUModelRunner` 之间的显式投影

### C014 现状（对比）

```
LLMEngine.add_request()
  ↓
StateManager.step()        ← 耦合了调度+执行+状态更新
  │
  ├─ Scheduler.schedule()  → list[Request]（无结构化输出）
  ├─ for req: _prefill()   ← 逐 request 执行，无 batch
  ├─ for req: _decode_step()
  └─ state 更新 + cleanup   ← 全部在 StateManager 中

KVCache（HF past_key_values）← Python 对象，非物理 GPU memory
BlockPool（仅逻辑存在）     ← Attention 不使用
无 InputBatch / AttentionBackend / OutputProcessor
```

### 关键职责差异

| 职责 | vLLM V1 | C031 目标 | C014 现状 |
|------|---------|-----------|-----------|
| 调度 | `Scheduler.schedule()` → `SchedulerOutput` | 同左 | `Scheduler.schedule()` → `list[Request]` |
| 执行 | `Executor.execute_model(SchedulerOutput)` | 同左 | `Worker.execute_prefill(rid, cache)` 逐 request |
| 批处理 | `InputBatch` (GPU tensor) | 同左 | 无，逐 request for 循环 |
| KV Cache | BlockPool + GPU 物理内存 | 同左 | HF `past_key_values` (Python) |
| Attention | `AttentionBackend` 接口 | 同左 | 直接调用 HF model attention |
| 采样 | `Sampler(nn.Module)` | 同左 | `greedy_sample()` 单函数 |
| 输出处理 | `OutputProcessor` | 同左 | LLMEngine.step() 内联 |
| 输入处理 | `InputProcessor` | 同左 | LLMEngine.add_request() 内联 |
| 解 token | `IncrementalDetokenizer` | 同左 | `tokenizer.decode()` 全量 |
| 状态管理 | `RequestState` (GPU UVA tensor) | 同左 | Python `dict[str, int]` |

---

## 三、核心数据结构

### 1. Request — 请求对象

```python
# C031 目标（对齐 V1）
class RequestStatus(IntEnum):
    WAITING = 0
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = 1
    WAITING_FOR_REMOTE_KVS = 2
    WAITING_FOR_STREAMING_REQ = 3
    RUNNING = 4
    PREEMPTED = 5
    FINISHED_STOPPED = 6
    FINISHED_LENGTH_CAPPED = 7
    FINISHED_ABORTED = 8
    FINISHED_IGNORED = 9
    FINISHED_ERROR = 10
    FINISHED_REPETITION = 11

@dataclass
class Request:
    request_id: str
    status: RequestStatus
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    max_tokens: int
    prompt_token_ids: list[int] | None
    prompt_embeds: torch.Tensor | None
    num_prompt_tokens: int
    _output_token_ids: list[int]          # 内部写入
    _all_token_ids: list[int]             # prompt + output 合一
    num_computed_tokens: int              # KV 已计算 token 数
    spec_token_ids: list[int]             # draft token IDs
    num_preemptions: int
    block_hashes: list[bytes]             # prefix cache hash

    # 对外只读视图
    output_token_ids: ConstantList
    all_token_ids: ConstantList
    num_tokens: int
    num_tokens_with_spec: int
    num_output_tokens: int
    is_prefill_chunk: bool
    is_finished() -> bool
```

```python
# C014 现状
class RequestStatus(IntEnum):
    WAITING = 0
    RUNNING = 1
    FINISHED_STOPPED = 2
    FINISHED_LENGTH_CAPPED = 3

@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    status: RequestStatus
    # 缺失: max_tokens, num_computed_tokens, all_token_ids, block_hashes,
    #       spec_token_ids, sampling_params, arrival_time, num_preemptions
```

### 2. SchedulerOutput — 调度结果

```python
# C031 目标（对齐 V1）
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    block_ids: tuple[list[int], ...]           # 每个 kv_cache_group 的 block IDs
    num_computed_tokens: int                    # prefix cache 命中数

@dataclass
class CachedRequestData:
    req_ids: list[str]                          # 继续运行的请求
    new_block_ids: list[tuple[list[int], ...] | None]  # 本次新增 block
    num_computed_tokens: list[int]             # 已计算 token 数
    num_output_tokens: list[int]               # 已输出 token 数

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]        # 每类 token 数
    total_num_scheduled_tokens: int
    finished_req_ids: set[str]
    new_block_ids_to_zero: list[int] | None    # 需清零的 block
    preempted_req_ids: set[str] | None
```

```python
# C014 现状
# Scheduler 返回 list[Request]，无法表达:
# - prefill / decode
# - block metadata
# - finished requests
# - token budgets
```

### 3. ModelRunnerOutput — 模型执行结果

```python
# C031 目标（对齐 V1）
@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor          # [num_reqs] 或 [num_reqs, max_generated]
    logprobs_tensors: LogprobsTensors | None

@dataclass
class ModelRunnerOutput:
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]       # 每请求的采样结果
    logprobs: LogprobsLists | None
    pooler_output: list[torch.Tensor | None] | None
```

```python
# C014 现状
# execute_prefill/execute_decode 返回单个 int (token_id)
# 无结构化输出，调度侧无法获得 batch metadata
```

### 4. InputBatch — GPU 批输入

```python
# C031 目标（对齐 V1）
# 字段按 commit 分阶段引入:
#   C017 → req_ids, input_ids, positions, seq_lens
#   C019 → block_table_tensor, slot_mapping
#   C022 → query_start_loc, num_computed_tokens_np, is_prefilling_np, idx_mapping
@dataclass
class InputBatch:
    req_ids: list[str]
    num_reqs: int
    num_tokens: int
    idx_mapping: torch.Tensor             # batch_idx → req_state_idx (C022)
    input_ids: torch.Tensor               # [num_tokens] (C017)
    positions: torch.Tensor               # [num_tokens] (C017)
    query_start_loc: torch.Tensor         # [num_reqs + 1] — ragged batch 索引 (C022)
    seq_lens: torch.Tensor                # [num_reqs] (C017)
    num_computed_tokens_np: np.ndarray    # [num_reqs] (C022)
    is_prefilling_np: np.ndarray          # [num_reqs] — bool (C022)
    block_table_tensor: torch.Tensor      # [num_reqs, max_blocks] (C019)
    slot_mapping: torch.Tensor            # [num_tokens] (C019)
```

```python
# C014 现状
# 无此结构。每个 request 独立构造 (1, seq_len) tensor 传入 model.forward()
```

### 5. KVCacheBlock + BlockPool

```python
# C031 目标（对齐 V1）
@dataclass
class KVCacheBlock:
    block_id: int                          # 物理 block 索引
    ref_cnt: int = 0                       # 引用计数（CoW/prefix cache）
    block_hash: bytes | None = None        # prefix cache hash
    block_hash_num_tokens: int | None = None
    is_null: bool = False                  # null block (block_id=0)

class FreeKVCacheBlockQueue:
    """双向链表，按 LRU eviction 顺序管理空闲 block。"""
    free_list_head: KVCacheBlock | None
    free_list_tail: KVCacheBlock | None
    num_free_blocks: int
    # pop(), append(), remove(block), clear()

class BlockPool:
    blocks: list[KVCacheBlock]             # 全量 block 数组
    free_block_queue: FreeKVCacheBlockQueue
    cached_block_hash_to_block: dict[bytes, KVCacheBlock]
    # get_new_blocks(n) → list[KVCacheBlock]
    # free_blocks(blocks)
    # touch(blocks)                        # 刷新 eviction 位置
    # evict_blocks(block_ids)
    # cache_full_blocks(request, blocks, ...)
    # get_cached_block(block_hash) → KVCacheBlock | None
```

```python
# C014 现状
@dataclass
class KVCacheBlock:
    block_id: int
    is_free: bool = True                   # 仅有 bool，无 ref_cnt/hash

class BlockPool:
    blocks: list[KVCacheBlock]
    _free_ids: list[int]                   # 简单 list，无 LRU eviction
    # 方法: get_new_blocks(), free_blocks()
    # 缺失: ref_cnt(CoW), block_hash(prefix cache), touch(), evict_blocks(),
    #       eviction queue, get_cached_block()
```

### 6. KVCacheManager

```python
# C031 目标（对齐 V1）
@dataclass
class KVCacheBlocks:
    blocks: tuple[list[KVCacheBlock], ...]  # 按 kv_cache_group 分组
    def get_block_ids(allow_none=False) -> tuple[list[int], ...] | None

class KVCacheManager:
    block_pool: BlockPool
    req_to_blocks: dict[str, KVCacheBlocks]      # 每个请求的 block 列表
    # allocate_slots(req, num_new_tokens, num_new_computed_tokens=0, ...)
    #   → KVCacheBlocks | None
    #   按 block 分配，含 prefix cache 复用
    # get_computed_blocks(req) → (KVCacheBlocks, num_new_computed_tokens, ...)
    # free(req)
    # get_blocks(req_id) → KVCacheBlocks
    # get_block_ids(req_id) → tuple[list[int], ...]
    # take_new_block_ids() → list[int]      # 需清零的 block IDs
```

```python
# C014 现状
class KVCacheManager:
    block_pool: BlockPool
    _caches: dict[str, KVCache]             # ← KVCache = HF past_key_values 包装
    _block_tables: dict[str, BlockTable]
    # allocate(req) → bool — 仅检查能否分配 block 数
    # 缺失: allocate_slots(shaped), get_computed_blocks(prefix cache),
    #       ref_cnt(CoW), take_new_block_ids(清零), KVCacheBlocks
```

### 7. RequestState（GPU 端）

```python
# C031 目标（对齐 V1 worker/gpu/states.py）
class RequestState:
    req_id_to_index: dict[str, int]         # 请求 → GPU slot index
    index_to_req_id: dict[int, str]
    free_indices: list[int]                 # 可用 slot
    all_token_ids: torch.Tensor             # [max_num_reqs, max_model_len] — UVA
    prompt_len: torch.Tensor                # [max_num_reqs]
    num_computed_tokens: torch.Tensor       # [max_num_reqs]
    num_computed_tokens_np: np.ndarray      # CPU mirror（optimistic upper bound）
    last_sampled_tokens: torch.Tensor       # [max_num_reqs, 1]
    # add_request(req_id, prompt_len, ...) → int index
    # remove_request(req_id) → int | None
    # apply_staged_writes()
```

```python
# C014 现状
# 无此结构。ModelRunner 用 Python dict 维护 per-request 状态:
#   self._steps: dict[str, int]
#   self._next_token: dict[str, int]
```

### 8. BlockTables（GPU 端）

```python
# C031 目标（对齐 V1 worker/gpu/block_table.py）
class BlockTables:
    block_sizes: list[int]
    max_num_reqs: int
    num_kv_cache_groups: int
    block_tables: list[torch.Tensor]        # 每 group: [max_num_reqs, max_num_blocks]
    num_blocks: torch.Tensor                # [num_groups, max_num_reqs]
    slot_mappings: torch.Tensor             # [num_tokens]
    # append_block_ids(req_index, new_block_ids)
    # apply_staged_writes()
    # gather_block_tables(idx_mapping, num_reqs_padded) → (block_table0, block_table1, ...)
    # compute_slot_mappings(query_start_loc, positions, num_tokens)
```

```python
# C014 现状
class BlockTable:
    block_size: int
    blocks: list[KVCacheBlock]              # Python list，非 GPU tensor
    # 方法: append(), block_ids (property)
    # 缺失: GPU tensor 化、slot_mappings、gather_block_tables()
```

---

## 四、核心数据流

### C031 目标（对齐 V1）

```
┌──────────────────────────────────────────────────────────────────┐
│ EngineCore.step()                                                │
│                                                                  │
│  1. scheduler_output = scheduler.schedule()                      │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │ Scheduler.schedule():                                    │ │
│     │   ├─ 遍历 running requests，计算 num_new_tokens            │ │
│     │   ├─ kv_cache_manager.allocate_slots() — 含 prefix cache │ │
│     │   ├─ 遍历 waiting queue (RequestQueue)，尝试分配 block     │ │
│     │   ├─ 分配失败 → preempt 低优先级 running request            │ │
│     │   └─ 构建 SchedulerOutput:                                │ │
│     │       ├─ scheduled_new_reqs: list[NewRequestData]        │ │
│     │       ├─ scheduled_cached_reqs: CachedRequestData        │ │
│     │       ├─ finished_req_ids, new_block_ids_to_zero         │ │
│     │       └─ num_scheduled_tokens: {prefill: N, decode: M}   │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  2. model_output = executor.execute_model(scheduler_output)      │
│     ┌──────────────────────────────────────────────────────────┐ │
│     │ Worker.execute_model(scheduler_output):                  │ │
│     │   ├─ GPUModelRunner._prepare_inputs(scheduler_output):   │ │
│     │   │   ├─ RequestState.add_request() — 新请求入 slot        │ │
│     │   │   ├─ BlockTables.append_block_ids() — 更新 block table│ │
│     │   │   ├─ 构造 InputBatch (input_ids, positions,           │ │
│     │   │   │   query_start_loc, seq_lens, slot_mapping...)    │ │
│     │   │   └─ 构造 CommonAttentionMetadata                     │ │
│     │   ├─ Model forward:                                      │ │
│     │   │   └─ 每层: AttentionImpl.forward()                    │ │
│     │   │       ├─ PagedAttention(query, key, value,           │ │
│     │   │       │   block_table, kv_cache) → output            │ │
│     │   │       └─ reshape_and_cache(K, V, kv_cache,           │ │
│     │   │           slot_mapping)                              │ │
│     │   └─ Sampler.forward(logits) → SamplerOutput             │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. engine_core_outputs =                                        │
│      scheduler.update_from_output(scheduler_output, model_output)│
│     ┌──────────────────────────────────────────────────────────┐ │
│     │ ├─ 提取 sampled_token_ids, logprobs                       │ │
│     │ ├─ 更新每个 request: num_computed_tokens += num_new_tokens │ │
│     │ ├─ 检查 stop 条件 (EOS / max_tokens)                       │ │
│     │ ├─ 释放 finished request 的 KV blocks (KVCacheManager.free)│ │
│     │ └─ 构建 dict[int, EngineCoreOutputs]                      │ │
│     └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  4. return (engine_core_outputs, has_more_work)                  │
└──────────────────────────────────────────────────────────────────┘

输出侧（LLMEngine.step() 调用）:
  5. output_processor.process_outputs(engine_core_outputs)
     ├─ 每 request: detokenizer.update(token_ids)
     ├─ RequestState.make_request_output() → RequestOutput
     └─ return OutputProcessorOutput(request_outputs, reqs_to_abort)
```

### C014 现状（对比）

```
StateManager.step():
  1. new_requests = scheduler.schedule()       → list[Request]
     （Scheduler 只做 FCFS 弹出 + block 预分配检查）

  2. for req in new_requests: _prefill(req)
     ├─ 独立构造 input_ids tensor [1, seq_len]
     ├─ worker.execute_prefill(rid, input_ids, cache)
     │   └─ KVCache.prefill(model, input_ids)  → 调用 HF model(**HF past_kv)
     └─ greedy_sample(logits) → int

  3. for req in running: _decode_step(req)
     ├─ worker.execute_decode(rid, cache)
     │   └─ KVCache.decode(model, input_id)    → 调用 HF model(**HF past_kv)
     └─ 检查 EOS / max_tokens

  4. 返回 finished list[Request]
     （LLMEngine.step() 再做 tokenizer.decode）

缺失链：
  ✗ 无 SchedulerOutput → 无 batch metadata 传递
  ✗ 无 InputBatch → 无 GPU batch 构造
  ✗ 无 BlockTables (GPU tensor) → Attention 不知道 block layout
  ✗ 无 AttentionBackend → 直接调 HF model，无法插入 PagedAttention
  ✗ 无 RequestState (GPU) → Python dict 维护状态
  ✗ 无 OutputProcessor → 输出处理耦合在 LLMEngine
  ✗ 无 InputProcessor → 输入处理耦合在 LLMEngine
  ✗ 无 Detokenizer → 全量 tokenizer.decode()
  ✗ KVCache 是 HF past_key_values → 非物理 GPU memory，Scheduler 无法真正管理
```

### 物理 KV Cache 数据流对比

```
C014 现状:
  model.forward(input_ids, past_key_values=python_tuple)
  → 每层 K,V 是 Python list of tensors
  → 无 block 管理，无 prefix cache 可能

C031 目标（对齐 V1）:
  BlockPool.allocate() → KVCacheBlock[]
  KVCacheManager.allocate_slots() → KVCacheBlocks
  GPU: [num_blocks, num_kv_heads, block_size, head_dim] 预分配物理 tensor
  BlockTables: [num_reqs, max_blocks] GPU tensor
  Slot mappings: [num_tokens] — 每个 token 映射到
                  (block_id * block_size + offset)
  Attention: query @ KV cache[block_table[i]] → output
             reshape_and_cache(K, V, kv_cache, slot_mapping)
```

---

## 五、与现有代码逐文件对比

| C031 文件 | C014 对应 | 差距 |
|-----------|-----------|------|
| `config.py` | **不存在** | 需新增：model_name, dtype, block_size, max_model_len, max_num_seqs, max_num_batched_tokens, gpu_memory_utilization, enforce_eager ... |
| `v1/request.py` | `engine/request.py` | 少 7 种状态（WAITING_FOR_\*, PREEMPTED, FINISHED_ABORTED/ERROR/REPETITION），缺 `num_computed_tokens`, `block_hashes`, `all_token_ids`, `spec_token_ids`, `sampling_params` |
| `v1/outputs.py` | **不存在** | 需新增 ModelRunnerOutput, SamplerOutput, LogprobsLists |
| `v1/utils.py` | **不存在** | 需新增 ConstantList（只读 list 包装） |
| `v1/engine/core.py` | `engine/state_manager.py` | StateManager 耦合调度+执行，需拆为 EngineCore（纯协调 loop） |
| `v1/engine/llm_engine.py` | `engine/llm_engine.py` | 需增加 InputProcessor/OutputProcessor 依赖，移除 `generate()` 同步方法 |
| `v1/engine/async_llm.py` | `server/async_llm.py` | C014 是轮询包装（thread-safe queue + sleep），需改真正异步（asyncio Queue + background output_handler） |
| `v1/engine/output_processor.py` | **不存在** | 需新增：EngineCoreOutputs → RequestOutput 的转换链，含 RequestOutputCollector |
| `v1/engine/input_processor.py` | **不存在** | 需新增：prompt → tokenize → EngineCoreRequest |
| `v1/engine/detokenizer.py` | **不存在** | 需新增 IncrementalDetokenizer（增量解码，支持 stop string 检测） |
| `v1/engine/exceptions.py` | **不存在** | 需新增 EngineError 异常类 |
| `v1/kv_cache_interface.py` | **不存在** | 需新增 KVCacheSpec — Attention ↔ KVCacheManager 架构边界（C018） |
| `v1/core/kv_cache_utils.py` | `cache/block.py` | 合并：KVCacheBlock（含 ref_cnt/block_hash）+ FreeKVCacheBlockQueue（C018） |
| `v1/core/scheduler.py` | `scheduler/scheduler.py` | 返回 `list[Request]` → 改为返回 `SchedulerOutput`；增加 preemption、prefix cache lookup |
| `v1/core/sched/output.py` | **不存在** | 需新增 NewRequestData, CachedRequestData, SchedulerOutput |
| `v1/core/sched/request_queue.py` | `engine/waiting_queue.py` + `engine/running_queue.py` | C014 用 Python list，需改为 RequestQueue 抽象 + FCFSRequestQueue |
| `v1/core/block_pool.py` | `cache/block_pool.py` | 缺 ref_cnt（CoW）、block_hash（prefix cache）、FreeKVCacheBlockQueue（LRU eviction）、touch()、get_cached_block()、evict_blocks() |
| `v1/core/kv_cache_manager.py` | `cache/kv_cache_manager.py` | 缺 allocate_slots()（按 token 增量分配）、get_computed_blocks()（prefix cache）、KVCacheBlocks、take_new_block_ids() |
| `v1/worker/worker_base.py` | `worker/worker_base.py` | C014 接口是 `execute_prefill(rid, cache)→int`，需改为 `execute_model(SchedulerOutput)→ModelRunnerOutput` |
| `v1/worker/gpu_worker.py` | `worker/gpu_worker.py` | 同上，增加 `sample_tokens()`, `initialize_from_config()` |
| `v1/worker/gpu_model_runner.py` | `worker/model_runner.py` | 逐 request → 批量执行；Python dict → GPU tensor 状态 |
| `v1/worker/gpu/input_batch.py` | **不存在** | 需新增：SchedulerOutput → GPU 张量构建 |
| `v1/worker/gpu/block_table.py` | `cache/block_table.py` | Python list → GPU tensor ([max_reqs, max_blocks])，增加 slot_mappings 计算 |
| `v1/worker/gpu/states.py` | **不存在** | 需新增 RequestState（UVA tensor 状态管理） |
| `v1/worker/gpu/model_states/interface.py` | **不存在** | 需新增 ModelState 初始化接口（C021） |
| `v1/worker/gpu/model_states/default.py` | **不存在** | 需新增默认模型状态管理（C021） |
| `v1/serial_utils.py` | **不存在** | 需新增 MsgpackEncoder/Decoder, IPC 支持（C031） |
| `v1/executor/abstract.py` | `executor/uniproc_executor.py` | C014 Executor 只有 `get_worker()`，需改为 `execute_model(SchedulerOutput)→ModelRunnerOutput` |
| `v1/executor/uniproc_executor.py` | `executor/uniproc_executor.py` | 同上接口变更 |
| `v1/attention/backend.py` | **不存在** | 需新增 AttentionBackend + AttentionImpl + CommonAttentionMetadata |
| `v1/attention/metadata.py` | **不存在** | 需新增 AttentionMetadata 基类 |
| `v1/attention/backends/flash_attn.py` | **不存在** | 需新增 FlashAttentionBackend（简化版：PyTorch SDPA + PagedAttention） |
| `v1/attention/backends/ops/paged_attn.py` | **不存在** | 需新增 PagedAttention PyTorch 实现 |
| `v1/sample/sampler.py` | `generation/sampler.py` | 6 行函数 → `Sampler(nn.Module)` + logits processor + penalties pipeline |
| `v1/metrics/collector.py` | `metrics/collector.py` | 内部重构，接口保留 |

### 将移除的文件

| C014 文件 | 原因 |
|-----------|------|
| `engine/state_manager.py` | 职责拆分到 EngineCore + Scheduler + Executor |
| `engine/waiting_queue.py` | 合并到 `core/sched/request_queue.py` |
| `engine/running_queue.py` | 合并到 `core/sched/request_queue.py` |
| `scheduler/scheduler.py` | 迁移到 `v1/core/scheduler.py` |
| `worker/model_runner.py` | 迁移到 `v1/worker/gpu_model_runner.py` |
| `cache/allocator.py` | C008 遗留的连续内存分配器，已被 BlockPool 替代 |
| `cache/block_table.py` | 迁移到 `v1/worker/gpu/block_table.py`（GPU tensor 版） |
| `cache/kv_cache.py` | C003 遗留的 HF past_key_values 包装，需物理 KV 替代 |
| `cache/kv_cache_manager.py` | 迁移到 `v1/core/kv_cache_manager.py` |
| `cache/block_pool.py` | 迁移到 `v1/core/block_pool.py` |
| `cache/block.py` | 合并到 `v1/core/kv_cache_utils.py` + `v1/core/block_pool.py` |
| `batch/batch_executor.py` | C005 遗留的静态 batch（padding），被 InputBatch 替代 |
| `generation/sampler.py` | 迁移到 `v1/sample/sampler.py` |
| `runtime/`（全部 5 文件） | C001-C002 遗留，功能已被后续 commit 覆盖 |
| `distributed/parallel_state.py` | 简化保留为独立工具或内联到 gpu_worker |

---

## 六、架构差距总结

| 维度 | C014 现状 | C031 目标 | 核心变化 |
|------|-----------|-----------|----------|
| **命名空间** | 扁平 `my_vllm/{engine,cache,scheduler,worker,...}` | `my_vllm/v1/{engine,core,worker,executor,...}` | 引入 `v1/` 层，与 vLLM 目录结构对齐 |
| **控制流** | StateManager 耦合调度+执行 | EngineCore.step() 纯协调循环 | Scheduler ↔ Executor 解耦 |
| **数据流** | Python dict/list 传递 | 结构化 dataclass（SchedulerOutput, ModelRunnerOutput） | 类型安全 + batch metadata |
| **批执行** | 逐 request for 循环 | InputBatch 构建后一次 forward | GPU batch 利用率 |
| **KV Cache** | HF past_key_values（Python 对象） | GPU 物理 tensor + BlockPool | 物理管理 + prefix cache 可能 |
| **Attention** | 直接调 HF model attention | AttentionBackend 接口 + PagedAttention | 可替换 backend |
| **状态管理** | Python dict（ModelRunner._steps） | RequestState（GPU UVA tensor） | GPU 原生状态，支持 CUDA Graph |
| **输出处理** | LLMEngine 内联 | OutputProcessor + Detokenizer | 职责分离，支持 streaming |
| **输入处理** | LLMEngine 内联 | InputProcessor | 职责分离，支持 multimodal |
| **调度协议** | `list[Request]` | `SchedulerOutput` | 结构化：prefill/decode/block/finished 独立通道 |
| **并发** | 轮询 AsyncLLM（queue.Queue + sleep） | 真正异步（asyncio Queue + background task） | 非阻塞 |

---

## 七、C031 故意保留的差距

以下属于工程规模与优化层面，不属于架构缺失，留给 C031+ 或独立优化阶段：

| 能力 | 状态 |
|------|------|
| FlashAttention / FlashInfer kernel | 仅保留 Backend 接口，PagedAttention 用 PyTorch 实现 |
| CUDA Graph | 开发模式（`enforce_eager=True`），无 Graph Capture/Replaying |
| 真正多 GPU（TP>1, DP, PP） | Stub，仅保留 Executor 抽象 |
| Speculative Decoding | 未加入 |
| Structured Output | 未加入 |
| KV Offload | 未加入 |
| MoE / Quantization | 未加入 |
| Prefix Caching | C029 加入简化版（block hash lookup） |
| Chunked Prefill | C030 加入简化版 |
| 优先调度 | FCFS 为主，Priority Queue 保留接口 |
| Prometheus Metrics | 保持 basic collector |

---

## 八、C031 完成后判断标准

1. **控制流**：`EngineCore.step() → Scheduler.schedule() → SchedulerOutput → Executor.execute_model() → ModelRunnerOutput → scheduler.update_from_output() → output_processor.process_outputs()`
2. **batch 执行**：ModelRunner 一次 `execute_model()` 处理所有 scheduled requests
3. **物理 KV Cache**：GPU tensor 预分配，Attention 通过 slot_mapping + block_table 读写
4. **接口协议**：所有组件间通信通过结构化 dataclass
5. **可替换性**：AttentionBackend / Sampler / Executor 通过抽象接口可替换
6. **可运行**：每个 C015-C031 commit 后 `pytest` 全绿

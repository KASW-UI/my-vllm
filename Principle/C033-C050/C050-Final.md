# C050 最终架构 — C033-C050 阶段目标

> C050 每个类名、方法名、dataclass 字段名必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。
> C050 是单 GPU 工业化 Runtime 的阶段性终点，核心架构完成之后还有 C051-C070（单 GPU 高级能力）、C071-C100（分布式 + 生态 + SDK）。

## 一、C050 最终文件树

```
my_vllm/
├── __init__.py
├── config.py                    # EngineConfig (C032) — 保持简化，C069 拆分
│
├── forward_context.py            # BatchDescriptor, set/get_forward_context()  ← C041
│
├── v1/
│   ├── __init__.py
│   ├── request.py               # Request + RequestStatus (12 状态) + SamplingParams
│   ├── outputs.py               # SamplerOutput, ModelRunnerOutput, LogprobsLists, LogprobsTensors
│   ├── utils.py                 # ConstantList
│   ├── serial_utils.py          # MsgpackEncoder, MsgpackDecoder                  ← C047
│   ├── kv_cache_interface.py    # KVCacheSpec, FullAttentionSpec,                  ← C033
│   │                            # SlidingWindowSpec, MLAAttentionSpec,
│   │                            # KVCacheGroupSpec, KVCacheConfig
│   ├── cudagraph_dispatcher.py  # CudagraphDispatcher                              ← C044
│   │
│   ├── engine/
│   │   ├── __init__.py          # EngineCoreRequest, EngineCoreOutputs, FinishReason, PauseMode
│   │   ├── core.py              # EngineCore — step(), add_request(), shutdown(), sleep(), wake_up()
│   │   ├── core_client.py       # EngineCoreClient(ABC), InprocClient, SyncMPClient ← C045
│   │   ├── llm_engine.py        # LLMEngine — add_request(), step(), get_metrics()
│   │   ├── async_llm.py         # AsyncLLM — generate(), abort()                  ← C047
│   │   ├── output_processor.py  # OutputProcessor, RequestOutputCollector, RequestState ← C047
│   │   ├── input_processor.py   # InputProcessor.process_inputs()
│   │   ├── detokenizer.py       # Fast/SlowIncrementalDetokenizer, check_stop_strings() ← C047
│   │   ├── exceptions.py        # EngineGenerateError, EngineDeadError
│   │   └── utils.py             # EngineZmqAddresses                              ← C046
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── kv_cache_coordinator.py   # KVCacheCoordinator, HybridKVCacheCoordinator ← C034
│   │   ├── block_pool.py             # BlockPool + BlockHashToBlockMap              ← C035
│   │   ├── kv_cache_manager.py       # KVCacheManager, KVCacheBlocks
│   │   ├── kv_cache_utils.py         # KVCacheBlock(slots=True), FreeKVCacheBlockQueue,
│   │   │                              # KVCacheBlockCopy, BlockHash, BlockHashWithGroupId
│   │   ├── prefix_cache.py           # CoW                                              ← C036
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── interface.py          # SchedulerInterface (ABC, 20 方法)                ← C037
│   │       ├── scheduler.py          # Scheduler — schedule(), update_from_output()     ← C038
│   │       ├── output.py             # SchedulerOutput, NewRequestData, CachedRequestData
│   │       └── request_queue.py      # FCFSRequestQueue
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── worker_base.py            # WorkerBase (ABC)
│   │   ├── gpu_worker.py             # Worker — execute_model(), sample_tokens()
│   │   ├── gpu_model_runner.py       # GPUModelRunner — capture_model(), initialize_kv_cache()
│   │   ├── gpu_input_batch.py        # CachedRequestState, InputBatch, InputBuffers    ← C039
│   │   ├── block_table.py            # BlockTables — Triton slot_mappings               ← C040
│   │   ├── utils.py                  # AttentionGroup, bind_kv_cache()
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── states.py             # RequestState — add_request(), apply_staged_writes()
│   │       ├── attn_utils.py         # init_attn_backend(), init_kv_cache(),              ← C044-A
│   │       │                         # build_attn_metadata(), build_slot_mappings_by_layer()
│   │       └── model_states/
│   │           ├── __init__.py
│   │           ├── interface.py      # ModelState (ABC)
│   │           └── default.py        # DefaultModelState
│   │
│   ├── executor/
│   │   ├── __init__.py
│   │   ├── abstract.py               # Executor (ABC)
│   │   ├── uniproc_executor.py       # UniProcExecutor
│   │   └── multiproc_executor.py     # MultiprocExecutor                               ← C046
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── backend.py                # AttentionType, AttentionBackend/Impl, MetadataBuilder
│   │   ├── selector.py               # get_attn_backend()
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py           # AttentionBackendEnum                           ← C042
│   │   │   ├── torch_attn.py         # my-vllm 特有: vLLM V1 无此后端, 仅 TORCH_SDPA 用于 ViT
│   │   │   ├── flash_attn.py         # FlashAttentionBackend + FlashAttentionImpl      ← C043-A
│   │   │   │                         #   + FlashAttentionMetadataBuilder
│   │   │   └── triton_attn.py        # TritonAttentionBackend                          ← C043-B
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── paged_attn.py
│   │       ├── triton_reshape_and_cache_flash.py              ← C043-C
│   │       ├── triton_decode_attention.py                   ← C043-C
│   │       ├── triton_prefill_attention.py                  ← C043-C
│   │       ├── triton_unified_attention.py                  ← C043-B (custom op dispatch)
│   │       ├── chunked_prefill_paged_decode.py              ← C043-C
│   │       └── flash_attention_tiled.py                     ← C043 (教育性 tiled attention kernel)
│   │
│   ├── sample/
│   │   ├── __init__.py
│   │   ├── sampler.py                # Sampler.forward(), greedy_sample(), compute_logprobs()
│   │   ├── metadata.py               # SamplingMetadata
│   │   └── logits_processor.py       # penalties
│   │
│   └── metrics/
│       ├── __init__.py
│       ├── collector.py              # C032 已有
│       ├── stats.py                  # SchedulerStats, IterationStats                 ← C048
│       ├── loggers.py                # PrometheusStatLogger                           ← C048
│       └── prometheus.py             # setup_multiprocess_prometheus()                 ← C048
│
├── model_executor/
│   ├── __init__.py
│   ├── model_loader/               # load_model() (C032 内联)
│   ├── models/opt.py
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                 # ColumnParallelLinear, RowParallelLinear
│       └── attention/
│           ├── __init__.py
│           └── attention.py          # Attention(nn.Module)                             ← C041
│                                     #   forward()
│
├── distributed/
│   ├── __init__.py
│   ├── parallel_state.py             # TP/PP groups, all_reduce/all_gather/broadcast
│   └── tensor_parallel.py            # TP=2                                           ← C050
│
├── entrypoints/                      #                                            ← C049
│   └── openai/
│       ├── api_server.py             # /v1/chat/completions, /v1/completions
│       └── protocol.py
│
├── runtime/                          # Legacy (test compat)
│   ├── __init__.py
│   ├── model_loader/, tokenizer.py, model.py
│
├── server/                           # C014 legacy, deprecated
├── examples/
└── tests/
```

**数量对照**

| 来源 | my_vllm/ .py 文件数 |
|------|---------------------|
| C032 基线 | ~51 |
| C050 目标 | ~80 |
| vLLM V1 `v1/` 目录 | ~220 |

C050 达到 vLLM 的 ~36% 文件覆盖率，聚焦**核心执行链工业化**。注意：MLA 通过 `v1/attention/backends/mla/` 目录管理，不是通过 `SingleTypeKVCacheManager` 子类；embedding/pooling 通过 `GPUModelRunner` pooling path 处理，不是独立 runner。

> **C050 是阶段终点，不是项目终点。** 其后有 C051-C070、C071-C100。

---

## 二、组件层次与职责边界（对齐 vLLM 类名/方法名）

```
User / OpenAI API (entrypoints/openai/)
  │
  ├─ EngineClient (InprocClient)                                   ← C045
  │
  ▼
LLMEngine  ← 对外 API
  │
  ├─ InputProcessor.process_inputs()  — prompt → EngineCoreRequest
  │
  ▼
EngineCore  ← step() / step_with_batch_queue() 循环
  │
  ├─▶ Scheduler (via SchedulerInterface)                           ← C037
  │   │   schedule(throttle_prefills) → SchedulerOutput
  │   │   update_from_output(scheduler_output, model_runner_output)
  │   │     → dict[int, EngineCoreOutputs]
  │   │
  │   ├─ RequestQueue (FCFSRequestQueue)
  │   └─ KVCacheCoordinator                                        ← C034
  │       ├─ find_longest_cache_hit(block_hashes, max_cache_hit_length)
  │       └─ SingleTypeKVCacheManager
              │   ├─ FullAttentionManager
              │   ├─ SlidingWindowManager
              │   └─ [MLA 通过 v1/attention/backends/mla/ 管理, 非 SingleTypeManager]
  │               └─ BlockPool
  │                   ├─ get_new_blocks() / free_blocks() / touch()
  │                   ├─ _maybe_evict_cached_block()                 ← C035-C036
  │                   ├─ BlockHashToBlockMap                       ← C035
  │                   └─ free_blocks() (ref_cnt cascading) / emit_cached_block_events()
  │
  └─▶ Executor (UniProc / Multiproc)                              ← C046
        └─▶ Worker
              │ execute_model(scheduler_output) → ModelRunnerOutput
              │ sample_tokens(grammar_output) → ModelRunnerOutput
              │
              ├─ RequestState.add_request(req_id, prompt_len, all_token_ids,
              │       num_computed_tokens, max_tokens)
              ├─ BlockTables.compute_slot_mappings(idx_mapping,
              │       query_start_loc, positions, num_tokens_padded)
              ├─ InputBatch + CachedRequestState
              │
              ├─ build_attn_metadata() + build_slot_mappings_by_layer()
              ├─ set_forward_context(attn_metadata, vllm_config, …)
              │
              ├─ CudagraphDispatcher.dispatch(num_tokens, uniform_decode)
              ├─ Model.forward()
              │   └─ Attention.forward(hidden):
              │       ├─ ctx = get_forward_context()
              │       ├─ Q, K, V = self.qkv_proj(hidden)
              │       └─ unified_attention_with_output(Q, K, V, output, layer_name)
              │           └─ self.impl.forward(Q, K, V, kv_cache, ctx.attn_metadata)
              │               └─ FlashAttention / Triton PagedAttention kernel
              │
              └─ Sampler.forward(logits, sampling_metadata)
                    ├─ apply_penalties() / greedy_sample()
                    └─ compute_logprobs() → SamplerOutput
```

输出侧:
```
OutputProcessor.process_outputs(engine_core_outputs)
  ├─ RequestOutputCollector                                 ← C047
  ├─ RequestState.make_request_output()
  │   ├─ IncrementalDetokenizer.update(new_token_ids, stop_terminated)
  │   │   └─ check_stop_strings()                           ← C047
  │   └─ → CompletionOutput → RequestOutput
```

---

## 三、核心数据结构（字段名/方法签名对齐 vLLM）

### 1. KVCacheSpec 体系

```python
# vLLM 字段名
class KVCacheSpec:                    # vLLM: 仅 block_size
    block_size: int

class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int                    # vLLM 用 head_size

class FullAttentionSpec(AttentionSpec):
    head_size_v: int
    sliding_window: int | None

class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str

class KVCacheGroupSpec:
    layer_names: list[str]            # vLLM 用 layer_names
    kv_cache_spec: KVCacheSpec        # vLLM 用 kv_cache_spec

class KVCacheConfig:
    num_blocks: int
    kv_cache_groups: list[KVCacheGroupSpec]
```

### 2. KVCacheBlock

```python
@dataclass(slots=True)                # vLLM slots=True
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None    # vLLM BlockHashWithGroupId
    _block_hash_num_tokens: int | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False

class KVCacheBlockCopy(NamedTuple):   # C035 新增, vLLM NamedTuple
    src_block_id: int
    dst_block_id: int
```

### 3. KVCacheCoordinator

```python
class KVCacheCoordinator:
    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) → tuple[tuple[list[KVCacheBlock], ...], int, int]: ...

    def allocate_new_blocks(self, request, num_new_tokens) → KVCacheBlocks | None: ...
    def free(self, request_id): ...
```

### 4. SchedulerInterface

```python
class SchedulerInterface(ABC):
    def schedule(self, throttle_prefills: bool = False) → SchedulerOutput: ...
    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_runner_output: ModelRunnerOutput) → dict[int, EngineCoreOutputs]: ...
    def add_request(self, request: Request) → None: ...
    def finish_requests(self, request_ids, finished_status) → list[Request]: ...
    def get_num_unfinished_requests(self) → int: ...
    def has_unfinished_requests(self) → bool: ...
    def has_finished_requests(self) → bool: ...
    def reset_prefix_cache(self, reset_running_requests=False,
                           reset_connector=False) → bool: ...
    def make_stats(self) → SchedulerStats | None: ...
    def shutdown(self) → None: ...
```

### 5. SchedulerOutput

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]         # {"prefill": N, "decode": M}
    total_num_scheduled_tokens: int
    num_common_prefix_blocks: list[int]           # C034 新增
    finished_req_ids: set[str]
    preempted_req_ids: set[str] | None            # C038
    new_block_ids_to_zero: list[int] | None        # C035
    kv_cache_block_copies: list[KVCacheBlockCopy] | None  # C035
```

### 6. ForwardContext

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0

@contextmanager
def set_forward_context(attn_metadata, vllm_config, num_tokens=None,
                        batch_descriptor=None, slot_mapping=None): ...

def get_forward_context() → ForwardContext: ...
```

### 7. CachedRequestState + InputBatch

```python
@dataclass
class CachedRequestState:              # vLLM 在 worker/gpu_input_batch.py
    req_id: str
    prompt_token_ids: list[int] | None
    block_ids: tuple[list[int], ...] = ()
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    lora_request: object | None = None
    sampling_params: object | None = None
    generator: torch.Generator | None = None

@dataclass
class InputBatch:
    req_ids: list[str]
    num_reqs: int
    idx_mapping: torch.Tensor
    num_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_computed_tokens_np: np.ndarray
    is_prefilling_np: np.ndarray
    input_ids: torch.Tensor
    positions: torch.Tensor
    logits_indices: torch.Tensor
```

### 8. SamplingMetadata

```python
@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    generators: dict[int, torch.Generator]
    max_num_logprobs: int | None
    no_penalties: bool
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    output_token_ids: list[list[int]]
```

---

### 9. Attention 层（`model_executor/layers/attention/attention.py`）  ← C041
```python
# C050 目标（对齐 vLLM）

class Attention(nn.Module):
    """替代 HF attention 层，与 V1 attention backends 集成。"""

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        output_shape=None, output_dtype=None,
    ) → torch.Tensor:
        """
        1. If !self.attn_backend.forward_includes_kv_cache_update:
             unified_kv_cache_update(key, value, self.layer_name)
        2. unified_attention_with_output(query, key, value, output, self.layer_name)
             → 内部调用 self.impl.forward(Q, K, V, kv_cache, ctx.attn_metadata)
        """


# standalone functions in attention.py（vLLM）
def unified_kv_cache_update(key, value, layer_name) → None:
    """Calls attn_layer.impl.do_kv_cache_update()."""


def unified_attention_with_output(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    output: torch.Tensor, layer_name: str,
) → None:
    """Custom op dispatch: get_forward_context → impl.forward()."""
```

```python
# C032 现状
# HF model.forward() 使用内置 attention, 无 ForwardContext/AttentionBackend 集成
```

---

## 四、核心数据流

```
EngineCore.step() / step_with_batch_queue()
  │
  1. scheduler_output = scheduler.schedule(throttle_prefills)
     │ KVCacheCoordinator.find_longest_cache_hit(block_hashes, ...)
     │ KVCacheCoordinator.allocate_new_blocks(request, num_new_tokens)
     │ BlockHashToBlockMap 哈希碰撞处理
     │ _maybe_evict_cached_block() CoW eviction
     │ → SchedulerOutput: sched_new_reqs, sched_cached_reqs, num_common_prefix_blocks, ...
  │
  2. model_output = model_executor.execute_model(scheduler_output)
     │ Worker.execute_model(scheduler_output)
     │   ├─ RequestState.add_request(req_id, prompt_len, all_token_ids, ...)
     │   ├─ BlockTables.compute_slot_mappings(idx_mapping, query_start_loc, positions, ...)
     │   ├─ CachedRequestState + InputBatch 构建
     │   ├─ build_attn_metadata(input_batch)
     │   ├─ set_forward_context(attn_metadata, vllm_config, ...)
     │   ├─ CudagraphDispatcher.dispatch(num_tokens, uniform_decode) → CUDAGraphMode
     │   ├─ Model.forward()
     │   │   └─ Attention.forward():
     │   │       ctx = get_forward_context()
     │   │       self.impl.forward(Q, K, V, kv_cache, ctx.attn_metadata)
     │   │       └─ Triton PagedAttention kernel
     │   └─ Sampler.forward(logits, sampling_metadata) → SamplerOutput
     │   → ModelRunnerOutput
  │
  3. engine_outputs = scheduler.update_from_output(scheduler_output, model_runner_output)
     → dict[int, EngineCoreOutputs]
  │
  4. OutputProcessor.process_outputs(engine_core_outputs)
     ├─ IncrementalDetokenizer.update(new_token_ids, stop_terminated)
     │   └─ check_stop_strings(output_text, new_char_count, stop, include_in_output)
     ├─ RequestState.make_request_output() → CompletionOutput → RequestOutput
     └─ RequestOutputCollector → async streaming → client
```

---

## 五、逐文件对比（C032 → C050 → vLLM）

| C050 文件 | C032 状态 | C050 vs vLLM |
|-----------|-----------|----------------|
| `v1/kv_cache_interface.py` | KVCacheSpec (block_size, dtype) | +AttentionSpec(head_size), FullAttentionSpec(head_size_v), SlidingWindowSpec, MLAAttentionSpec, KVCacheGroupSpec(layer_names, kv_cache_spec), KVCacheConfig |
| `v1/core/kv_cache_coordinator.py` | **不存在** | KVCacheCoordinator.find_longest_cache_hit() + HybridKVCacheCoordinator（对齐 vLLM） |
| `v1/core/block_pool.py` | ref_cnt, LRU, hash cache | +BlockHashToBlockMap, free_blocks() ref_cnt cascading (C054c), _maybe_evict_cached_block(), emit_cached_block_events() |
| `v1/core/kv_cache_utils.py` | KVCacheBlock, FreeKVCacheBlockQueue | +KVCacheBlockCopy(NamedTuple), BlockHash, BlockHashWithGroupId, slots=True |
| `v1/core/sched/interface.py` | **不存在** | SchedulerInterface (20 方法，对齐 vLLM 签名) |
| `v1/core/sched/scheduler.py` | FCFS, update_from_output | +schedule(throttle_prefills), finish_requests(), make_stats(), shutdown() |
| `v1/core/sched/output.py` | 基础字段 | +num_common_prefix_blocks, kv_cache_block_copies, new_block_ids_to_zero |
| `v1/worker/gpu_input_batch.py` | InputBatch dataclass | CachedRequestState (18 字段) + InputBatch (15 字段) + InputBuffers |
| `v1/worker/gpu/states.py` | 4 GPU 字段 | +add_request(req_id, prompt_len, all_token_ids, num_computed_tokens, max_tokens), apply_staged_writes() |
| `v1/worker/block_table.py` | Python for loop | Triton compute_slot_mappings(idx_mapping, query_start_loc, positions, num_tokens_padded), gather_block_tables(), apply_staged_writes() |
| `v1/worker/gpu/attn_utils.py` | **不存在** | init_attn_backend(), init_kv_cache(), build_attn_metadata(), build_slot_mappings_by_layer() |
| `v1/worker/utils.py` | **不存在** | AttentionGroup, bind_kv_cache() |
| `forward_context.py` | **不存在** | BatchDescriptor(frozen, has_lora, num_active_loras), set_forward_context(), get_forward_context() |
| `v1/attention/backend.py` | 基础 ABC | +AttentionType(DECODER/ENCODER/...), AttentionImplBase, AttentionMetadataBuilder(ABC), CommonAttentionMetadata, get_kv_cache_shape() |
| `v1/attention/backends/registry.py` | **不存在** | AttentionBackendEnum (FLASH_ATTN, TRITON_ATTN, TORCH_SDPA) |
| `v1/attention/backends/flash_attn.py` | 存在 (stub) | C043-A: FlashAttentionBackend + FlashAttentionImpl + FlashAttentionMetadataBuilder 完整实现（PyTorch, 不依赖 flash-attn 库）。对齐 vLLM get_kv_cache_shape 和 backend dispatch |
| `v1/attention/backends/triton_attn.py` | **不存在** | C043-B: TritonAttentionBackend + TritonAttentionImpl（对齐 vLLM） |
| `v1/attention/ops/triton_reshape_and_cache_flash.py` | **不存在** | C043-C: Triton reshape_and_cache kernel |
| `v1/attention/ops/triton_decode_attention.py` | **不存在** | C043-C: Triton decode attention kernel |
| `v1/attention/ops/triton_prefill_attention.py` | **不存在** | C043-C: Triton prefill attention kernel |
| `v1/attention/ops/triton_unified_attention.py` | **不存在** | C043-B: unified_attention_with_output custom op |
| `v1/attention/ops/chunked_prefill_paged_decode.py` | **不存在** | C043-C: chunked prefill + paged decode fusion kernel |
| `v1/worker/gpu/attn_utils.py` | **不存在** | C044-A: init_attn_backend(), init_kv_cache(), build_attn_metadata(), build_slot_mappings_by_layer() |
| `v1/cudagraph_dispatcher.py` | **不存在** | C044-B: CudagraphDispatcher.dispatch(num_tokens, uniform_decode), get_capture_descs() |
| `v1/engine/__init__.py` | 空 | EngineCoreRequest(msgspec), EngineCoreOutputs, FinishReason(IntEnum), PauseMode |
| `v1/engine/core.py` | step() 闭环 | +add_request, abort_requests, shutdown, sleep, wake_up, reset_prefix_cache, step_with_batch_queue |
| `v1/engine/core_client.py` | **不存在** | EngineCoreClient(ABC), InprocClient, SyncMPClient |
| `v1/engine/async_llm.py` | asyncio.Queue | +output_handler, generate() → AsyncGenerator, streaming queue |
| `v1/engine/output_processor.py` | token→text | +RequestOutputCollector, RequestState.make_request_output() |
| `v1/engine/detokenizer.py` | incremental | +FastIncrementalDetokenizer, SlowIncrementalDetokenizer, check_stop_strings() |
| `v1/executor/multiproc_executor.py` | **不存在** | MultiprocExecutor (WorkerProcess lifecycle + shared memory dispatch) |
| `v1/metrics/stats.py` | **不存在** | SchedulerStats, IterationStats |
| `v1/metrics/loggers.py` | **不存在** | PrometheusStatLogger (TTFT/TPOT/ITL/throughput/KV util) |
| `v1/metrics/prometheus.py` | **不存在** | setup_multiprocess_prometheus() |
| `entrypoints/openai/api_server.py` | **不存在** | /v1/chat/completions (streaming), /v1/completions |
| `distributed/tensor_parallel.py` | **不存在** | TP=2 with ColumnParallelLinear + RowParallelLinear + all_reduce/all_gather |

---

## 六、架构差距总结

| 维度 | C032 → C050 | 对齐 vLLM 关键点 |
|------|-------------|-------------------|
| KV Cache | KVCacheSpec 体系 + KVCacheCoordinator.find_longest_cache_hit() | 字段名: head_size, layer_names, kv_cache_spec |
| Block 管理 | BlockHashToBlockMap + free_blocks() + CoW | 方法名: _maybe_evict_cached_block, emit_cached_block_events |
| 调度 | SchedulerInterface ABC → schedule(throttle_prefills) | 方法签名同 vLLM |
| Attention | ForwardContext → Attention.forward() → unified_kv_cache_update / unified_attention_with_output（独立函数） → FlashAttentionImpl/TritonAttentionImpl.forward | C041 + C042 + C043-A/B/C + C044-A 全链路 |
| Triton | reshape_and_cache + decode + prefill + unified_attention + chunked_prefill_paged_decode 5 kernel | 文件命名同 vLLM ops |
| CUDA Graph | CudagraphDispatcher.dispatch() | 类名/方法名同 vLLM |
| Engine | EngineCoreClient(ABC) → InprocClient | 类名同 vLLM client hierarchy |
| 多进程 | MultiprocExecutor | WorkerProcess lifecycle |
| 采样 | Sampler.forward(logits, sampling_metadata) | SamplingMetadata 18 字段 |
| Streaming | RequestOutputCollector + check_stop_strings() | 类名/函数名同 vLLM |
| TP | ColumnParallelLinear + RowParallelLinear | 类名同 vLLM |

---

## 七、C050 故意保留的差距（C051+ 阶段）

| 能力 | 目标阶段 |
|------|----------|
| Priority Scheduling / Preemption / Async Scheduler | C051-C053 |
| KV Metrics / KV Offload | C054-C055 |
| Advanced Sampler / Incremental Logprobs | C056-C057 |
| Speculative Decoding (Eagle/Medusa/n-gram) + RejectionSampler | C058-C062 |
| Structured Output (XGrammar/Outlines) | C063-C065 |
| Pooling (Embedding + Late Interaction) | C066-C067 |
| SingleTypeKVCacheManager 其余 3 种 | → C068 (C071-C100) |
| Config 10 类 + VllmConfig | C069 |
| LoRA / MultiModal / Quantization | C078-C088 |
| PP / DP / Collective Ops | C074-C077 |
| SDK (LLM API / CLI) | C096-C098 |
| Benchmark + Integration | C099-C100 |
| FlashInfer / ROCm / MLA / 其余 attention backends | C101+ 生态适配 |

## 八、C050 判断标准

1. `EngineCore.step() → schedule(throttle_prefills) → execute_model → update_from_output(scheduler_output, model_runner_output) → dict[int, EngineCoreOutputs]`
2. KVCacheSpec(FullAttentionSpec/SlidingWindowSpec/MLAAttentionSpec) + KVCacheCoordinator.find_longest_cache_hit()
3. KVCacheBlockCopy + _maybe_evict_cached_block + BlockHashToBlockMap
4. SchedulerInterface 20 方法签名与 vLLM 一致
5. **Attention 层**: `Attention(nn.Module)` + `unified_kv_cache_update()` + `unified_attention_with_output()` 独立函数
6. **FlashAttentionBackend**: C043-A — FlashAttentionImpl + FlashAttentionMetadataBuilder (get_kv_cache_shape, stride_order, block_table 参数)
7. **TritonAttentionBackend**: C043-B — TritonAttentionImpl + unified_attention kernel (2D prefill / 3D decode)
8. **Triton ops**: C043-C — reshape_and_cache + decode_attention + prefill_attention + unified_attention + chunked_prefill_paged_decode + flash_attention_tiled 6 个 kernel
9. **KV Cache Init**: C044-A — init_attn_backend, init_kv_cache, build_attn_metadata, build_slot_mappings_by_layer 完整 pipeline
10. **CUDA Graph**: C044-B — CudagraphDispatcher.dispatch() batch=[1,2,4,8]
11. **EngineCoreClient(ABC) → InprocClient**
12. **MultiprocExecutor**
13. **RequestOutputCollector + FastIncrementalDetokenizer + check_stop_strings()**
14. **PrometheusStatLogger**: TTFT, TPOT, ITL, throughput, KV util
15. **/v1/chat/completions** (streaming), /v1/completions
16. **TP=2** ColumnParallelLinear + RowParallelLinear
17. **每个 commit `pytest` 全绿**

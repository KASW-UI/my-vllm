# C100 最终架构 — 对照 vLLM V1 参考实现

> C100 每个类名、方法名、dataclass 字段必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。
> 差距控制在 10% 以内（仅省略 CUDA kernel / 平台适配 / 全路径 error handling）。

## 一、C100 最终文件树

```
my_vllm/
├── __init__.py
│
├── config/                                    # C069 + C078-C080
│   ├── __init__.py
│   ├── vllm.py                                # VllmConfig (聚合 10 config)
│   ├── model.py                               # ModelConfig
│   ├── cache.py                               # CacheConfig
│   ├── scheduler.py                           # SchedulerConfig
│   ├── parallel.py                            # ParallelConfig
│   ├── lora.py                                # LoRAConfig          ← C078
│   ├── speculative.py                         # SpeculativeConfig   ← C078
│   ├── quantization.py                        # QuantizationConfig  ← C078
│   ├── multimodal.py                          # MultiModalConfig    ← C078
│   └── observability.py                       # ObservabilityConfig ← C080
│
├── v1/                                        # V1 Core Architecture
│   ├── __init__.py
│   ├── request.py                             # Request + RequestStatus + SamplingParams
│   ├── outputs.py                             # SamplerOutput, ModelRunnerOutput, LogprobsLists,
│   │                                          # LogprobsTensors, DraftTokenIds
│   ├── utils.py                               # ConstantList, CpuGpuBuffer
│   ├── serial_utils.py                        # MsgpackEncoder, MsgpackDecoder
│   ├── kv_cache_interface.py                  # KVCacheSpec, FullAttentionSpec, SlidingWindowSpec,
│   │                                          # MLASpec, MambaSpec, ChunkedLocalAttentionSpec,
│   │                                          # CrossAttentionSpec, KVCacheGroupSpec, KVCacheConfig
│   ├── kv_cache_spec_registry.py              # KVCacheSpecRegistry                    ← C068
│   ├── forward_context.py                     # BatchDescriptor, set_forward_context,
│   │                                          # get_forward_context                     ← C041
│   ├── cudagraph_dispatcher.py                # CudagraphDispatcher                     ← C044
│   │
│   ├── engine/                                # Engine layer
│   │   ├── __init__.py                        # EngineCoreRequest, EngineCoreOutput,
│   │   │                                      # EngineCoreOutputs, FinishReason,
│   │   │                                      # EngineCoreRequestType, PauseMode
│   │   ├── core.py                            # EngineCore — step(), add_request(),
│   │   │                                      # abort_requests(), shutdown(), sleep(), wake_up(),
│   │   │                                      # reset_prefix_cache(), add_lora(), remove_lora(),
│   │   │                                      # collective_rpc()
│   │   ├── core_client.py                     # EngineCoreClient(ABC), InprocClient,     ← C045/C090
│   │   │                                      # SyncMPClient, AsyncMPClient
│   │   ├── llm_engine.py                      # LLMEngine — from_engine_args(),
│   │   │                                      # add_request(), step(), get_metrics(),
│   │   │                                      # collective_rpc(), apply_model()
│   │   ├── async_llm.py                       # AsyncLLM — generate(), encode(),        ← C047
│   │   │                                      # abort(), add_request()
│   │   ├── output_processor.py                # OutputProcessor, RequestOutputCollector, ← C047
│   │   │                                      # RequestState
│   │   ├── input_processor.py                 # InputProcessor — process_inputs()        ← C090 MM
│   │   ├── detokenizer.py                     # IncrementalDetokenizer,
│   │   │                                      # FastIncrementalDetokenizer,              ← C047
│   │   │                                      # SlowIncrementalDetokenizer,
│   │   │                                      # check_stop_strings()
│   │   ├── exceptions.py                      # EngineGenerateError, EngineDeadError
│   │   ├── logprobs.py                        # LogprobsProcessor                        ← C057
│   │   ├── parallel_sampling.py               # ParentRequest                           ← C058
│   │   └── utils.py                           # EngineZmqAddresses,                      ← C073
│   │                                          # wait_for_engine_startup()
│   │
│   ├── core/                                  # Scheduler & KV Cache core
│   │   ├── __init__.py
│   │   ├── block_pool.py                      # KVCacheBlock, BlockPool,
│   │   │                                      # BlockHashToBlockMap                     ← C035
│   │   ├── kv_cache_manager.py                # KVCacheManager, KVCacheBlocks
│   │   ├── kv_cache_coordinator.py            # KVCacheCoordinator,                      ← C034
│   │   │                                      # HybridKVCacheCoordinator
│   │   ├── single_type_kv_cache_manager.py     # FullAttentionManager,                   ← C034+C068
│   │   │                                      # SlidingWindowManager,
│   │   │                                      # [MLA via v1/attention/backends/mla/]
│   │   │                                      # CrossAttentionManager,                   ← C068
│   │   │                                      # MambaManager,                            ← C068
│   │   │                                      # ChunkedLocalAttentionManager             ← C068
│   │   ├── kv_cache_utils.py                  # KVCacheBlock, FreeKVCacheBlockQueue,
│   │   │                                      # KVCacheBlockCopy, BlockHash,
│   │   │                                      # BlockHashWithGroupId
│   │   ├── kv_cache_metrics.py                # BlockMetricsState,                       ← C054
│   │   │                                      # KVCacheMetricsCollector
│   │   ├── encoder_cache_manager.py           # EncoderCacheManager,                     ← C089
│   │   │                                      # EncoderDecoderCacheManager
│   │   ├── prefix_cache.py                    # CoW: shared block → private block        ← C036
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── interface.py                   # SchedulerInterface (ABC)                  ← C037
│   │       ├── scheduler.py                   # Scheduler — schedule(),
│   │       │                                  # update_from_output(), get_grammar_bitmask(),
│   │       │                                  # finish_requests(), make_stats(), shutdown()
│   │       ├── async_scheduler.py             # AsyncScheduler                           ← C053
│   │       ├── output.py                      # SchedulerOutput, NewRequestData,
│   │       │                                  # CachedRequestData, GrammarOutput
│   │       ├── request_queue.py               # FCFSRequestQueue, PriorityRequestQueue,
│   │       │                                  # SchedulingPolicy                        ← C051
│   │       └── utils.py                       # check_stop(), remove_all(),
│   │                                          # check_sequence_repetition()
│   │
│   ├── worker/                                # Model execution
│   │   ├── __init__.py
│   │   ├── worker_base.py                     # WorkerBase (ABC) — execute_model(),
│   │   │                                      # sample_tokens(), compile_or_warm_up_model(),
│   │   │                                      # add_lora(), remove_lora(), list_loras()
│   │   ├── gpu_worker.py                      # Worker — init_device(), load_model(),
│   │   │                                      # execute_model(), sample_tokens(),
│   │   │                                      # add_lora(), remove_lora(), shutdown()
│   │   ├── gpu_model_runner.py                # GPUModelRunner — execute_model(),
│   │   │                                      # sample_tokens(), propose_draft_token_ids(),
│   │   │                                      # initialize_kv_cache(), capture_model()
│   │   ├── gpu_input_batch.py                 # CachedRequestState, InputBatch, InputBuffers ← C039
│   │   ├── block_table.py                     # BlockTables — append_block_ids(),         ← C040
│   │   │                                      # gather_block_tables(), compute_slot_mappings()
│   │   ├── lora_model_runner_mixin.py         # LoRA forward integration                 ← C084
│   │   ├── kv_connector_model_runner_mixin.py # KV connector integration                 ← C092
│   │   ├── dp_utils.py                        # DP batch coordination                    ← C075
│   │   ├── ubatching.py                       # Micro-batch splitting
│   │   ├── utils.py                           # AttentionGroup, bind_kv_cache()
│   │   ├── workspace.py                       # GPU workspace management
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── input_batch.py                 # InputBatch, CachedRequestState
│   │       ├── block_table.py                 # BlockTables (Triton 化)
│   │       ├── states.py                      # RequestState — add_request(),
│   │       │                                  # remove_request(), apply_staged_writes()
│   │       ├── attn_utils.py                  # init_attn_backend(), init_kv_cache(),
│   │       │                                  # build_attn_metadata(), build_slot_mappings_by_layer()
│   │       ├── model_states/
│   │       │   ├── __init__.py
│   │       │   ├── interface.py               # ModelState (ABC)
│   │       │   └── default.py                 # DefaultModelState
│   │       ├── lora_utils.py                  # LoRA LRU cache                           ← C083
│   │       ├── pp_utils.py                    # Pipeline parallel utils                  ← C074
│   │       ├── kv_connector.py                # KV connector worker                      ← C092
│   │       ├── mm/                            # MultiModal encoder runner                ← C091
│   │       │   └── encoder_runner.py
│   │       ├── spec_decode/                   # Spec decode worker                       ← C058
│   │       │   └── ...
│   │       └── structured_outputs.py          # Grammar bitmask at GPU                   ← C063
│   │
│   ├── executor/                              # Execution backend
│   │   ├── __init__.py
│   │   ├── abstract.py                        # Executor (ABC) — execute_model(),
│   │   │                                      # initialize_cache()
│   │   ├── uniproc_executor.py                # UniProcExecutor
│   │   └── multiproc_executor.py              # MultiprocExecutor                        ← C046/C071
│   │
│   ├── attention/                             # Attention backends
│   │   ├── __init__.py
│   │   ├── backend.py                         # AttentionType (DECODER/ENCODER/...),
│   │   │                                      # AttentionBackend (ABC),
│   │   │                                      # AttentionImpl (ABC), AttentionImplBase,
│   │   │                                      # AttentionMetadataBuilder (ABC),
│   │   │                                      # CommonAttentionMetadata
│   │   ├── selector.py                        # get_attn_backend()
│   │   ├── registry.py                        # AttentionBackendEnum                     ← C042
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── torch_attn.py                  # [my-vllm 特有] SDPA fallback, vLLM V1 无此backend
│   │       ├── flash_attn.py                  # FlashAttentionBackend (C050 完整, PyTorch)
│   │       ├── triton_attn.py                 # TritonAttentionBackend                   ← C043
│   │       └── ops/
│   │           ├── __init__.py
│   │           ├── paged_attn.py              # Python 参考实现
│   │           ├── triton_reshape_and_cache_flash.py # Triton reshape_and_cache                ← C043
│   │           ├── triton_decode_attention.py  # Triton decode attention                 ← C043
│   │           └── triton_prefill_attention.py # Triton prefill attention                ← C043
│   │
│   ├── sample/                                # Sampling pipeline
│   │   ├── __init__.py
│   │   ├── sampler.py                         # Sampler — forward(), sample(),
│   │   │                                      # compute_logprobs(), gather_logprobs(),
│   │   │                                      # greedy_sample(), apply_temperature(),
│   │   │                                      # apply_penalties(), apply_logits_processors()
│   │   ├── rejection_sampler.py               # RejectionSampler — forward(),            ← C059
│   │   │                                      # rejection_sample(), parse_output()
│   │   ├── metadata.py                        # SamplingMetadata — temperature, top_k,
│   │   │                                      # top_p, generator, max_num_logprobs,
│   │   │                                      # no_penalties, frequency_penalties,
│   │   │                                      # presence_penalties, repetition_penalties,
│   │   │                                      # output_token_ids, bad_words_token_ids,
│   │   │                                      # allowed_token_ids_mask, logitsprocs
│   │   ├── logits_processor/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py                   # LogitsProcessor ABC
│   │   │   ├── builtin.py                     # MinTokensLogitsProcessor
│   │   │   └── state.py                       # BatchUpdateBuilder
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── penalties.py                   # apply_all_penalties (Triton)
│   │       ├── bad_words.py                   # apply_bad_words
│   │       ├── logprobs.py                    # batch count greater than
│   │       ├── topk_topp_sampler.py           # TopKTopPSampler
│   │       └── topk_topp_triton.py            # Triton top-k/top-p kernel
│   │
│   ├── spec_decode/                           # Speculative decoding                    ← C058-C062
│   │   ├── __init__.py
│   │   ├── eagle.py                           # EagleProposer
│   │   ├── medusa.py                          # MedusaProposer
│   │   ├── ngram_proposer.py                  # NgramProposer
│   │   └── metadata.py                        # SpecDecodeMetadata
│   │
│   ├── structured_output/                     # Structured output                        ← C063-C065
│   │   ├── __init__.py
│   │   ├── backend_xgrammar.py                # XgrammarBackend
│   │   ├── backend_outlines.py                # OutlinesBackend
│   │   └── backend_types.py                   # StructuredOutputBackend (ABC)
│   │
│   ├── pool/                                  # Pooling                                  ← C066-C067
│   │   ├── __init__.py
│   │   ├── pooling_runner.py                # GPUModelRunner pooling path (encode)
│   │   └── late_interaction_runner.py         # LateInteractionRunner
│   │
│   ├── kv_offload/                            # KV Cache offload                        ← C055
│   │   ├── __init__.py
│   │   ├── base.py                            # OffloadingManager (ABC)
│   │   ├── config.py                          # OffloadConfig
│   │   └── cpu/
│   │       ├── __init__.py
│   │       └── cpu_backend.py                 # CPU RAM backend
│   │
│   └── metrics/                               # Metrics                                 ← C048
│       ├── __init__.py
│       ├── stats.py                           # SchedulerStats, IterationStats,
│       │                                      # PrefixCacheStats, RequestStateStats
│       ├── loggers.py                         # PrometheusStatLogger, StatLoggerBase
│       ├── prometheus.py                      # setup_multiprocess_prometheus()
│       └── reader.py                          # Metric, Counter, Gauge, Histogram
│
├── model_executor/                            # Model loading bridge
│   ├── __init__.py
│   ├── model_loader.py                        # load_model()
│   ├── models/
│   │   └── opt.py
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                          # ColumnParallelLinear, RowParallelLinear (TP)
│       └── attention/
│           ├── __init__.py
│           └── attention.py                   # Attention(nn.Module)
│
├── triton_kernels/                            # GPU 计算基础（C050b + C043 扩展）
│   ├── gemm.py                                # Triton tiled GEMM kernel
│   └── flash_attention_tiled.py              # FlashAttention tiling 教育性 kernel
│
├── lora/                                      # LoRA                                     ← C081-C082
│   ├── __init__.py
│   ├── layers.py                              # BaseLayerWithLoRA, ColumnParallelLinearWithLoRA
│   ├── config.py                              # LoRAConfig export
│   └── request.py                             # LoRARequest
│
├── quantization/                              # Quantization dispatch (in model_executor/layers/)
│   ├── __init__.py
│   └── config.py                              # QuantizationConfig export
│
├── multimodal/                                # MultiModal                               ← C088
│   ├── __init__.py
│   ├── registry.py                            # MultiModalRegistry
│   └── vision.py                              # VisionFeature
│
├── distributed/                               # Distributed
│   ├── __init__.py
│   ├── parallel_state.py                      # TP/PP/DP group management,
│   │                                          # all_reduce(), all_gather(), broadcast()
│   └── communication_op.py                    # collective ops
│
├── engine/                                    # Engine public layer
│   ├── arg_utils.py                           # EngineArgs, AsyncEngineArgs              ← C095
│   └── protocol.py                            # EngineClient (ABC)                       ← C095
│
├── entrypoints/                               # Entry points                             ← C096-C098
│   ├── llm.py                                 # LLM — generate(), chat(), encode(),
│   │                                          # collective_rpc(), apply_model()
│   ├── openai/
│   │   ├── __init__.py
│   │   └── api_server.py                      # /v1/chat/completions, /v1/completions,
│   │                                          # /v1/embeddings
│   ├── cli/
│   │   ├── serve.py                           # python -m my_vllm serve
│   │   └── generate.py                        # python -m my_vllm generate
│   └── chat_utils.py
│
├── runtime/                                   # Legacy runtime (test compat)
│   ├── __init__.py
│   ├── model_loader.py
│   ├── tokenizer.py
│   └── model.py                               # LLM shim
│
└── server/                                    # C014 legacy (deprecated after C049)
    └── ...
```

**数量对照**

| 来源 | my_vllm/ .py 文件数 |
|------|---------------------|
| C032 基线 | ~51 |
| C100 目标 | ~120 |
| vLLM V1 `v1/` 目录 | ~220 |

C100 达到 vLLM 的 ~55% 文件覆盖率，但核心执行链（`v1/engine/` + `v1/core/` + `v1/worker/` + `v1/attention/` + `v1/sample/`）达到 ~85%。

---

## 二、组件层次与职责边界

### C100 目标（对齐 vLLM V1）

```
API / SDK Layer:
  entrypoints/llm.py:LLM.generate(…) / chat(…) / encode(…)
  entrypoints/openai/api_server.py:/v1/chat/completions
  engine/arg_utils.py:EngineArgs → VllmConfig

Engine Layer:
  engine/protocol.py:EngineClient (ABC)
    ├── InprocClient          — 同进程直接调用 EngineCore
    ├── SyncMPClient          — ZMQ REP/REQ 同步
    └── AsyncMPClient         — ZMQ DEALER/ROUTER 异步
  v1/engine/async_llm.py:AsyncLLM(EngineClient)
    ├── generate() → AsyncGenerator[RequestOutput]
    ├── encode() → AsyncGenerator[PoolingRequestOutput]
    └── output_handler background task

EngineCore (v1/engine/core.py):
  step() → step_with_batch_queue()
    │
    ├─▶ Scheduler (v1/core/sched/scheduler.py)
    │   │   via SchedulerInterface (v1/core/sched/interface.py)
    │   │   schedule(throttle_prefills) → SchedulerOutput
    │   │   update_from_output(scheduler_output, model_runner_output)
    │   │     → dict[int, EngineCoreOutputs]
    │   │   get_grammar_bitmask(scheduler_output) → GrammarOutput
    │   │
    │   ├─ RequestQueue (FCFSRequestQueue / PriorityRequestQueue)
    │   │    + SchedulingPolicy (C051)
    │   ├─ Preemption: RECOMPUTE / SWAP (C052)
    │   ├─ Async scheduling mode (C053)
    │   │
    │   └─ KVCacheCoordinator (v1/core/kv_cache_coordinator.py)
    │       ├─ find_longest_cache_hit(block_hashes, max_cache_hit_length)
    │       │     → tuple[tuple[list[KVCacheBlock], ...], int, int]
    │       └─ SingleTypeKVCacheManager (v1/core/single_type...py)
    │           ├─ FullAttentionManager       (via KVCacheManager)
    │           ├─ SlidingWindowManager        │
    │           └─ [MLA: v1/attention/backends/mla/ 管理, 非 SingleTypeManager]
    │           ├─ CrossAttentionManager       │
    │           ├─ MambaManager                │
    │           └─ ChunkedLocalAttentionManager│
    │               └─ BlockPool (v1/core/block_pool.py)
    │                   ├─ get_new_blocks(), free_blocks()
    │                   ├─ touch(), get_cached_block()
    │                   ├─ _maybe_evict_cached_block()  (CoW)
    │                   ├─ BlockHashToBlockMap
    │                   ├─ free_blocks()
    │                   └─ emit_cached_block_events()
    │
    └─▶ Executor (MP) → Worker × N (TP×PP×DP)
          │
          ├─ MultiprocExecutor (v1/executor/multiproc_executor.py)
          │   ├─ WorkerProcess lifecycle (spawn/health_check/shutdown)
          │   ├─ shared memory batch dispatch via tensor_ipc
          │   └─ Worker failure callback
          │
          └─▶ Worker (v1/worker/gpu_worker.py)
                │ execute_model(SchedulerOutput) → ModelRunnerOutput
                │
                └─▶ GPUModelRunner (v1/worker/gpu_model_runner.py)
                      │ execute_model(scheduler_output) → ModelRunnerOutput
                      │
                      ├─ RequestState.add_request() / apply_staged_writes()
                      ├─ BlockTables.append_block_ids() / compute_slot_mappings()
                      ├─ InputBatch / CachedRequestState
                      │
                      ├─ ForwardContext.set(attn_metadata, slot_mapping, …)
                      ├─ Model.forward(input_ids, …)
                      │   └─ AttentionLayer.forward():
                      │       ├─ get_forward_context() → ForwardContext
                      │       ├─ QKV proj
                      │       └─ AttentionImpl.forward(q, k, v, kv_cache, attn_metadata)
                      │           └─ Triton PagedAttention (paged_attn.py → Triton)
                      │
                      ├─ CUDAGraph [batch=1,2,4,8] (CudagraphDispatcher)
                      │
                      └─ RejectionSampler (if spec decode)
                          │ acceptance/rejection + bonus token
                          └─ Sampler.forward(logits, sampling_metadata)
                                ├─ apply_logits_processors()
                                ├─ apply_penalties()
                                ├─ apply_temperature()
                                ├─ greedy_sample() / top_k top_p sample()
                                └─ compute_logprobs()
```

输出侧:
```
OutputProcessor.process_outputs(engine_core_outputs)
  ├─ RequestOutputCollector — asyncio.Queue async aggregation
  ├─ RequestState.make_request_output()
  │     ├─ IncrementalDetokenizer (FastIncrementalDetokenizer / Slow)
  │     │     └─ check_stop_strings()
  │     └─ LogprobsProcessor.update_from_output()
  └─ RequestOutput / PoolingRequestOutput → streaming delta
```

关键约束:
- `LLMEngine` 不直接调用 `Scheduler`，只通过 `EngineCore.step()`
- `SchedulerInterface` 是调度器的唯一抽象入口
- `KVCacheCoordinator` 是多 attention type 的唯一路由层
- `ForwardContext` 是 attention backend 接入 model forward 的唯一通道
- `EngineClient` 是 engine 与 frontend 的唯一边界
- 所有组件间通信通过结构化 dataclass / msgspec.Struct

---

## 三、核心数据结构（与 vLLM 对齐）

### 1. EngineCoreRequest（`v1/engine/__init__.py`）

```python
# C100 目标（与 vLLM 对齐，省略 mm_features, trace_headers, prompt_embeds 字段）
class EngineCoreRequest(msgspec.Struct):
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None          # C078
    cache_salt: str | None
    priority: int = 0                          # C051
```

### 2. KVCacheSpec 体系（`v1/kv_cache_interface.py`）

```python
# C100 目标（9 种 spec，对齐 vLLM）
class KVCacheSpecKind(str, Enum):             # 对齐 vLLM KVCacheSpecKind
    FULL_ATTENTION, MLA_ATTENTION, SLIDING_WINDOW, MAMBA,
    CHUNKED_LOCAL_ATTENTION, CROSS_ATTENTION, ENCODER_ONLY_ATTENTION

class KVCacheSpec:                            # 对齐 vLLM KVCacheSpec
    block_size: int

class AttentionSpec(KVCacheSpec):             # 对齐 vLLM AttentionSpec
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype

class FullAttentionSpec(AttentionSpec):       # 对齐 vLLM FullAttentionSpec
    head_size_v: int
    sliding_window: int | None

class MLAAttentionSpec(FullAttentionSpec):    # 对齐 vLLM MLAAttentionSpec
    cache_dtype_str: str

class SlidingWindowSpec(AttentionSpec):       # 对齐 vLLM SlidingWindowSpec
    sliding_window: int
    head_size_v: int

class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int

class CrossAttentionSpec(AttentionSpec): ...

class MambaSpec(KVCacheSpec):
    shapes: list[tuple[int, ...]]
    dtypes: list[torch.dtype]

class KVCacheGroupSpec:                       # 对齐 vLLM KVCacheGroupSpec
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec

class KVCacheConfig:                          # 对齐 vLLM KVCacheConfig
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
```

### 3. KVCacheBlock + BlockPool（`v1/core/kv_cache_utils.py` + `block_pool.py`）

```python
# C100 目标（完全对齐 vLLM 字段，slots=True）
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    _block_hash: BlockHashWithGroupId | None = None       # vLLM 使用 BlockHashWithGroupId
    _block_hash_num_tokens: int | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False
    # Properties: block_hash, block_hash_num_tokens
    # Methods: set_block_hash(), reset_hash()

class FreeKVCacheBlockQueue:
    num_free_blocks: int

class KVCacheBlockCopy(NamedTuple):           # 对齐 vLLM
    src_block_id: int
    dst_block_id: int

class BlockHashToBlockMap:                    # 对齐 vLLM（C035）
    def get(block_hash: BlockHash) → KVCacheBlock | None
    def insert(block_hash: BlockHash, block: KVCacheBlock)
```

### 4. SchedulerInterface（`v1/core/sched/interface.py`）

```python
# C100 目标（完全对齐 vLLM 方法签名）
class SchedulerInterface(ABC):
    def schedule(self, throttle_prefills: bool = False) → SchedulerOutput
    def get_grammar_bitmask(self, scheduler_output: SchedulerOutput) → GrammarOutput | None
    def update_from_output(self, scheduler_output: SchedulerOutput,
                           model_runner_output: ModelRunnerOutput) → dict[int, EngineCoreOutputs]
    def add_request(self, request: Request) → None
    def finish_requests(self, request_ids, finished_status) → list[Request]
    def get_num_unfinished_requests(self) → int
    def has_unfinished_requests(self) → bool
    def has_finished_requests(self) → bool
    def reset_prefix_cache(self, reset_running_requests=False,
                           reset_connector=False) → bool
    def make_stats(self) → SchedulerStats | None
    def shutdown(self) → None
```

### 5. SchedulerOutput（`v1/core/sched/output.py`）

```python
# C100 目标（17 个字段，对齐 vLLM）
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    block_ids: tuple[list[int], ...]           # vLLM: per kv_cache_group
    num_computed_tokens: int
    lora_request: LoRARequest | None

@dataclass
class CachedRequestData:
    req_ids: list[str]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
    @property
    def num_reqs(self) → int

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]    # C058
    num_common_prefix_blocks: list[int]                   # C034
    finished_req_ids: set[str]
    preempted_req_ids: set[str] | None                    # C052
    new_block_ids_to_zero: list[int] | None               # C035 CoW
    kv_cache_block_copies: list[KVCacheBlockCopy] | None  # C035 CoW

@dataclass
class GrammarOutput:                                      # C063
    structured_output_request_ids: list[str]
    grammar_bitmask: npt.NDArray[np.int32]
```

### 6. ForwardContext（`v1/forward_context.py` + `forward_context.py`）

```python
# C100 目标（完全对齐 vLLM）
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0

class ForwardContext:                          # 对齐 vLLM
    attn_metadata: Any
    vllm_config: VllmConfig
    num_tokens: int | None
    batch_descriptor: BatchDescriptor | None

@contextmanager
def set_forward_context(attn_metadata, vllm_config, num_tokens=None,
                        batch_descriptor=None, slot_mapping=None): ...

def get_forward_context() → ForwardContext: ...
```

### 7. CachedRequestState + InputBatch（`v1/worker/gpu_input_batch.py`）

```python
# C100 目标（对齐 vLLM 字段，35+ 字段）
@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list = field(default_factory=list)
    sampling_params: object | None = None
    generator: torch.Generator | None = None
    block_ids: tuple[list[int], ...] = ()
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    lora_request: object | None = None           # ← C081
    pooling_params: object | None = None
    prompt_embeds: torch.Tensor | None = None

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
    has_structured_output_reqs: bool = False
```

### 8. SamplingMetadata（`v1/sample/metadata.py`）

```python
# C100 目标（完全对齐 vLLM 字段）
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
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: object  # LogitsProcessors
```

---

## 四、核心数据流（C100 与 vLLM 对齐）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LLM.generate(prompt) / AsyncLLM.generate(prompt)                        │
│   ├─ InputProcessor.process_inputs() → EngineCoreRequest                │
│   │   ├─ tokenizer.encode(prompt)                                       │
│   │   ├─ MultiModalRegistry.process() (if multimodal)                   │
│   │   └─ LoRARequest attach (if lora)                                   │
│   │                                                                     │
│   ├─ EngineClient.add_request(EngineCoreRequest)                        │
│   │   └─ InprocClient → EngineCore.add_request()                        │
│   │      SyncMPClient → ZMQ REQ → EngineCoreProc ZMQ REP                │
│   │                                                                     │
│   └─ OutputCollector → async yield RequestOutput (streaming)            │
│                                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ EngineCore.step() / step_with_batch_queue()                             │
│                                                                         │
│  1. scheduler_output = scheduler.schedule(throttle_prefills)            │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ Scheduler.schedule():                                       │     │
│     │   ├─ SchedulingPolicy.select_from_queue (FCFS / Priority)   │     │
│     │   ├─ request_queue.pop_request()                            │     │
│     │   ├─ KVCacheCoordinator.allocate_slots():                   │     │
│     │   │   ├─ find_longest_cache_hit(block_hashes)               │     │
│     │   │   ├─ FullAttentionManager.allocate_new_blocks()         │     │ 
│     │   │   ├─ SlidingWindowManager.allocate_new_blocks()         │     │
│     │   │   └─ maybe _maybe_evict_cached_block() (CoW)              │     │
│     │   ├─ EncoderCacheManager (if multimodal)                    │     │ 
│     │   ├─ Draft token propose (if spec decode)                   │     │
│     │   ├─ Grammar compilation (if structured output)             │     │
│     │   └─ → SchedulerOutput                                      │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  2. grammar_output = scheduler.get_grammar_bitmask(scheduler_output)    │
│                                                                         │
│  3. future = model_executor.execute_model(scheduler_output)             │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ Worker.execute_model(scheduler_output):                     │     │
│     │   ├─ RequestState.add_request(req_id, prompt_len,           │     │
│     │   │   all_token_ids, num_computed_tokens)                   │     │
│     │   ├─ BlockTables.append_block_ids(req_index, new_block_ids) │     │
│     │   ├─ CachedRequestState 构建                                 │     │
│     │   ├─ InputBatch 构建 (input_ids, positions, query_start_loc, │     │
│     │   │   seq_lens, block_table, slot_mapping, logits_indices)  │     │
│     │   ├─ attn_metadata = build_attn_metadata(input_batch)       │     │
│     │   │                                                         │     │
│     │   ├─ ForwardContext.set(attn_metadata, vllm_config, …)      │     │
│     │   ├─ CudagraphDispatcher.dispatch(num_tokens, uniform):     │     │
│     │   │   → CUDAGraphMode + BatchDescriptor                     │     │
│     │   │                                                         │     │
│     │   ├─ Model.forward(input_ids, positions, …)                 │     │
│     │   │   └─ Each layer: AttentionLayer.forward(hidden_states): │     │
│     │   │       ├─ Q, K, V = self.qkv_proj(hidden)                │     │
│     │   │       ├─ ctx = get_forward_context()                    │     │
│     │   │       └─ self.impl.forward(Q, K, V, kv_cache,           │     │
│     │   │           ctx.attn_metadata, output=output)             │     │
│     │   │           └─ Triton PagedAttention kernel               │     │
│     │   │                                                         │     │
│     │   └─ Sampler.forward(logits, sampling_metadata):            │     │
│     │       ├─ apply_logits_processors(logits, metadata)          │     │
│     │       ├─ apply_penalties(logits, metadata, output_token_ids)│     │
│     │       ├─ greedy_sample(logits)                              │     │
│     │       └─ compute_logprobs(logits) → LogprobsTensors         │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  4. sample_tokens: RejectionSampler.forward (if spec decode)            │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ rejection_sample(draft_token_ids, num_draft_tokens,         │     │
│     │   draft_probs, target_logits, bonus_token_ids, metadata)    │     │
│     │ → accepted_mask, bonus_token_id                             │     │
│     │ parse_output(output_token_ids, vocab_size) → list[list[int]]│     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  5. engine_outputs = scheduler.update_from_output(                      │
│       scheduler_output, model_runner_output)                            │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ ├─ 提取 sampled_token_ids, logprobs                          │     │
│     │ ├─ 检查 stop (EOS, max_tokens, check_sequence_repetition)    │     │
│     │ ├─ finish_requests(request_ids, FINISHED_STOPPED/LENGTH)    │     │
│     │ ├─ KVCacheManager.free(finished)                            │     │
│     │ └─ → dict[int, EngineCoreOutputs]                           │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

输出侧:
  6. OutputProcessor.process_outputs(engine_core_outputs)
     ├─ RequestState.make_request_output(new_token_ids, finish_reason)
     │   ├─ IncrementalDetokenizer.update(new_token_ids, stop_terminated)
     │   │   └─ check_stop_strings(output_text, …)
     │   ├─ LogprobsProcessor.update_from_output(engine_core_output)
     │   └─ → CompletionOutput(token_ids, finish_reason, stop_reason)
     ├─ RequestOutputCollector.put(RequestOutput)
     └─ client: async for output in collector → yield RequestOutput
```

---

## 五、逐文件对比（C032 → C100 → vLLM）

| C100 文件 | C032 状态 | C100 目标 vs vLLM |
|-----------|-----------|--------------------|
| `config/vllm.py` | 不存在 | VllmConfig 聚合 10 config（vLLM: 30 config） |
| `config/model.py` | 不存在 | ModelConfig（vLLM 同名，简化至 ~15 字段） |
| `config/cache.py` | 不存在 | CacheConfig（vLLM 同名，简化至 ~12 字段） |
| `config/scheduler.py` | 不存在 | SchedulerConfig（vLLM 同名，简化至 ~10 字段） |
| `config/parallel.py` | 不存在 | ParallelConfig（vLLM 同名，TP+PP+DP 字段） |
| `config/lora.py` | 不存在 | LoRAConfig（vLLM 同名，~6 字段） |
| `config/speculative.py` | 不存在 | SpeculativeConfig（vLLM 同名，~8 字段） |
| `config/quantization.py` | 不存在 | QuantizationConfig（vLLM 同名，~4 字段） |
| `config/multimodal.py` | 不存在 | MultiModalConfig（vLLM 同名，~5 字段） |
| `config/observability.py` | 不存在 | ObservabilityConfig（vLLM 同名，~4 字段） |
| `v1/request.py` | 11 种状态 | 完全对齐 vLLM `RequestStatus` 全 12 种（含 FINISHED_REPETITION） |
| `v1/outputs.py` | SamplerOutput, ModelRunnerOutput, EngineCoreOutputs | +LogprobsLists, LogprobsTensors, DraftTokenIds（对齐 vLLM） |
| `v1/utils.py` | ConstantList | +CpuGpuBuffer（对齐 vLLM） |
| `v1/serial_utils.py` | JSON stub | MsgpackEncoder/Decoder（对齐 vLLM，省略 OOB tensor IPC） |
| `v1/kv_cache_interface.py` | KVCacheSpec(block_size, dtype) | KVCacheSpecKind + 9 种 spec + KVCacheGroupSpec + KVCacheConfig（对齐 vLLM） |
| `v1/kv_cache_spec_registry.py` | 不存在 | KVCacheSpecRegistry（对齐 vLLM） |
| `v1/forward_context.py` | 不存在 | BatchDescriptor + set/get_forward_context（完全对齐 vLLM） |
| `v1/cudagraph_dispatcher.py` | 不存在 | CudagraphDispatcher（完全对齐 vLLM） |
| `v1/engine/__init__.py` | 空 | EngineCoreRequest(msgspec), FinishReason, EngineCoreEventType, PauseMode（对齐 vLLM） |
| `v1/engine/core.py` | step() 闭环 | +add_request, abort_requests, shutdown, sleep, wake_up, reset_prefix_cache, add_lora/remove_lora, collective_rpc（对齐 vLLM） |
| `v1/engine/core_client.py` | 不存在 | EngineCoreClient(ABC) + InprocClient + SyncMPClient + AsyncMPClient（对齐 vLLM client hierarchy） |
| `v1/engine/llm_engine.py` | 基础 | +from_engine_args, collective_rpc, apply_model, get_metrics（对齐 vLLM） |
| `v1/engine/async_llm.py` | asyncio.Queue 基础 | +output_handler, streaming input, encode(), pause/resume, add_lora/remove_lora（对齐 vLLM） |
| `v1/engine/output_processor.py` | token→text | RequestOutputCollector(async queue+delta streaming), RequestState.make_request_output()（对齐 vLLM） |
| `v1/engine/input_processor.py` | 文本 tokenizer | +MultiModalRegistry 集成, cache_salt, LoRARequest attach（对齐 vLLM 核心） |
| `v1/engine/detokenizer.py` | incremental 基础 | FastIncrementalDetokenizer, SlowIncrementalDetokenizer, check_stop_strings()（对齐 vLLM） |
| `v1/engine/logprobs.py` | 不存在 | LogprobsProcessor — from_new_request, update_from_output, pop_prompt_logprobs（对齐 vLLM） |
| `v1/engine/parallel_sampling.py` | 不存在 | ParentRequest — get_child_info, get_outputs（对齐 vLLM） |
| `v1/engine/exceptions.py` | EngineGenerateError, EngineDeadError | 完全对齐 vLLM |
| `v1/core/block_pool.py` | ref_cnt, LRU, hash cache | +BlockHashToBlockMap, free_blocks, _maybe_evict_cached_block, emit_cached_block_events（对齐 vLLM） |
| `v1/core/kv_cache_manager.py` | single-group | +multi-group allocate_slots, CoW 感知, record_prefix_cache_stats（对齐 vLLM） |
| `v1/core/kv_cache_coordinator.py` | 不存在 | KVCacheCoordinator + HybridKVCacheCoordinator + find_longest_cache_hit（对齐 vLLM） |
| `v1/core/single_type_kv_cache_manager.py` | 不存在 | FullAttentionManager + SlidingWindowManager + CrossAttentionManager + MambaManager + ChunkedLocalAttentionManager（对齐 vLLM 6/9 种，MLA 由 backends/mla/ 管理） |
| `v1/core/kv_cache_utils.py` | KVCacheBlock, FreeKVCacheBlockQueue | +KVCacheBlockCopy, BlockHash, BlockHashWithGroupId（对齐 vLLM） |
| `v1/core/kv_cache_metrics.py` | 不存在 | BlockMetricsState + KVCacheMetricsCollector（对齐 vLLM） |
| `v1/core/encoder_cache_manager.py` | 不存在 | EncoderCacheManager（对齐 vLLM） |
| `v1/core/sched/interface.py` | 不存在 | SchedulerInterface (20 方法，对齐 vLLM) |
| `v1/core/sched/scheduler.py` | FCFS + update_from_output | +Priority/weighted scheduling, preemption(RECOMPUTE/SWAP), get_grammar_bitmask, make_stats, shutdown（对齐 vLLM） |
| `v1/core/sched/async_scheduler.py` | 不存在 | AsyncScheduler（对齐 vLLM） |
| `v1/core/sched/output.py` | 基础字段 | +spec_decode_tokens, num_common_prefix_blocks, preempted_req_ids, GrammarOutput, kv_cache_block_copies（对齐 vLLM） |
| `v1/core/sched/request_queue.py` | FCFSRequestQueue | +PriorityRequestQueue, SchedulingPolicy（对齐 vLLM） |
| `v1/core/sched/utils.py` | 不存在 | check_stop, check_sequence_repetition, remove_all（对齐 vLLM） |
| `v1/worker/worker_base.py` | execute_model | +sample_tokens, compile_or_warm_up_model, add_lora/remove_lora/list_loras（对齐 vLLM） |
| `v1/worker/gpu_worker.py` | execute_model | +execute_model, sample_tokens, init_device, load_model, add_lora, remove_lora, shutdown（对齐 vLLM） |
| `v1/worker/gpu_model_runner.py` | batched prefill+decode | +execute_model, sample_tokens, propose_draft_token_ids, initialize_kv_cache, capture_model（对齐 vLLM） |
| `v1/worker/gpu_input_batch.py` | InputBatch dataclass | CachedRequestState (18+ 字段) + InputBatch (15+ 字段) + InputBuffers（对齐 vLLM） |
| `v1/worker/block_table.py` | Python for loop | +Triton kernel: compute_slot_mappings, gather_block_tables, copy_blocks（对齐 vLLM） |
| `v1/worker/gpu/states.py` | 4 GPU fields | +add_request, remove_request, apply_staged_writes（对齐 vLLM） |
| `v1/worker/gpu/attn_utils.py` | 不存在 | init_attn_backend, init_kv_cache, build_attn_metadata, build_slot_mappings_by_layer（对齐 vLLM） |
| `v1/worker/utils.py` | 不存在 | AttentionGroup, bind_kv_cache（对齐 vLLM） |
| `v1/executor/multiproc_executor.py` | 不存在 | MultiprocExecutor: WorkerProcess lifecycle, shared memory dispatch, failure callback（对齐 vLLM） |
| `v1/attention/backend.py` | 基础 ABC | +AttentionType enum, supported_dtypes, get_kv_cache_shape, get_kv_cache_layout, AttentionMetadataBuilder, CommonAttentionMetadata（对齐 vLLM） |
| `v1/attention/registry.py` | 不存在 | AttentionBackendEnum（3 种 + Mamba 2 种，vLLM 有 30+ 种后端枚举） |
| `v1/attention/backends/triton_attn.py` | 不存在 | TritonAttentionBackend + TritonAttentionImpl（对齐 vLLM） |
| `v1/attention/ops/triton_reshape_and_cache_flash.py` | 不存在 | Triton reshape_and_cache kernel（对齐 vLLM） |
| `v1/attention/ops/triton_decode_attention.py` | 不存在 | Triton decode attention kernel（对齐 vLLM） |
| `v1/attention/ops/triton_prefill_attention.py` | 不存在 | Triton prefill attention kernel（对齐 vLLM） |
| `v1/sample/sampler.py` | greedy+top-k/top-p+penalties | +LogprobsMode, compute_logprobs, gather_logprobs, apply_logits_processors, greedy_sample（静态方法）（对齐 vLLM） |
| `v1/sample/rejection_sampler.py` | 不存在 | RejectionSampler — forward, rejection_sample, parse_output（对齐 vLLM） |
| `v1/sample/metadata.py` | 5 字段 | +max_num_logprobs, generators, allowed_token_ids_mask, bad_words_token_ids, logitsprocs（对齐 vLLM） |
| `v1/sample/logits_processor/` | 3 函数 | +interface.py, builtin.py, state.py（对齐 vLLM 目录结构） |
| `v1/sample/ops/` | 不存在 | penalties.py, bad_words.py, logprobs.py, topk_topp_sampler.py, topk_topp_triton.py（对齐 vLLM） |
| `v1/spec_decode/eagle.py` | 不存在 | EagleProposer（对齐 vLLM） |
| `v1/spec_decode/medusa.py` | 不存在 | MedusaProposer（对齐 vLLM） |
| `v1/spec_decode/ngram_proposer.py` | 不存在 | NgramProposer（对齐 vLLM） |
| `v1/structured_output/` | 不存在 | backend_xgrammar.py, backend_outlines.py, backend_types.py（对齐 vLLM） |
| `v1/pool/` | 不存在 | pooling_runner.py (GPUModelRunner path), late_interaction_runner.py（对齐 vLLM） |
| `v1/kv_offload/` | 不存在 | base.py(OffloadingManager), config.py, cpu/（对齐 vLLM 结构） |
| `v1/metrics/` | collector.py | +stats.py, loggers.py(PrometheusStatLogger), prometheus.py, reader.py（对齐 vLLM） |
| `lora/` | 不存在 | layers.py(BaseLayerWithLoRA), request.py(LoRARequest)（对齐 vLLM） |
| `quantization/` | 不存在 | registry.py(QuantMethod)（对齐 vLLM dispatch，无 kernel） |
| `multimodal/` | 不存在 | registry.py(MultiModalRegistry), vision.py(VisionFeature)（对齐 vLLM） |
| `distributed/` | parallel_state.py(stub) | +all_reduce, all_gather, broadcast 真实实现, communication_op.py（对齐 vLLM 核心） |
| `engine/arg_utils.py` | 不存在 | EngineArgs（对齐 vLLM） |
| `engine/protocol.py` | 不存在 | EngineClient(ABC)（对齐 vLLM） |
| `entrypoints/llm.py` | 不存在 | LLM — generate, chat, encode, collective_rpc, apply_model（对齐 vLLM） |
| `entrypoints/openai/` | 不存在 | api_server.py: /v1/chat/completions, /v1/completions, /v1/embeddings（对齐 vLLM） |

---

## 六、架构差距总结

| 维度 | C032 基线 | C100 目标 | vLLM 参考 | 差距率 |
|------|-----------|-----------|-----------|--------|
| **引擎层** | step() 闭环，直接调用 | EngineClient 三层 (Inproc/SyncMP/AsyncMP) + OutputCollector streaming | 多 client + DP/EP + ZMQ IPCs + profile + sleep/wake + LoRA | 15% |
| **调度** | FCFS + update_from_output | SchedulerInterface ABC, Priority/Preempt/Async, grammar_bitmask | 完整 preempt (recompute/swapped), KV/EC connector, spec decode sched | 20% |
| **KV Cache** | single manager + prefix cache | KVCacheCoordinator (Hybrid), 6 种 SingleTypeManager, CoW, BlockHashToBlockMap | 9 种 manager + KV connector + KV events + metrics collector | 15% |
| **Model Runner** | HF model + 物理 KV 后处理 | ForwardContext injection, AttentionBackend dispatch, Triton PagedAttn, CUDA Graph, Spec Decode | CUDA kernel 全路径, full cudagraph specializations, LoRA mapping, MoE | 20% |
| **Attention** | 1 backend (Torch) + Python paged_attn | 2 backends (Torch + Triton) + registry + 3 Triton kernels | 30+ backends, 19+ Triton kernels, MLA, flashinfer, flex_attention | 40% |
| **Sampling** | greedy + top-k/top-p + 3 penalties | 完整 SamplingMetadata(18字段) + RejectionSampler + logits_processor/ + ops/ 目录 | 完整 penalties Triton kernel, min_tokens, FP64 gumbel, full logprobs | 10% |
| **分布式** | TP stub | TP=2 + PP + DP, all_reduce/all_gather, MultiprocExecutor | Ray/EP/CP/DCP, KV/EC transfer, elastic EP, full device communicator | 30% |
| **LoRA** | 无 | LoRAConfig, LoRALinear, LoRARequest, scheduler LoRA routing, runner LoRA mixin | Full LoRA layers, sharded LoRA, mm LoRA, LoRA ops | 30% |
| **多模态** | 无 | MultiModalRegistry, VisionFeature, EncoderCacheManager, MMInputProcessor | Audio/Video/OCR, 完整 renderer, full vision backends | 40% |
| **量化** | 无 | QuantizationConfig, registry dispatch (FP16/BF16/FP8/INT8/GPTQ/AWQ) | FP8/INT4/NVFP4 CUDA kernels, Marlin, Cutlass, full quant ops | 50% |
| **Spec Decode** | 无 | Eagle/Medusa/NGram proposer + RejectionSampler | +dFlash/Gemma4/suffix decoding/dynamic scheduler | 30% |
| **Structured Output** | 无 | StructuredOutputBackend(ABC) + XGrammar/Outlines backends | +Guidance/LM Format Enforcer | 20% |
| **Metrics** | collector.py | Prometheus: TTFT, TPOT, ITL, throughput, KV utilization | +CachingMetrics, PrefixCacheStats, PerfMetrics, Ray wrappers | 25% |
| **SDK** | run.py CLI | LLM.generate/chat/encode, OpenAI API, CLI serve, EngineArgs | +Anthropic API, gRPC, Launcher, full chat_utils | 25% |
| **Config** | 1 EngineConfig | 10 config 类 (VllmConfig 聚合) | 31 config 类 | 30% |

**平均差距**: ~25%（核心执行链 10-20% / 生态适配 30-50%）

---

## 七、C100 故意保留的差距（C101+ 生态适配）

| 领域 | 具体内容 | 保留原因 |
|------|----------|----------|
| Attention Backend | flashinfer, flex_attention, mamba1/2, linear, cpu, MLA(20 files), ROCm(3), turboquant, diffkv(2) | 硬件/算法适配 |
| Triton Ops (15+) | prefix_prefill, merge_attn_states, triton_fp8_mqa_logits, triton_turboquant_decode/store 等 | CUDA 优化 |
| Model Executor Layers | MLAAttention layer 完整实现 | 编译层 |
| SingleTypeKVCacheManager | RSWA, SinkFull, TQFull, SlidingWindowMLA, HiddenState (5/9 种，C034 已有 3 种 + C068 补齐 1 种 = 6 种已完成) | 模型特有 |
| Distributed | Ray executor, Ray utils, DP/EP/CP/DCP 完整, Elastic EP, device communicator 完整, Weight Transfer | Ray/云平台适配 |
| LoRA | sharded LoRA, mm LoRA, LoRA ops, peft_helper, model_manager | PEFT 集成 |
| Quantization | GPTQ/AWQ/FP8/INT4 NVFP4 完整 kernel 实现 | CUDA kernel |
| MultiModal | Audio/Video/OCR, renderer 完整, vision backends | 平台适配 |
| Engine | DP Coordinator, CoreEngineProcManager, Ray actor, EEP 弹性伸缩 | 云平台 |
| Utils | 33/39 文件 (system_utils, numa, collection, 各种 wrapper) | 工具类 |
| Config | 21/31 配置类 (compilation, device, diffusion, ec_transfer, kernel, kv_events, kv_transfer, load, mamba, model_arch, offload, pooler, profiler, reasoning, structured_output 完整, weight_transfer, speech_to_text) | 生态配置 |
| Tracing/Plugin | OpenTelemetry, plugin system | 运维 |
| Entrypoints | Anthropic API, gRPC, Launcher 完整, Kubernetes | 服务化 |

---

## 八、C100 完成后判断标准

1. **EngineClient 三层**: `InprocClient` → `SyncMPClient` → `AsyncMPClient` 全部工作
2. **SchedulerInterface** 20 方法签名与 vLLM 一致，Scheduler 支持 Priority + Preempt + Async
3. **KVCacheCoordinator**: `find_longest_cache_hit` → 6 种 SingleTypeManager 路由正确
4. **BlockPool**: `get_new_blocks, free_blocks, _maybe_evict_cached_block, free_blocks, BlockHashToBlockMap`
5. **ForwardContext**: `set_forward_context` → `get_forward_context` → `AttentionImpl.forward` 完整注入链
6. **Triton PagedAttention**: `reshape_and_cache, decode_attention, prefill_attention` 三个 kernel 工作
7. **CUDA Graph**: `CudagraphDispatcher` 支持 batch=[1,2,4,8] graph capture/replay
8. **RejectionSampler**: `forward, rejection_sample, parse_output` 与 vLLM 签名一致
9. **Sampler**: `apply_logits_processors, apply_penalties, greedy_sample, compute_logprobs` 完整 pipeline
10. **LogprobsProcessor**: `from_new_request, update_from_output, pop_prompt_logprobs` 工作
11. **IncrementalDetokenizer**: `Fast/Slow + check_stop_strings` 工作
12. **OutputProcessor**: `RequestOutputCollector` async queue + delta streaming 工作
13. **Spec Decode**: Eagle/Medusa/NGram proposer + RejectionSampler 端到端
14. **Structured Output**: StructuredOutputBackend → fill_bitmask() → Sampler bitmask integration 工作
15. **TP=2 + PP=2 + DP=2**: MultiprocExecutor 协同工作
16. **LoRA**: LoRALinear, LoRARequest, scheduler LoRA routing, runner LoRA integration 工作
17. **MultiModal**: MultiModalRegistry → EncoderCacheManager → scheduler → runner 工作
18. **Quantization**: FP16/BF16/FP8/INT8/GPTQ/AWQ dispatch 正确（无 CUDA kernel）
19. **Prometheus**: TTFT, TPOT, ITL, throughput, KV utilization 指标
20. **OpenAI API**: /v1/chat/completions (streaming + logprobs), /v1/completions, /v1/embeddings
21. **LLM SDK**: LLM.generate, chat, encode, collective_rpc, apply_model 工作
22. **每个 Commit 可运行**: `pytest` 全绿

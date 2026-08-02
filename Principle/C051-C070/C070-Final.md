# C070 最终架构 — C051-C070 阶段目标

> C070 每个类名、方法名、dataclass 字段名必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。
> C070 是单 GPU 高级能力的终点。C050 已覆盖核心执行链 100%，C051-C070 补齐 Scheduler 策略、Spec Decode、Structured Output、Pooling、Logprobs、KV Offload 和基础 Config。

## 一、C070 最终文件树

```
my_vllm/
├── __init__.py
│
├── config/                                    # C069 拆分，6 个配置类
│   ├── __init__.py
│   ├── vllm.py                                # VllmConfig (聚合)
│   ├── model.py                               # ModelConfig
│   ├── cache.py                               # CacheConfig
│   ├── scheduler.py                           # SchedulerConfig
│   ├── parallel.py                            # ParallelConfig
│   └── metrics.py                             # MetricsConfig → C080 合并入 observability.py
│
├── v1/
│   ├── __init__.py
│   ├── request.py                             # Request + RequestStatus (12 状态) + SamplingParams
│   ├── outputs.py                             # SamplerOutput, ModelRunnerOutput, DraftTokenIds,
│   │                                          # LogprobsLists, LogprobsTensors
│   ├── utils.py                               # ConstantList
│   ├── serial_utils.py                        # MsgpackEncoder, MsgpackDecoder
│   ├── kv_cache_interface.py                  # KVCacheSpec, FullAttentionSpec,
│   │                                          # SlidingWindowSpec, MLAAttentionSpec,
│   │                                          # KVCacheGroupSpec, KVCacheConfig
│   ├── forward_context.py                     # BatchDescriptor, set/get_forward_context()
│   ├── cudagraph_dispatcher.py                # CudagraphDispatcher
│   │
│   ├── engine/
│   │   ├── __init__.py                        # EngineCoreRequest, EngineCoreOutputs,
│   │   │                                      # FinishReason(IntEnum), PauseMode
│   │   ├── core.py                            # EngineCore — step(), add_request(),
│   │   │                                      # abort_requests(), _process_aborts_queue(),   ← C054b
│   │   │                                      # profile(), shutdown(), sleep(), wake_up(),
│   │   │                                      # reset_prefix_cache()
│   │   ├── core_client.py                     # EngineCoreClient(ABC), InprocClient,
│   │   │                                      # SyncMPClient
│   │   ├── llm_engine.py                      # LLMEngine — add_request(), step(),
│   │   │                                      # get_metrics()
│   │   ├── async_llm.py                       # AsyncLLM — generate(), abort()              ← C053
│   │   ├── output_processor.py                # OutputProcessor, RequestOutputCollector,
│   │   │                                      # RequestState
│   │   ├── input_processor.py                 # InputProcessor.process_inputs()
│   │   ├── detokenizer.py                     # FastIncrementalDetokenizer,
│   │   │                                      # SlowIncrementalDetokenizer,
│   │   │                                      # check_stop_strings()
│   │   ├── exceptions.py                      # EngineGenerateError, EngineDeadError
│   │   ├── logprobs.py                        # LogprobsProcessor                          ← C057
│   │   │                                      #   from_new_request(), update_from_output(),
│   │   │                                      #   pop_prompt_logprobs()
│   │   └── utils.py                           # EngineZmqAddresses
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── kv_cache_coordinator.py             # KVCacheCoordinator, HybridKVCacheCoordinator
│   │   ├── block_pool.py                       # BlockPool + BlockHashToBlockMap
│   │   │                                      #   + free_blocks() ref_cnt cascading (C054c)
│   │   │                                      #   + _maybe_evict_cached_block()
│   │   │                                      #   + free_blocks() ref_cnt 级联释放           ← C054c
│   │   ├── kv_cache_manager.py                 # KVCacheManager, KVCacheBlocks
│   │   │                                      #   + allocate_slots() CoW 感知分配            ← C054c
│   │   │                                      #   + take_kv_cache_block_copies()
│   │   ├── kv_cache_utils.py                   # KVCacheBlock(slots=True),
│   │   │                                      # FreeKVCacheBlockQueue, KVCacheBlockCopy,
│   │   │                                      # BlockHash, BlockHashWithGroupId
│   │   ├── kv_cache_metrics.py                 # BlockMetricsState, KVCacheMetricsCollector ← C054
│   │   ├── prefix_cache.py                     # CoW: shared block → private block
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── interface.py                    # SchedulerInterface (ABC)
│   │       ├── scheduler.py                   # Scheduler — schedule(throttle_prefills),
│   │       │                                  # update_from_output(), get_grammar_bitmask(),
│   │       │                                  # finish_requests(), make_stats(), shutdown()
│   │       │                                  # + Priority/weighted scheduling               ← C051
│   │       │                                  # + Preemption RECOMPUTE/SWAP                  ← C052
│   │       ├── async_scheduler.py             # AsyncScheduler                              ← C053
│   │       ├── request_queue.py               # FCFSRequestQueue, PriorityRequestQueue      ← C051
│   │       └── utils.py                       # check_stop(), remove_all(),                 ← C051
│   │                                          # check_sequence_repetition()
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── worker_base.py                     # WorkerBase (ABC)
│   │   ├── gpu_worker.py                      # Worker — execute_model(), sample_tokens()
│   │   ├── gpu_model_runner.py                # GPUModelRunner — execute_model(),
│   │   │                                      # sample_tokens(), propose_draft_token_ids(),    ← C058
│   │   │                                      # capture_model(), initialize_kv_cache()
│   │   ├── gpu_input_batch.py                 # CachedRequestState, InputBatch, InputBuffers
│   │   ├── block_table.py                     # BlockTables — Triton compute_slot_mappings,
│   │   │                                      # gather_block_tables(), apply_staged_writes()
│   │   ├── utils.py                           # AttentionGroup, bind_kv_cache()
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── states.py                      # RequestState
│   │       ├── attn_utils.py                  # init_attn_backend(), build_attn_metadata()
│   │       ├── spec_decode/                   # Spec decode worker                          ← C058
│   │       │   └── utils.py
│   │       ├── structured_outputs.py          # Grammar bitmask at GPU                       ← C063
│   │       └── model_states/
│   │           ├── __init__.py
│   │           ├── interface.py               # ModelState (ABC)
│   │           └── default.py                 # DefaultModelState
│   │
│   ├── executor/
│   │   ├── __init__.py
│   │   ├── abstract.py                        # Executor (ABC)
│   │   ├── uniproc_executor.py                # UniProcExecutor
│   │   └── multiproc_executor.py              # MultiprocExecutor
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── backend.py                         # AttentionType(DECODER/ENCODER/...),
│   │   │                                      # AttentionBackend(ABC), AttentionImpl(ABC),
│   │   │                                      # AttentionMetadataBuilder(ABC),
│   │   │                                      # CommonAttentionMetadata
│   │   ├── selector.py                        # get_attn_backend()
│   │   ├── registry.py                        # AttentionBackendEnum
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── torch_attn.py                 # TorchAttentionBackend (fallback)
│   │       ├── flash_attn.py                 # FlashAttentionBackend (完整 PyTorch 实现)
│   │       ├── triton_attn.py                # TritonAttentionBackend
│   │       └── ops/
│   │           ├── __init__.py
│   │           ├── paged_attn.py
│   │           ├── triton_reshape_and_cache_flash.py
│   │           ├── triton_decode_attention.py
│   │           ├── triton_prefill_attention.py
│   │           ├── triton_unified_attention.py
│   │           └── chunked_prefill_paged_decode.py
│   │
│   ├── sample/
│   │   ├── __init__.py
│   │   ├── sampler.py                         # Sampler.forward(), greedy_sample(),
│   │   │                                      # compute_logprobs(), gather_logprobs(),
│   │   │                                      # apply_logits_processors(), apply_penalties() ← C056
│   │   ├── rejection_sampler.py               # RejectionSampler.forward(),                  ← C059
│   │   │                                      # rejection_sample(), parse_output()
│   │   ├── metadata.py                        # SamplingMetadata — bad_words_token_ids,      ← C056
│   │   │                                      # allowed_token_ids_mask, logitsprocs,
│   │   │                                      # spec_token_ids, generators
│   │   ├── logits_processor/                  #                                           ← C056
│   │   │   ├── __init__.py
│   │   │   ├── interface.py                   # LogitsProcessor (ABC)
│   │   │   ├── builtin.py                     # MinTokensLogitsProcessor
│   │   │   └── state.py                       # BatchUpdateBuilder
│   │   └── ops/                               #                                           ← C056
│   │       ├── __init__.py
│   │       ├── penalties.py                   # apply_all_penalties (Triton fused)
│   │       ├── bad_words.py                   # apply_bad_words
│   │       ├── logprobs.py                    # batched_count_greater_than
│   │       ├── topk_topp_sampler.py           # TopKTopPSampler
│   │       └── topk_topp_triton.py            # Triton top-k/top-p kernel
│   │
│   ├── spec_decode/                           #                                           ← C058-C062
│   │   ├── __init__.py
│   │   ├── base.py                            # SpecDecodeBaseProposer                       ← C058
│   │   ├── eagle.py                           # EagleProposer
│   │   ├── medusa.py                          # MedusaProposer
│   │   ├── ngram_proposer.py                  # NgramProposer
│   │   └── metadata.py                        # SpecDecodeMetadata
│   │
│   ├── structured_output/                     #                                           ← C063-C065
│   │   ├── __init__.py
│   │   ├── backend_types.py                   # StructuredOutputBackend(ABC), StructuredOutputGrammar
│   │   ├── backend_xgrammar.py                # XgrammarBackend
│   │   ├── backend_outlines.py                # OutlinesBackend
│   │   ├── request.py                         # StructuredOutputRequest
│   │   └── utils.py
│   │
│   ├── pool/                                  #                                           ← C066-C067
│   │   ├── __init__.py
│   │   ├── pooling_runner.py                # GPUModelRunner pooling path (encode)
│   │   ├── late_interaction_runner.py         # LateInteractionRunner
│   │   └── metadata.py                        # PoolingMetadata
│   │
│   ├── kv_offload/                            #                                           ← C055
│   │   ├── __init__.py
│   │   ├── base.py                            # OffloadingManager(ABC), OffloadingWorker
│   │   └── config.py                          # OffloadConfig
│   │
│   └── metrics/
│       ├── __init__.py
│       ├── stats.py                           # SchedulerStats, IterationStats
│       ├── loggers.py                         # PrometheusStatLogger
│       └── prometheus.py                      # setup_multiprocess_prometheus()
│
├── model_executor/
│   ├── __init__.py
│   ├── model_loader.py                        # load_model()
│   ├── models/opt.py
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                          # ColumnParallelLinear, RowParallelLinear
│       └── attention/
│           ├── __init__.py
│           └── attention.py                   # Attention(nn.Module)
│
├── triton_kernels/                            # GPU 计算基础                               ← C050b
│   └── gemm.py                                # Triton tiled GEMM kernel
│
├── distributed/
│   ├── __init__.py
│   ├── parallel_state.py                      # all_reduce(), all_gather(), broadcast()
│   └── tensor_parallel.py                     # TP=2
│
├── entrypoints/
│   └── openai/
│       ├── api_server.py                      # /v1/chat/completions, /v1/completions
│       └── protocol.py
│
├── runtime/                                   # Legacy (test compat)
│   ├── __init__.py
│   ├── model_loader.py, tokenizer.py, model.py
│
├── server/                                    # C014 legacy, deprecated
├── examples/
└── tests/
```

**数量对照**

| 来源 | my_vllm/ .py 文件数 |
|------|---------------------|
| C050 基线 | ~80 |
| C070 目标 | ~110 |
| vLLM V1 `v1/` 目录 | ~220 |

C070 达到 vLLM 的 ~50% 文件覆盖率。C050 已覆盖核心执行链 100%；C051-C070 补齐 Scheduler 进化、Sampling/SpecDecode/StructuredOutput/Pooling/KVOffload/Logprobs/Config 基础。

> **C070 是阶段终点，不是项目终点。** 其后有 C071-C100（分布式 + LoRA + 多模态 + 量化 + SDK）。

---

## 二、组件层次与职责边界（对齐 vLLM 类名/方法名）

```
User / OpenAI API (entrypoints/openai/)
  │
  ├─ EngineClient (InprocClient / SyncMPClient)                  ← C045
  │
  ▼
LLMEngine  ← 对外 API
  │
  ├─ InputProcessor.process_inputs()  — prompt → EngineCoreRequest
  │
  ▼
EngineCore  ← step() / step_with_batch_queue() 循环
  │   + profile() / _process_aborts_queue() / abort_requests() ← C054b
  │
  ├─▶ Scheduler (via SchedulerInterface)
  │   │   schedule(throttle_prefills) → SchedulerOutput
  │   │   update_from_output(scheduler_output, model_runner_output)
  │   │     → dict[int, EngineCoreOutputs]
  │   │   get_grammar_bitmask(scheduler_output) → GrammarOutput           ← C063
  │   │
  │   ├─ SchedulingPolicy (FCFSPolicy / PriorityPolicy)                   ← C051
  │   ├─ Preemption: RECOMPUTE / SWAP                                     ← C052
  │   ├─ AsyncScheduler (background schedule)                             ← C053
  │   ├─ RequestQueue (FCFSRequestQueue / PriorityRequestQueue)           ← C051
  │   │
  │   └─ KVCacheCoordinator
  │       ├─ find_longest_cache_hit(block_hashes, max_cache_hit_length)
  │       └─ SingleTypeKVCacheManager
  │           └─ BlockPool
  │               ├─ free_blocks() (ref_cnt cascading, C054c) + _maybe_evict_cached_block()
  │               ├─ free_blocks() ref_cnt 级联释放                       ← C054c
  │               └─ KVCacheMetricsCollector: on_block_allocated() 等     ← C054
  │
  └─▶ Executor (UniProc / Multiproc)
        └─▶ Worker
              │ execute_model(scheduler_output) → ModelRunnerOutput
              │ sample_tokens(grammar_output) → ModelRunnerOutput
              │
              ├─ RequestState.add_request()
              ├─ BlockTables.compute_slot_mappings()
              ├─ InputBatch + CachedRequestState
              │
              ├─ build_attn_metadata() + set_forward_context()
              ├─ CudagraphDispatcher.dispatch()
              ├─ Model.forward()
              │   └─ Attention.forward():
              │       ├─ do_kv_cache_update(K, V, layer_name)
              │       └─ unified_attention_with_output(Q, K, V, output, layer_name)
              │           └─ self.impl.forward() → FlashAttention / Triton
              │
              ├─ [if spec_decode]: propose_draft_token_ids()               ← C058-C062
              │     └─ EagleProposer / MedusaProposer / NgramProposer
              │
              └─ Sampler.forward(logits, sampling_metadata)
                    ├─ apply_logits_processors()                            ← C056
                    │   └─ bad_words / min_tokens / allowed_tokens / logit_bias
                    ├─ apply_penalties()                                    ← C056
                    ├─ [if grammar]: grammar_bitmask[~allowed] = -inf       ← C063
                    ├─ greedy_sample() / top_k / top_p
                    └─ compute_logprobs()

  Spec Decode 分支:
    RejectionSampler.forward(metadata, draft_probs, logits, sampling_metadata)
      ├─ rejection_sample(draft_token_ids, num_draft_tokens, max_spec_len,
      │      cu_num_draft_tokens, draft_probs, target_logits,
      │      bonus_token_ids, sampling_metadata)
      │   → accepted_mask, bonus_token_id
      └─ parse_output(output_token_ids, vocab_size) → list[list[int]]

  Pooling 分支:
    LLMEngine.encode(prompt)
      ├─ GPUModelRunner pooling path:
      │     model.forward() → last hidden state → pool → embedding
      └─ LateInteractionRunner.run(request)
            → multi-vector [num_tokens, hidden_dim]

  KV Offload 分支:
    BlockPool.evict_blocks()
      ├─ OffloadingManager.prepare_store(block)
      └─ resume: OffloadingManager.prepare_load(block) → GPU

输出侧:
  OutputProcessor.process_outputs(engine_core_outputs)
    ├─ LogprobsProcessor.update_from_output(output)                        ← C057
    │     ├─ _update_sample_logprobs(logprobs_lists)
    │     └─ pop_prompt_logprobs() → PromptLogprobs
    ├─ RequestState.make_request_output(new_token_ids, finish_reason)
    │   ├─ IncrementalDetokenizer.update(new_token_ids, stop_terminated)
    │   │   └─ check_stop_strings()
    │   └─ → CompletionOutput → RequestOutput
    └─ RequestOutputCollector → async streaming → client
```

---

## 三、核心数据结构（C070 新增，字段名/方法签名对齐 vLLM）

### 1. RequestQueue 体系（`v1/core/sched/request_queue.py`）

```python
# C070 目标（对齐 vLLM request_queue.py）

class RequestQueue(ABC):
    @abstractmethod
    def select_request(self, queue: list) → Request | None: ...
    @abstractmethod
    def add_request(self, request: Request) → None: ...

class FCFSRequestQueue(RequestQueue): ...    # vLLM: C032 已有，C070 保持
class PriorityRequestQueue(RequestQueue): ... # vLLM: PriorityRequestQueue
```

### 2. Scheduler 方法扩展（`v1/core/sched/scheduler.py`）

```python
# C070 目标（对齐 vLLM scheduler.py）

class Scheduler(SchedulerInterface):
    def schedule(self, throttle_prefills: bool = False) → SchedulerOutput: ...

    def update_from_output(
        self, scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) → dict[int, EngineCoreOutputs]: ...

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput,
    ) → GrammarOutput | None: ...                                   # ← C063

    def update_draft_token_ids(
        self, draft_token_ids: DraftTokenIds,
    ) → None: ...                                                   # ← C058

    def finish_requests(
        self, request_ids, finished_status,
    ) → list[Request]: ...

    def make_stats(self) → SchedulerStats | None: ...
    def shutdown(self) → None: ...

    # C051-C052 新增逻辑:
    #   schedule 内部调用 self.policy.select_request()
    #   block_pool._maybe_evict_cached_block() 按 priority 选 victim
```

### 3. SchedulerOutput 扩展（`v1/core/sched/output.py`）

```python
# C070 目标（对齐 vLLM scheduled_spec_decode_tokens + GrammarOutput）

@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int

@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]
    scheduled_cached_reqs: CachedRequestData
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int
    scheduled_spec_decode_tokens: dict[str, list[int]]             # ← C058
    num_common_prefix_blocks: list[int]
    finished_req_ids: set[str]
    preempted_req_ids: set[str] | None                              # ← C052
    new_block_ids_to_zero: list[int] | None
    kv_cache_block_copies: list[KVCacheBlockCopy] | None
    has_structured_output_requests: bool | None                     # ← C063

@dataclass
class GrammarOutput:                                                # ← C063
    structured_output_request_ids: list[str]
    grammar_bitmask: npt.NDArray[np.int32]
```

### 4. SchedulerConfig（`config/scheduler.py`）

```python
# C070 目标（对齐 vLLM config/scheduler.py:SchedulerConfig）

@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    max_model_len: int = 4096
    enable_chunked_prefill: bool = True
    policy: str = "fcfs"                                             # ← C051
    async_scheduling: bool = False                                   # ← C053
    stream_interval: int = 1
```

### 5. RejectionSampler（`v1/sample/rejection_sampler.py`）

```python
# C070 目标（对齐 vLLM 方法签名）

class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler, spec_config=None, device=None): ...

    def forward(
        self,
        metadata: SpecDecodeMetadata,        # vLLM: 第一参数
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) → SamplerOutput: ...

    def apply_penalties(
        logits: torch.Tensor, sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata, repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) → torch.Tensor: ...

    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor, vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) → tuple[list[list[int]], LogprobsLists | None]: ...

def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: torch.Tensor,
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    use_fp64_gumbel: bool = False,
) → torch.Tensor: ...

def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
    use_fp64_gumbel: bool = False,
) → torch.Tensor: ...
```

### 6. SamplingMetadata 扩展（`v1/sample/metadata.py`）

```python
# C070 目标（对齐 vLLM 新增字段）

@dataclass
class SamplingMetadata:
    # C050 已有:
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

    # C056 新增:
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: object  # LogitsProcessors

    # C058 新增:
    spec_token_ids: list[list[int]] | None
```

### 7. LogprobsProcessor（`v1/engine/logprobs.py`）

```python
# C070 目标（对齐 vLLM 方法签名）

@dataclass
class LogprobsProcessor:
    tokenizer: object | None
    logprobs: object | None         # SampleLogprobs
    prompt_logprobs: object | None  # PromptLogprobs
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None

    @classmethod
    def from_new_request(
        cls, tokenizer, request: EngineCoreRequest,
    ) → LogprobsProcessor: ...

    def update_from_output(self, output: EngineCoreOutput) → None: ...

    def pop_prompt_logprobs(self) → object | None: ...  # → PromptLogprobs
```

### 8. StructuredOutputBackend（`v1/structured_output/backend_types.py`）

```python
# C070 目标（对齐 vLLM backend_types.py）

class StructuredOutputBackend(ABC):
    @abstractmethod
    def compile_grammar(self, request_type: object, grammar_spec: object) → StructuredOutputGrammar: ...

class StructuredOutputGrammar(ABC):
    @abstractmethod
    def fill_bitmask(
        self, bitmask, batch_index: int,
    ) → None: ...  # → bitmask
```

### 9. KVCacheMetricsCollector（`v1/core/kv_cache_metrics.py`）

```python
# C070 目标（对齐 vLLM 方法签名）

class BlockMetricsState:
    def record_access(self) → None: ...
    def get_lifetime_seconds(self) → float: ...
    def get_idle_time_seconds(self) → float: ...
    def get_reuse_gaps_seconds(self) → list[float]: ...

class KVCacheMetricsCollector:
    def should_sample_block(self, block: KVCacheBlock) → bool: ...
    def on_block_allocated(self, block: KVCacheBlock) → None: ...
    def on_block_accessed(self, block: KVCacheBlock) → None: ...
    def on_block_evicted(self, block: KVCacheBlock) → None: ...
    def drain_events(self) → list[dict]: ...
```

### 10. OffloadingManager（`v1/kv_offload/base.py`）

```python
# C070 目标（对齐 vLLM kv_offload/base.py）

class OffloadingManager(ABC):
    @abstractmethod
    def prepare_load(self, block) → None: ...
    @abstractmethod
    def complete_load(self, block) → None: ...
    @abstractmethod
    def prepare_store(self, block) → None: ...
    @abstractmethod
    def complete_store(self, block) → None: ...
    @abstractmethod
    def touch(self, block) → None: ...


### 11. VllmConfig（`config/vllm.py`）

```python
# C070 目标（对齐 vLLM VllmConfig, 简化至 6 个子 config）

@config
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    scheduler_config: SchedulerConfig
    parallel_config: ParallelConfig
    observability_config: ObservabilityConfig
    instance_id: str = ""
```

---

## 四、核心数据流（对齐 vLLM）

```
┌──────────────────────────────────────────────────────────────────────────┐
│ EngineCore.step() / step_with_batch_queue()                              │
│                                                                          │
│  0. _process_aborts_queue()                                              │ ← C054b
│                                                                          │
│  1. scheduler_output = scheduler.schedule(throttle_prefills)             │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ Scheduler.schedule():                                            │ │
│     │   ├─ SchedulingPolicy.select_request()  — FCFS / Priority        │ ← C051
│     │   ├─ Preemption: _maybe_evict_cached_block() 按 priority 选 victim │ ← C052
│     │   ├─ KVCacheCoordinator.find_longest_cache_hit()                 │ │
│     │   ├─ KVCacheCoordinator.allocate_new_blocks()                    │ │
│     │   │   └─ allocate_slots() CoW 感知 (ref_cnt>1 → KVCacheBlockCopy)│ ← C054c
│     │   ├─ BlockPool.free_blocks()                                      │ │
│     │   ├─ KVCacheMetricsCollector.on_block_allocated()                │ ← C054
│     │   ├─ SpecDecodeBaseProposer  (if spec decode)                    │ ← C058
│     │   ├─ StructuredOutputBackend.compile_grammar()  (if structured output)  │ ← C063
│     │   └─ → SchedulerOutput:                                           │ │
│     │        scheduled_spec_decode_tokens, grammar_output,               │ │
│     │        preempted_req_ids, has_structured_output_requests           │ │
│     └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  2. grammar_output = scheduler.get_grammar_bitmask(scheduler_output)     │ ← C063
│                                                                          │
│  3. model_output = model_executor.execute_model(scheduler_output)        │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ Worker.execute_model() / GPUModelRunner.execute_model():          │ │
│     │                                                                   │ │
│     │   [if spec_decode]: propose_draft_token_ids()                     │ ← C058
│     │     └─ EagleProposer / MedusaProposer / NgramProposer              │ │
│     │                                                                   │ │
│     │   Model.forward()                                                 │ │
│     │   └─ Attention.forward():                                         │ │
│     │       ├─ do_kv_cache_update(K, V, layer_name)                     │ │
│     │       └─ unified_attention_with_output(Q, K, V, output, name)     │ │
│     │           └─ self.impl.forward() → FlashAttention / Triton        │ │
│     │                                                                   │ │
│     │   Sampler.forward(logits, sampling_metadata)                      │ │
│     │     ├─ apply_logits_processors()  — bad_words/min_tokens/allowed  │ ← C056
│     │     ├─ apply_penalties()  — Triton fused                          │ ← C056
│     │     ├─ [if grammar]: grammar_bitmask[~allowed] = -inf             │ ← C063
│     │     ├─ greedy_sample() / top_k / top_p                            │ │
│     │     └─ compute_logprobs() → SamplerOutput                         │ │
│     │                                                                   │ │
│     │   [if spec_decode]: RejectionSampler.forward():                   │ ← C059
│     │     ├─ rejection_sample(draft_token_ids, num_draft_tokens, ...)   │ │
│     │     │   → accepted_mask, bonus_token_id                           │ │
│     │     └─ parse_output(output_token_ids, vocab_size)                 │ │
│     │                                                                   │ │
    │     │   [if kv offload]: OffloadingManager.prepare_store(block)  │ ← C055
│     └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  4. engine_outputs = scheduler.update_from_output(                       │
│       scheduler_output, model_runner_output)                             │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ ├─ check_stop (EOS, max_tokens, check_sequence_repetition)       │ ← C051
│     │ ├─ finish_requests(request_ids, FINISHED_STOPPED/LENGTH)         │ │
│     │ ├─ KVCacheMetricsCollector.on_block_evicted(block)               │ ← C054
│     │ ├─ KVCacheManager.free()  — ref_cnt 级联释放                      │ ← C054c
│     │ └─ → dict[int, EngineCoreOutputs]                                 │ │
│     └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  5. OutputProcessor.process_outputs(engine_core_outputs)                 │
│     ├─ LogprobsProcessor.update_from_output(output)                      │ ← C057
│     │   ├─ _update_sample_logprobs(logprobs_lists)                      │ │
│     │   └─ pop_prompt_logprobs() → PromptLogprobs                       │ │
│     ├─ IncrementalDetokenizer.update(new_token_ids, stop_terminated)    │ │
│     │   └─ check_stop_strings(output_text, new_char_count, ...)         │ │
│     ├─ RequestState.make_request_output() → CompletionOutput            │ │
│     └─ RequestOutputCollector → async streaming → client                │ │
└──────────────────────────────────────────────────────────────────────────┘

Pooling 独立路径:
  LLMEngine.encode(prompt)
    ├─ InputProcessor.process_inputs() → EngineCoreRequest (pooling_params)
    ├─ GPUModelRunner pooling path:
    │     model.forward() → last hidden state → pool → embedding
    └─ or LateInteractionRunner.run(request)
          → per-token embeddings [num_tokens, hidden_dim]
```

---

## 五、逐文件对比（C050 → C070 → vLLM）

| C070 文件 | C050 状态 | C070 vs vLLM |
|-----------|-----------|----------------|
| `triton_kernels/gemm.py` | **不存在** | Triton tiled GEMM kernel（C050b，GPU 计算基础教育性 kernel） |
| `v1/core/sched/request_queue.py` | FCFSRequestQueue | + PriorityRequestQueue（对齐 vLLM） |
| `v1/core/sched/scheduler.py` | schedule/update/finish | + Priority/weighted scheduling + Preemption(RECOMPUTE/SWAP) + get_grammar_bitmask + update_draft_token_ids（对齐 vLLM） |
| `v1/core/sched/async_scheduler.py` | **不存在** | AsyncScheduler（对齐 vLLM） |
| `v1/core/sched/request_queue.py` | FCFSRequestQueue | + PriorityRequestQueue + SchedulingPolicy（对齐 vLLM） |
| `v1/core/sched/utils.py` | **不存在** | check_stop(), remove_all(), check_sequence_repetition()（对齐 vLLM） |
| `v1/core/block_pool.py` | CoW + BlockHashToBlockMap | + free_blocks() ref_cnt 级联释放 + KVCacheMetricsCollector 集成（对齐 vLLM） |
| `v1/core/kv_cache_manager.py` | single-group | + allocate_slots() CoW 感知分配 + take_kv_cache_block_copies()（对齐 vLLM） |
| `v1/core/kv_cache_metrics.py` | **不存在** | BlockMetricsState + KVCacheMetricsCollector（对齐 vLLM） |
| `v1/kv_offload/base.py` | **不存在** | OffloadingManager(ABC) + OffloadingWorker（对齐 vLLM kv_offload/base.py） |
| `v1/engine/core.py` | step/add/shutdown/sleep | + profile() + _process_aborts_queue() + abort_requests()（对齐 vLLM） |
| `v1/engine/logprobs.py` | **不存在** | LogprobsProcessor — from_new_request, update_from_output, pop_prompt_logprobs（对齐 vLLM） |
| `v1/sample/metadata.py` | 13 字段 | + allowed_token_ids_mask, bad_words_token_ids, logitsprocs, spec_token_ids（对齐 vLLM） |
| `v1/sample/sampler.py` | forward/greedy/compute_logprobs | + apply_logits_processors(bad_words/min_tokens/allowed) + apply_penalties(Triton fused)（对齐 vLLM） |
| `v1/sample/logits_processor/` | 不存在 | interface.py(LogitsProcessor ABC), builtin.py(MinTokensLogitsProcessor), state.py(BatchUpdateBuilder)（对齐 vLLM） |
| `v1/sample/ops/penalties.py` | 不存在 | apply_all_penalties（Triton fused, 对齐 vLLM） |
| `v1/sample/ops/bad_words.py` | 不存在 | apply_bad_words（对齐 vLLM） |
| `v1/sample/ops/logprobs.py` | 不存在 | batched_count_greater_than（对齐 vLLM） |
| `v1/sample/ops/topk_topp_sampler.py` | 不存在 | TopKTopPSampler（对齐 vLLM） |
| `v1/sample/ops/topk_topp_triton.py` | 不存在 | Triton top-k/top-p kernel（对齐 vLLM） |
| `v1/sample/rejection_sampler.py` | **不存在** | RejectionSampler — forward(), rejection_sample(), parse_output(), sample_recovered_tokens()（对齐 vLLM） |
| `v1/spec_decode/base.py` | **不存在** | SpecDecodeBaseProposer（对齐 vLLM llm_base_proposer.py） |
| `v1/spec_decode/eagle.py` | **不存在** | EagleProposer（对齐 vLLM） |
| `v1/spec_decode/medusa.py` | **不存在** | MedusaProposer（对齐 vLLM） |
| `v1/spec_decode/ngram_proposer.py` | **不存在** | NgramProposer（对齐 vLLM ngram_proposer.py） |
| `v1/spec_decode/metadata.py` | **不存在** | SpecDecodeMetadata（对齐 vLLM） |
| `v1/structured_output/backend_types.py` | **不存在** | StructuredOutputBackend(ABC) + StructuredOutputGrammar(ABC)（对齐 vLLM） |
| `v1/structured_output/backend_xgrammar.py` | **不存在** | XgrammarBackend（对齐 vLLM） |
| `v1/structured_output/backend_outlines.py` | **不存在** | OutlinesBackend（对齐 vLLM） |
| `v1/structured_output/request.py` | **不存在** | StructuredOutputRequest（对齐 vLLM） |
| `v1/structured_output/utils.py` | **不存在** | (对齐 vLLM) |
| `v1/pool/pooling_runner.py` | **不存在** | GPUModelRunner pooling path（对齐 vLLM，encode 通过 model runner） |
| `v1/pool/late_interaction_runner.py` | **不存在** | LateInteractionRunner（对齐 vLLM late_interaction_runner.py） |
| `v1/pool/metadata.py` | **不存在** | PoolingMetadata（对齐 vLLM pool/metadata.py） |
| `v1/worker/gpu/spec_decode/utils.py` | **不存在** | Spec decode worker 端工具（对齐 vLLM gpu/spec_decode） |
| `v1/worker/gpu/structured_outputs.py` | **不存在** | GPU 端 grammar bitmask 处理（对齐 vLLM） |
| `v1/worker/gpu_model_runner.py` | execute_model/sample | + propose_draft_token_ids()（对齐 vLLM） |
| `config/vllm.py` | **不存在** | VllmConfig（聚合 6 config, 对齐 vLLM） |
| `config/model.py` | **不存在** | ModelConfig（对齐 vLLM, 简化至 ~10 字段） |
| `config/cache.py` | **不存在** | CacheConfig（对齐 vLLM, 简化至 ~10 字段） |
| `config/scheduler.py` | **不存在** | SchedulerConfig（对齐 vLLM, 简化至 ~7 字段） |
| `config/parallel.py` | **不存在** | ParallelConfig（对齐 vLLM, TP+PP 字段） |
| `config/metrics.py` | **不存在** | MetricsConfig（对齐 vLLM observability 简化版） |

---

## 六、架构差距总结（C050 → C070）

| 维度 | C050 → C070 | 对齐 vLLM 关键点 |
|------|-------------|-------------------|
| **调度策略** | FCFS only → SchedulePolicy(ABC) + FCFS/Priority/Weighted | SchedulingPolicy + PriorityRequestQueue |
| **抢占** | 无 → RECOMPUTE + SWAP preempt modes | Preemption in schedule() |
| **异步调度** | 无 → AsyncScheduler + background schedule | async_scheduler.py |
| **Block 生命周期** | CoW 基础 → free_blocks() ref_cnt 级联 + CoW 感知分配 | KVCacheBlockCopy 完整集成 |
| **KV 可观测性** | 无 → BlockMetricsState + KVCacheMetricsCollector | on_block_allocated/accessed/evicted |
| **KV 卸载** | 无 → OffloadingManager(ABC) + cpu backend | kv_offload/base.py |
| **Sampler** | forward/greedy/compute_logprobs → +bad_words/min_tokens/allowed + Triton fused penalties | apply_logits_processors + apply_penalties |
| **Logprobs** | 无 → LogprobsProcessor + prompt/sample/streaming | from_new_request + update_from_output |
| **Spec Decode** | 无 → Eagle/Medusa/NGram proposer + RejectionSampler | rejection_sample() + parse_output() |
| **Structured Output** | 无 → StructuredOutputBackend → fill_bitmask() → Sampler | fill_bitmask() → logits[~mask] = -inf |
| **Pooling** | 无 → GPUModelRunner pooling path + LateInteractionRunner | LLM.encode() |
| **Config** | 1 EngineConfig → VllmConfig(6 子 config) | ModelConfig/CacheConfig/SchedulerConfig/ParallelConfig |
| **Engine 健壮性** | step/shutdown/sleep → +profile/_process_aborts_queue/abort_requests | abort 在 step 之间正确清理 |

---

## 七、C070 故意保留的差距（C071+ 阶段）

C070 之后还有 **C071-C100**（分布式 + 生态 + SDK）：

| 能力 | 目标阶段 |
|------|----------|
| PP/DP/CP + MultiprocExecutor 完善 + Tensor IPC + ZMQ + Collective Ops | C071-C077 |
| LoRA (LoRAConfig, LoRALinear, LoRARequest, Scheduler routing, Runner mixin) | C078-C081 |
| MultiModal (MultiModalRegistry, VisionFeature, EncoderCacheManager, MMInputProcessor) | C085-C088 |
| Quantization dispatch (QuantizationConfig, ModelLoader dispatch, Linear/Attention/KV dtype) | C082-C084 |
| SingleTypeKVCacheManager 其余 3 种 (CrossAttention, Mamba, ChunkedLocal) | → C068 (C071-C100) |
| Config 扩展 (LoRA/Speculative/Quantization/MultiModal/Observability → 10 config 类) | C093-C095 |
| SDK (LLM API, OpenAI 完善, CLI) | C096-C098 |
| Benchmark + 全链路集成 | C099-C100 |
| FlashInfer / flex_attention / MLA(20 files) / ROCm / 其余 14+ Triton ops | C101+ 生态适配 |
| Ray executor / device communicator / Weight Transfer / Elastic EP | C101+ 云平台适配 |
| Utils (33/39 文件) / Tracing / Plugin / 其余 21 Config 类 / Entrypoints 完整 | C101+ 工具/服务 |

---

## 八、C070 完成后判断标准

1. **Priority Scheduling**: FCFSRequestQueue + PriorityRequestQueue 工作
2. **Preemption**: schedule() 内 RECOMPUTE/SWAP 抢占逻辑正确
3. **Async Scheduler**: AsyncScheduler background schedule loop 工作
4. **ref_cnt 级联**: free_blocks() 递归递减 + allocate_slots() CoW 感知分配正确
5. **KV Metrics**: BlockMetricsState + KVCacheMetricsCollector 指标输出
6. **KV Offload**: OffloadingManager.prepare_load/complete_store 工作
7. **Scheduler get_grammar_bitmask**: StructuredOutputBackend → fill_bitmask() → Sampler 通道
8. **Spec Decode**: Eagle/Medusa/NGram proposer → RejectionSampler.forward() → rejection_sample() → parse_output() 端到端
9. **Sampler 完整**: apply_logits_processors(bad_words/min_tokens/allowed) + apply_penalties(Triton fused) 工作
10. **LogprobsProcessor**: from_new_request → update_from_output → pop_prompt_logprobs 工作
11. **Structured Output**: XgrammarBackend + OutlinesBackend → fill_bitmask() → bitmask 采样
12. **Pooling**: GPUModelRunner pooling path + LateInteractionRunner 工作
13. **EngineCore**: profile() + _process_aborts_queue() + abort_requests() 工作
14. **Config**: VllmConfig(6 config) 聚合 + validate 工作
15. **可运行**: 每个 C051-C070 commit 后 `pytest` 全绿

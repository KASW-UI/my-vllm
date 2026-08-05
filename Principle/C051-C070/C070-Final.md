# C070 最终架构 — C051-C070 阶段目标

> C070 每个类名、方法名、dataclass 字段名必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。
> C070 是 Runtime 主干 + 现代模型能力的终点。C050 已覆盖核心执行链 100%，C051-C070 补齐 Scheduler 策略、Spec Decode、Structured Output、Logprobs、FlashAttention kernel、MLA 基础。

## 一、C070 最终文件树

```
my_vllm/
├── __init__.py
│
├── config/                                    # C069 拆分，3 个配置类
│   ├── __init__.py
│   ├── vllm.py                                # VllmConfig（聚合 ModelConfig + CacheConfig + SchedulerConfig 内联）
│   ├── model.py                               # ModelConfig
│   └── cache.py                               # CacheConfig
│
├── forward_context.py                         # BatchDescriptor, set/get_forward_context()  ← C041
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
│   ├── cudagraph_dispatcher.py                # CudagraphDispatcher
│   │
│   ├── engine/
│   │   ├── __init__.py                        # EngineCoreRequest, EngineCoreOutputs,
│   │   │                                      # FinishReason(IntEnum)
│   │   ├── core.py                            # EngineCore — step(), add_request(),
│   │   │                                      # abort_requests(), profile(),
│   │   │                                      # shutdown(), sleep(), wake_up(),
│   │   │                                      # reset_prefix_cache()
│   │   ├── core_client.py                     # EngineCoreClient(ABC), InprocClient,
│   │   │                                      # SyncMPClient
│   │   ├── llm_engine.py                      # LLMEngine — add_request(), step(),
│   │   │                                      # get_metrics()
│   │   ├── async_llm.py                       # AsyncLLM — generate(), abort()
│   │   ├── output_processor.py                # OutputProcessor, RequestOutputCollector,
│   │   │                                      # RequestState
│   │   ├── input_processor.py                 # InputProcessor.process_inputs()
│   │   ├── detokenizer.py                     # FastIncrementalDetokenizer,
│   │   │                                      # SlowIncrementalDetokenizer,
│   │   │                                      # check_stop_strings()
│   │   ├── exceptions.py                      # EngineGenerateError, EngineDeadError
│   │   ├── logprobs.py                        # LogprobsProcessor              ← C057
│   │   │                                      #   from_new_request(), update_from_output(),
│   │   │                                      #   pop_prompt_logprobs()
│   │   └── utils.py                           # EngineZmqAddresses
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── kv_cache_coordinator.py             # KVCacheCoordinator, HybridKVCacheCoordinator
│   │   ├── block_pool.py                       # BlockPool + BlockHashToBlockMap
│   │   │                                      #   + free_blocks() ref_cnt cascading  ← C054c
│   │   │                                      #   + _maybe_evict_cached_block()
│   │   ├── kv_cache_manager.py                 # KVCacheManager, KVCacheBlocks
│   │   │                                      #   + allocate_slots() CoW 感知分配    ← C054c
│   │   │                                      #   + take_kv_cache_block_copies()
│   │   ├── kv_cache_utils.py                   # KVCacheBlock(slots=True),
│   │   │                                      #   FreeKVCacheBlockQueue, KVCacheBlockCopy,
│   │   │                                      #   BlockHash, BlockHashWithGroupId
│   │   ├── kv_cache_metrics.py                 # BlockMetricsState                   ← C054
│   │   ├── single_type_kv_cache_manager.py     # FullAttentionManager,              ← C068
│   │   │                                      #   SlidingWindowManager,
│   │   │                                      #   CrossAttentionManager,
│   │   │                                      #   MambaManager,
│   │   │                                      #   ChunkedLocalAttentionManager
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── interface.py                    # SchedulerInterface (ABC)
│   │       ├── scheduler.py                   # Scheduler — schedule(),
│   │       │                                  #   update_from_output(), get_grammar_bitmask(),
│   │       │                                  #   finish_requests(), make_stats(), shutdown()
│   │       │                                  #   + Priority scheduling              ← C051
│   │       │                                  #   + Preemption RECOMPUTE/SWAP        ← C052
│   │       ├── async_scheduler.py             # AsyncScheduler                      ← C053
│   │       ├── output.py                      # SchedulerOutput, NewRequestData,
│   │       │                                  #   CachedRequestData, GrammarOutput
│   │       ├── request_queue.py               # FCFSRequestQueue, PriorityRequestQueue ← C051
│   │       └── utils.py                       # check_stop(), remove_all(),
│   │                                          #   check_sequence_repetition()
│   │
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── worker_base.py                     # WorkerBase (ABC)
│   │   ├── gpu_worker.py                      # Worker — execute_model(), sample_tokens()
│   │   ├── gpu_model_runner.py                # GPUModelRunner — execute_model(),
│   │   │                                      #   sample_tokens(), propose_draft_token_ids(), ← C058
│   │   │                                      #   capture_model(), initialize_kv_cache()
│   │   ├── gpu_input_batch.py                 # CachedRequestState, InputBatch, InputBuffers
│   │   ├── block_table.py                     # BlockTables — compute_slot_mappings,
│   │   │                                      #   gather_block_tables()
│   │   ├── utils.py                           # AttentionGroup, bind_kv_cache()
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── states.py                      # RequestState
│   │       ├── attn_utils.py                  # init_attn_backend(), build_attn_metadata()
│   │       ├── spec_decode/                   # Spec decode worker                  ← C058
│   │       │   └── utils.py
│   │       ├── structured_outputs.py          # Grammar bitmask at GPU              ← C063
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
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py                    # AttentionBackendEnum
│   │   │   ├── torch_attn.py                 # TorchAttentionBackend (fallback)
│   │   │   ├── flash_attn.py                 # FlashAttentionBackend (PyTorch 实现)
│   │   │   └── triton_attn.py                # TritonAttentionBackend
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── paged_attn.py
│   │       ├── triton_reshape_and_cache_flash.py
│   │       ├── triton_decode_attention.py
│   │       └── triton_prefill_attention.py
│   │
│   ├── sample/
│   │   ├── __init__.py
│   │   ├── sampler.py                         # Sampler — forward(), greedy_sample(),
│   │   │                                      #   compute_logprobs(), gather_logprobs(),
│   │   │                                      #   apply_logits_processors(), apply_penalties() ← C056
│   │   ├── rejection_sampler.py               # RejectionSampler — forward(),         ← C059
│   │   │                                      #   rejection_sample(), parse_output()
│   │   ├── metadata.py                        # SamplingMetadata — bad_words_token_ids, ← C056
│   │   │                                      #   allowed_token_ids_mask, logitsprocs,
│   │   │                                      #   spec_token_ids, generators
│   │   ├── logits_processor/                  #                                    ← C056
│   │   │   ├── __init__.py
│   │   │   ├── interface.py                   # LogitsProcessor (ABC)
│   │   │   ├── builtin.py                     # MinTokensLogitsProcessor
│   │   │   └── state.py                       # BatchUpdateBuilder
│   │   └── ops/                               #                                    ← C056
│   │       ├── __init__.py
│   │       ├── penalties.py                   # apply_all_penalties (Triton fused)
│   │       ├── bad_words.py                   # apply_bad_words
│   │       ├── logprobs.py                    # batched_count_greater_than
│   │       ├── topk_topp_sampler.py           # TopKTopPSampler
│   │       └── topk_topp_triton.py            # Triton top-k/top-p kernel
│   │
│   ├── spec_decode/                           #                                    ← C058-C062
│   │   ├── __init__.py
│   │   ├── eagle.py                           # EagleProposer
│   │   ├── medusa.py                          # MedusaProposer
│   │   ├── ngram_proposer.py                  # NgramProposer
│   │   └── metadata.py                        # SpecDecodeMetadata
│   │
│   ├── structured_output/                     #                                    ← C063-C064
│   │   ├── __init__.py
│   │   ├── backend_types.py                   # StructuredOutputBackend(ABC), StructuredOutputGrammar
│   │   └── backend_xgrammar.py                # XgrammarBackend
│   │
│   ├── pool/                                  #                                    ← C066
│   │   ├── __init__.py
│   │   └── pooling_runner.py                # GPUModelRunner pooling path (encode)
│   │
│   └── metrics/
│       ├── __init__.py
│       └── collector.py                       # SchedulerStats, IterationStats
│
├── model_executor/
│   ├── __init__.py
│   ├── model_loader/                            # load_model()
│   ├── models/
│   │   └── opt.py
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                          # ColumnParallelLinear, RowParallelLinear
│       └── attention/
│           ├── __init__.py
│           └── attention.py                   # Attention(nn.Module)
│
├── triton_kernels/                            # GPU 计算基础
│   ├── gemm.py                                # C050b: Triton tiled GEMM kernel
│   └── flash_attention.py                    # C063b: Triton FlashAttention kernel
│                                              #   (tiled QK^T + causal mask + softmax + PV)
│
├── mla/                                       # C062b: MLA 研究
│   ├── __init__.py
│   └── mla_layer.py                           # MLA decoder layer 简化实现
│                                              #   (based on vllm/v1/attention/backends/mla/)
│
├── distributed/
│   ├── __init__.py
│   ├── parallel_state.py                      # all_reduce(), all_gather(), broadcast()
│   └── communication_op.py                    # collective ops
│
├── runtime/                                   # Legacy (test compat)
│   ├── __init__.py
│   ├── model_loader/, tokenizer.py, model.py
│
├── examples/
└── tests/
```

**数量对照**

| 来源 | my_vllm/ .py 文件数 |
|------|---------------------|
| C050 基线 | ~80 |
| C070 目标 | ~85 |
| vLLM V1 `v1/` 目录 | ~220 |

C070 达到 vLLM V1 文件覆盖率的 ~40%，但 Runtime 主干（Engine + Scheduler + KV Cache + Worker + Attention + Sampler）对齐度 ~85%。

> **C070 之后的 C071-C100 专注于现代模型能力**：MoE、LoRA、量化 kernel (FP8)、MLA kernel、TP 分布式，而非生态适配。

---

## 二、组件层次与职责边界（对齐 vLLM 类名/方法名）

```
EngineClient (InprocClient / SyncMPClient)
  │
  ▼
LLMEngine  ← 对外 API
  │
  ├─ InputProcessor.process_inputs()  — prompt → EngineCoreRequest
  │
  ▼
EngineCore  ← step() / step_with_batch_queue() 循环
  │   + profile() / abort_requests()
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
  │       └─ SingleTypeKVCacheManager × 5                                ← C068
  │           ├─ FullAttentionManager
  │           ├─ SlidingWindowManager
  │           ├─ CrossAttentionManager
  │           ├─ MambaManager
  │           ├─ ChunkedLocalAttentionManager
  │           └─ BlockPool
  │               ├─ free_blocks() (ref_cnt cascading, C054c)
  │               └─ _maybe_evict_cached_block()
  │
  └─▶ Executor (UniProc / Multiproc)
        └─▶ Worker
              │ execute_model(scheduler_output) → ModelRunnerOutput
              │
              ├─ RequestState.add_request()
              ├─ BlockTables.compute_slot_mappings()
              ├─ InputBatch + CachedRequestState
              │
              ├─ build_attn_metadata() + set_forward_context()
              ├─ CudagraphDispatcher.dispatch()
              ├─ Model.forward()
              │   └─ AttentionLayer.forward():
              │       ├─ QKV proj
              │       ├─ get_forward_context()
              │       └─ AttentionImpl.forward() → FlashAttention / Triton
              │
              ├─ [if spec_decode]: propose_draft_token_ids()               ← C058-C062
              │     └─ EagleProposer / MedusaProposer / NgramProposer
              │
              └─ Sampler.forward(logits, sampling_metadata)
                    ├─ apply_logits_processors()                            ← C056
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

  FlashAttention 分支:
    triton_kernels/flash_attention.py                                    ← C063b
      └─ @triton.jit flash_attention_kernel(Q, K, V, mask, sm_scale):
            tile QK^T → causal mask → softmax → tile PV
            输出: attention output [batch, seqlen, head_dim]

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
class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"

class RequestQueue(ABC):
    def add_request(self, request: Request) -> None: ...
    def pop_request(self) -> Request: ...
    def prepend_request(self, request: Request) -> None: ...

class FCFSRequestQueue(deque[Request], RequestQueue): ...
class PriorityRequestQueue(RequestQueue): ...  # heapq-based
```

### 2. SchedulerOutput 扩展（`v1/core/sched/output.py`）

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    lora_request: object | None

@dataclass
class CachedRequestData:
    req_ids: list[str]
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
    @property
    def num_reqs(self) -> int: ...

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

@dataclass
class GrammarOutput:                                                # ← C063
    structured_output_request_ids: list[str]
    grammar_bitmask: npt.NDArray[np.int32]
```

### 3. VllmConfig（`config/vllm.py`）

```python
@dataclass
class VllmConfig:
    model_config: ModelConfig
    cache_config: CacheConfig
    # scheduler config fields inline:
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 8
    enable_chunked_prefill: bool = True
    policy: str = "fcfs"
    # parallel config inline:
    tensor_parallel_size: int = 1
    # instance:
    instance_id: str = ""
```

### 4. RejectionSampler（`v1/sample/rejection_sampler.py`）

```python
class RejectionSampler(nn.Module):
    def __init__(self, sampler: Sampler, spec_config=None, device=None): ...
    def forward(
        self, metadata, draft_probs, logits, sampling_metadata,
    ) -> SamplerOutput: ...

def rejection_sample(
    draft_token_ids, num_draft_tokens, max_spec_len,
    cu_num_draft_tokens, draft_probs, target_logits,
    bonus_token_ids, sampling_metadata, use_fp64_gumbel=False,
) -> torch.Tensor: ...
```

### 5. LogprobsProcessor（`v1/engine/logprobs.py`）

```python
@dataclass
class LogprobsProcessor:
    @classmethod
    def from_new_request(cls, tokenizer, request) -> "LogprobsProcessor": ...
    def update_from_output(self, output: EngineCoreOutput) -> None: ...
    def pop_prompt_logprobs(self) -> object | None: ...
```

### 6. StructuredOutputBackend（`v1/structured_output/backend_types.py`）

```python
class StructuredOutputBackend(ABC):
    def compile_grammar(self, request_type, grammar_spec) -> StructuredOutputGrammar: ...

class StructuredOutputGrammar(ABC):
    def fill_bitmask(self, bitmask, batch_index: int) -> None: ...
```

### 7. BlockMetricsState（`v1/core/kv_cache_metrics.py`）

```python
class BlockMetricsState:
    def record_access(self) -> None: ...
    def get_lifetime_seconds(self) -> float: ...
    def get_idle_time_seconds(self) -> float: ...
    def get_reuse_gaps_seconds(self) -> list[float]: ...
```

### 8. SamplingMetadata 扩展（`v1/sample/metadata.py`）

```python
@dataclass
class SamplingMetadata:
    # C050 已有:
    temperature, all_greedy, all_random, top_p, top_k,
    generators, max_num_logprobs, no_penalties,
    prompt_token_ids, frequency_penalties, presence_penalties,
    repetition_penalties, output_token_ids

    # C056 新增:
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logitsprocs: object

    # C058 新增:
    spec_token_ids: list[list[int]] | None
```

### 9. FlashAttention kernel（`triton_kernels/flash_attention.py`）

```python
@triton.jit
def _flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    Lse_ptr,               # log-sum-exp for backward/recompute
    sm_scale,               # 1/sqrt(head_dim)
    batch: int, heads: int, seqlen: int, head_dim: int,
    BLOCK_M: tl.constexpr,  # Q tile size (along seqlen)
    BLOCK_N: tl.constexpr,  # KV tile size (along seqlen)
):
    """Tiled FlashAttention: O = softmax(Q @ K^T / sqrt(d)) @ V
    完整在线 softmax + causal mask 支持。
    """
```
注意：`flash_attention.py` 放在 `triton_kernels/` 而非 `v1/attention/ops/`，因为它是**教育性 kernel**——注释驱动的完整解释，不是生产级优化实现。

---

## 四、核心数据流（对齐 vLLM）

```
┌──────────────────────────────────────────────────────────────────────────┐
│ EngineCore.step() / step_with_batch_queue()                              │
│                                                                          │
│  1. scheduler_output = scheduler.schedule(throttle_prefills)             │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ Scheduler.schedule():                                            │ │
│     │   ├─ SchedulingPolicy.select_request()  — FCFS / Priority        │ ← C051
│     │   ├─ Preemption: _maybe_evict_cached_block()                     │ ← C052
│     │   ├─ KVCacheCoordinator.find_longest_cache_hit()                 │ │
│     │   ├─ KVCacheCoordinator.allocate_new_blocks()                    │ │
│     │   │   └─ allocate_slots() CoW 感知 (ref_cnt>1 → KVCacheBlockCopy)│ ← C054c
│     │   ├─ SpecDecodeBaseProposer  (if spec decode)                    │ ← C058
│     │   ├─ StructuredOutputBackend.compile_grammar()                    │ ← C063
│     │   └─ → SchedulerOutput                                            │ │
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
│     │   CudagraphDispatcher.dispatch()                                  │ │
│     │   Model.forward()                                                 │ │
│     │   └─ AttentionLayer.forward():                                    │ │
│     │       ├─ QKV proj                                                 │ │
│     │       ├─ get_forward_context()                                    │ │
│     │       └─ AttentionImpl.forward() → FlashAttention / Triton        │ │
│     │           └─ Triton PagedAttention kernel                         │ │
│     │                                                                   │ │
│     │   Sampler.forward(logits, sampling_metadata)                      │ │
│     │     ├─ apply_logits_processors()                                   │ ← C056
│     │     ├─ apply_penalties()  — Triton fused                          │ ← C056
│     │     ├─ [if grammar]: grammar_bitmask[~allowed] = -inf             │ ← C063
│     │     ├─ greedy_sample() / top_k / top_p                            │ │
│     │     └─ compute_logprobs() → SamplerOutput                         │ │
│     │                                                                   │ │
│     │   [if spec_decode]: RejectionSampler.forward()                    │ ← C059
│     │     ├─ rejection_sample(draft_token_ids, num_draft_tokens, ...)   │ │
│     │     │   → accepted_mask, bonus_token_id                           │ │
│     │     └─ parse_output(output_token_ids, vocab_size)                 │ │
│     └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  4. engine_outputs = scheduler.update_from_output(                       │
│       scheduler_output, model_runner_output)                             │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ ├─ check_stop (EOS, max_tokens, check_sequence_repetition)       │ │
│     │ ├─ finish_requests(request_ids, FINISHED_STOPPED/LENGTH)         │ │
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
    └─ GPUModelRunner pooling path:
          model.forward() → last hidden state → pool → embedding

FlashAttention 教育性 kernel:
  triton_kernels/flash_attention.py                                     ← C063b
    └─ @triton.jit _flash_attention_kernel(Q, K, V, mask, sm_scale):
          outer loop: tile Q along seqlen (BLOCK_M)
          inner loop: tile K,V along seqlen (BLOCK_N)
          online softmax: m_ij, l_ij, P_ij, O_i 逐 tile 更新
          输出: attention output + Lse (optional)
```

---

## 五、逐文件对比（C050 → C070 → vLLM）

| C070 文件 | C050 状态 | C070 vs vLLM |
|-----------|-----------|----------------|
| `triton_kernels/gemm.py` | **不存在** | Triton tiled GEMM kernel（C050b，GPU 计算基础教育性 kernel） |
| `triton_kernels/flash_attention.py` | **不存在** | Triton FlashAttention kernel（C063b，完整 tiled QK^T+softmax+PV） |
| `mla/mla_layer.py` | **不存在** | MLA decoder layer 简化实现（C062b，对齐 vLLM `v1/attention/backends/mla/`） |
| `v1/core/sched/request_queue.py` | FCFSRequestQueue | + PriorityRequestQueue（对齐 vLLM） |
| `v1/core/sched/scheduler.py` | schedule/update/finish | + Priority/weighted + Preemption + get_grammar_bitmask + update_draft_token_ids（对齐 vLLM） |
| `v1/core/sched/async_scheduler.py` | **不存在** | AsyncScheduler（对齐 vLLM） |
| `v1/core/sched/utils.py` | **不存在** | check_stop(), remove_all(), check_sequence_repetition()（对齐 vLLM） |
| `v1/core/block_pool.py` | CoW + BlockHashToBlockMap | + free_blocks() ref_cnt 级联释放（对齐 vLLM） |
| `v1/core/kv_cache_manager.py` | single-group | + allocate_slots() CoW 感知分配 + take_kv_cache_block_copies()（对齐 vLLM） |
| `v1/core/kv_cache_metrics.py` | **不存在** | BlockMetricsState（对齐 vLLM） |
| `v1/core/single_type_kv_cache_manager.py` | Full + SlidingWindow | + CrossAttention + Mamba + ChunkedLocalAttention（对齐 vLLM 5/9 种） |
| `v1/engine/core.py` | step/add/shutdown/sleep | + profile() + abort_requests()（对齐 vLLM） |
| `v1/engine/logprobs.py` | **不存在** | LogprobsProcessor（对齐 vLLM） |
| `v1/sample/metadata.py` | 13 字段 | + allowed_token_ids_mask, bad_words_token_ids, logitsprocs, spec_token_ids（对齐 vLLM） |
| `v1/sample/sampler.py` | forward/greedy/compute_logprobs | + apply_logits_processors + apply_penalties（对齐 vLLM） |
| `v1/sample/logits_processor/` | 不存在 | interface.py + builtin.py + state.py（对齐 vLLM 目录结构） |
| `v1/sample/ops/penalties.py` | 不存在 | apply_all_penalties（Triton fused，对齐 vLLM） |
| `v1/sample/ops/bad_words.py` | 不存在 | apply_bad_words（对齐 vLLM） |
| `v1/sample/ops/logprobs.py` | 不存在 | batched_count_greater_than（对齐 vLLM） |
| `v1/sample/ops/topk_topp_sampler.py` | 不存在 | TopKTopPSampler（对齐 vLLM） |
| `v1/sample/ops/topk_topp_triton.py` | 不存在 | Triton top-k/top-p kernel（对齐 vLLM） |
| `v1/sample/rejection_sampler.py` | **不存在** | RejectionSampler（对齐 vLLM） |
| `v1/spec_decode/` | **不存在** | eagle.py + medusa.py + ngram_proposer.py + metadata.py（对齐 vLLM） |
| `v1/structured_output/backend_types.py` | **不存在** | StructuredOutputBackend(ABC) + StructuredOutputGrammar(ABC)（对齐 vLLM） |
| `v1/structured_output/backend_xgrammar.py` | **不存在** | XgrammarBackend（对齐 vLLM） |
| `v1/pool/pooling_runner.py` | **不存在** | GPUModelRunner pooling path（对齐 vLLM） |
| `v1/worker/gpu/spec_decode/utils.py` | **不存在** | Spec decode worker utils（对齐 vLLM） |
| `v1/worker/gpu/structured_outputs.py` | **不存在** | GPU 端 grammar bitmask（对齐 vLLM） |
| `v1/worker/gpu_model_runner.py` | execute_model/sample | + propose_draft_token_ids()（对齐 vLLM） |
| `config/vllm.py` | **不存在** | VllmConfig（聚合 3 config，对齐 vLLM） |
| `config/model.py` | **不存在** | ModelConfig（对齐 vLLM，简化至 ~10 字段） |
| `config/cache.py` | **不存在** | CacheConfig（对齐 vLLM，简化至 ~10 字段） |

---

## 六、架构差距总结（C050 → C070）

| 维度 | C050 → C070 | 对齐 vLLM 关键点 |
|------|-------------|-------------------|
| **调度策略** | FCFS only → SchedulingPolicy + FCFS/Priority | SchedulingPolicy + PriorityRequestQueue |
| **抢占** | 无 → RECOMPUTE + SWAP preempt | Preemption in schedule() |
| **异步调度** | 无 → AsyncScheduler background loop | async_scheduler.py |
| **Block 生命周期** | CoW 基础 → free_blocks() ref_cnt 级联 + CoW 感知分配 | KVCacheBlockCopy 完整集成 |
| **KV 可观测性** | 无 → BlockMetricsState | on_block_allocated/accessed/evicted |
| **Sampler** | forward/greedy/compute_logprobs → +bad_words/min_tokens/allowed + Triton fused penalties | apply_logits_processors + apply_penalties |
| **Logprobs** | 无 → LogprobsProcessor + prompt/sample | from_new_request + update_from_output |
| **Spec Decode** | 无 → Eagle/Medusa/NGram proposer + RejectionSampler | rejection_sample() + parse_output() |
| **Structured Output** | 无 → StructuredOutputBackend → fill_bitmask() | fill_bitmask() → logits[~mask] = -inf |
| **Pooling** | 无 → GPUModelRunner pooling path | LLM.encode() |
| **Config** | 1 EngineConfig → VllmConfig(3 config) | ModelConfig + CacheConfig + 内联字段 |
| **Engine 健壮性** | step/shutdown/sleep → +profile + abort_requests | abort 在 step 之间正确清理 |
| **GPU 计算基础** | 无 → Triton GEMM + FlashAttention kernel | tile QK^T + online softmax + causal mask |
| **MLA 基础** | 无 → MLA decoder layer 简化实现 | 对齐 vLLM `v1/attention/backends/mla/` |
| **SingleTypeKVCacheManager** | 2 种 → 5 种（+Cross/Mamba/ChunkedLocal） | 对齐 vLLM 5/9 种 |

---

## 七、C070 故意保留的差距（C071+ 阶段）

C070 之后 C071-C100 聚焦**分布式 Runtime + 现代模型能力 + 集群调度**：

| 能力 | 目标阶段 | 深度 |
|------|----------|------|
| **TP (Tensor Parallel)** — NCCL all_reduce/all_gather/broadcast | C071-C100 | ⭐4 完整实现 |
| **PP (Pipeline Parallel)** — Micro-batch + 1F1B + bubble analysis | C071-C100 | ⭐4 完整 |
| **EP (Expert Parallel)** — All-to-All dispatch/combine | C071-C100 | ⭐4 完整 |
| **CP (Context Parallel)** — Ring Attention + LSE aggregation | C071-C100 | ⭐4 完整 |
| **Multi-GPU Integration** — TP×PP×EP×CP topology | C071-C100 | ⭐4 分析 |
| **MoE** — router + fused expert kernel (Triton) + load balancing | C071-C100 | ⭐4 完整 |
| **LoRA** — LoRALinear + LoRARequest + GPU runner mixin | C071-C100 | ⭐4 完整 |
| **FP8 量化** — FP8 dispatch + 简化逐通道量化 kernel | C071-C100 | ⭐4 含 kernel |
| **MLA 深度分析** — absorb + memory + RoPE verification | C071-C100 | ⭐3 分析 |
| **Ray** — actor lifecycle + PlacementGroup + RayExecutor | C071-C100 | ⭐3 理解级 |
| **K8s** — Deployment + HPA + readiness probe | C071-C100 | ⭐3 理解级 |
| **Autoscaling** — queue_depth 决策逻辑 + 冷启动 | C071-C100 | ⭐3 理解级 |
| **SDK** — LLM.generate() / encode() 基本 API | C071-C100 | ⭐3 基础 |

以下不再纳入 C071-C100 范围（C101+ 生态适配）：

| 领域 | 保留原因 |
|------|----------|
| 多模态完整（Audio/Video/OCR/Renderer） | Runtime 基本不变，理解一次抽象即可 |
| 更多 Attention Backend（flashinfer、ROCm、cpu 等） | Backend 抽象已掌握，增加只是"再写一个 Backend" |
| gRPC / Anthropic API / 完整 Launcher | 平台工程，非 AI Infra 核心 |
| 完整 Config 体系（31 类） | 产品化配置管理，4 个足以表达核心参数 |
| 完整 Prometheus / Tracing | 运维工具，非推理引擎核心 |
| DP / 弹性 EP / Weight Transfer | TP+PP+EP+CP 已覆盖核心并行通信模式 |

---

## 八、C070 完成后判断标准

1. **Priority Scheduling**: FCFSRequestQueue + PriorityRequestQueue 工作
2. **Preemption**: schedule() 内 RECOMPUTE/SWAP 抢占逻辑正确
3. **Async Scheduler**: AsyncScheduler background schedule loop 工作
4. **ref_cnt 级联**: free_blocks() 递归递减 + allocate_slots() CoW 感知分配正确
5. **KV Metrics**: BlockMetricsState 指标正确
6. **Sampler 完整**: apply_logits_processors + apply_penalties(Triton fused) 工作
7. **LogprobsProcessor**: from_new_request → update_from_output → pop_prompt_logprobs 工作
8. **Spec Decode**: Eagle/Medusa/NGram proposer → RejectionSampler.rejection_sample() → parse_output() 端到端
9. **Structured Output**: XgrammarBackend → fill_bitmask() → bitmask 采样
10. **Pooling**: GPUModelRunner pooling path 工作
11. **EngineCore**: profile() + abort_requests() 工作
12. **Config**: VllmConfig(3 config) 聚合 + validate 工作
13. **FlashAttention kernel**: Triton @triton.jit tiled QK^T + online softmax + causal mask + PV，$O(n^2d)$ 算法到 $O(nd)$ 内存的正确性验证
14. **MLA layer**: 基于 vLLM `v1/attention/backends/mla/` 源码分析的简化 MLA decoder layer，理解 KV compression 机制
15. **SingleTypeKVCacheManager 5 种**: CrossAttention + Mamba + ChunkedLocalAttention 新增正确
16. **可运行**: 每个 C051-C070 commit 后 `pytest` 全绿

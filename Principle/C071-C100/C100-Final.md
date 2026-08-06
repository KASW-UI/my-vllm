# C100 最终架构 — Runtime 主干 + 分布式 Runtime + 现代模型能力 + 集群调度

> C100 每个类名、方法名、dataclass 字段必须能在 `/home/kasw/文档/GitHub/GPU-Project/vllm` 中找到对应。
> C100 聚焦 ⭐5 Runtime 主干 + ⭐4 分布式 Runtime（TP/PP/EP/CP）+ ⭐4 现代模型能力 + ⭐3 集群调度理解。

## 一、C100 最终文件树

```
my_vllm/
├── __init__.py
│
├── config/                                    # C069 拆分，4 个配置类
│   ├── __init__.py
│   ├── vllm.py                                # VllmConfig（聚合 ModelConfig + CacheConfig + ParallelConfig + 内联字段）
│   ├── model.py                               # ModelConfig
│   ├── cache.py                               # CacheConfig
│   └── parallel.py                            # ParallelConfig (tp_size, pp_size, ep_size, cp_size) ← C086
│
├── forward_context.py                         # BatchDescriptor, set_forward_context, ← C041
│                                              # get_forward_context
│
├── v1/                                        # V1 Core Architecture
│   ├── __init__.py
│   ├── request.py                             # Request + RequestStatus + SamplingParams
│   ├── outputs.py                             # SamplerOutput, ModelRunnerOutput, LogprobsLists,
│   │                                          # LogprobsTensors, DraftTokenIds
│   ├── utils.py                               # ConstantList, CpuGpuBuffer
│   ├── serial_utils.py                        # MsgpackEncoder, MsgpackDecoder
│   ├── kv_cache_interface.py                  # KVCacheSpec, FullAttentionSpec,
│   │                                          # SlidingWindowSpec, MLAAttentionSpec,
│   │                                          # MambaSpec, ChunkedLocalAttentionSpec,
│   │                                          # CrossAttentionSpec, KVCacheGroupSpec, KVCacheConfig
│   ├── kv_cache_spec_registry.py              # KVCacheSpecRegistry
│   ├── cudagraph_dispatcher.py                # CudagraphDispatcher
│   │
│   ├── engine/                                # Engine layer
│   │   ├── __init__.py                        # EngineCoreRequest, EngineCoreOutput,
│   │   │                                      # EngineCoreOutputs, FinishReason
│   │   ├── core.py                            # EngineCore — step(), add_request(),
│   │   │                                      # abort_requests(), shutdown(), sleep(), wake_up(),
│   │   │                                      # reset_prefix_cache(), add_lora(), remove_lora(),
│   │   │                                      # collective_rpc()
│   │   ├── core_client.py                     # EngineCoreClient(ABC), InprocClient,
│   │   │                                      # SyncMPClient
│   │   ├── llm_engine.py                      # LLMEngine — add_request(), step(),
│   │   │                                      # get_metrics(), collective_rpc(), apply_model()
│   │   ├── async_llm.py                       # AsyncLLM — generate(), encode(),
│   │   │                                      # abort(), add_request()
│   │   ├── output_processor.py                # OutputProcessor, RequestOutputCollector,
│   │   │                                      # RequestState
│   │   ├── input_processor.py                 # InputProcessor — process_inputs()
│   │   ├── detokenizer.py                     # IncrementalDetokenizer,
│   │   │                                      # FastIncrementalDetokenizer,
│   │   │                                      # SlowIncrementalDetokenizer,
│   │   │                                      # check_stop_strings()
│   │   ├── exceptions.py                      # EngineGenerateError, EngineDeadError
│   │   ├── logprobs.py                        # LogprobsProcessor
│   │   └── utils.py                           # EngineZmqAddresses
│   │
│   ├── core/                                  # Scheduler & KV Cache core
│   │   ├── __init__.py
│   │   ├── block_pool.py                      # KVCacheBlock, BlockPool,
│   │   │                                      # BlockHashToBlockMap
│   │   ├── kv_cache_manager.py                # KVCacheManager, KVCacheBlocks
│   │   ├── kv_cache_coordinator.py            # KVCacheCoordinator,
│   │   │                                      # HybridKVCacheCoordinator
│   │   ├── single_type_kv_cache_manager.py     # FullAttentionManager,
│   │   │                                      # SlidingWindowManager,
│   │   │                                      # CrossAttentionManager,
│   │   │                                      # MambaManager,
│   │   │                                      # ChunkedLocalAttentionManager
│   │   ├── kv_cache_utils.py                  # KVCacheBlock, FreeKVCacheBlockQueue,
│   │   │                                      # KVCacheBlockCopy, BlockHash,
│   │   │                                      # BlockHashWithGroupId
│   │   ├── kv_cache_metrics.py                # BlockMetricsState
│   │   └── sched/
│   │       ├── __init__.py
│   │       ├── interface.py                   # SchedulerInterface (ABC)
│   │       ├── scheduler.py                   # Scheduler — schedule(),
│   │       │                                  # update_from_output(), get_grammar_bitmask(),
│   │       │                                  # finish_requests(), make_stats(), shutdown()
│   │       ├── async_scheduler.py             # AsyncScheduler
│   │       ├── output.py                      # SchedulerOutput, NewRequestData,
│   │       │                                  # CachedRequestData, GrammarOutput
│   │       ├── request_queue.py               # FCFSRequestQueue, PriorityRequestQueue,
│   │       │                                  # SchedulingPolicy
│   │       └── utils.py                       # check_stop(), remove_all(),
│   │                                          # check_sequence_repetition()
│   │
│   ├── worker/                                # Model execution
│   │   ├── __init__.py
│   │   ├── worker_base.py                     # WorkerBase (ABC) — execute_model(),
│   │   │                                      # sample_tokens(), compile_or_warm_up_model(),
│   │   │                                      # add_lora(), remove_lora()
│   │   ├── gpu_worker.py                      # Worker — init_device(), load_model(),
│   │   │                                      # execute_model(), sample_tokens(),
│   │   │                                      # add_lora(), remove_lora(), shutdown()
│   │   │                                      # + PP send/recv activation handling        ← C084
│   │   ├── gpu_model_runner.py                # GPUModelRunner — execute_model(),
│   │   │                                      # sample_tokens(), propose_draft_token_ids(),
│   │   │                                      # initialize_kv_cache(), capture_model()
│   │   │                                      # + PP stage-aware layer execution           ← C084
│   │   │                                      # + CP ring attention dispatch               ← C085
│   │   ├── gpu_input_batch.py                 # CachedRequestState, InputBatch, InputBuffers
│   │   ├── block_table.py                     # BlockTables — append_block_ids(),
│   │   │                                      # gather_block_tables(), compute_slot_mappings()
│   │   ├── lora_model_runner_mixin.py         # LoRA forward integration
│   │   ├── utils.py                           # AttentionGroup, bind_kv_cache()
│   │   └── gpu/
│   │       ├── __init__.py
│   │       ├── input_batch.py                 # InputBatch, CachedRequestState
│   │       ├── block_table.py                 # BlockTables (Triton 化)
│   │       ├── states.py                      # RequestState — add_request(),
│   │       │                                  # remove_request(), apply_staged_writes()
│   │       ├── attn_utils.py                  # init_attn_backend(), init_kv_cache(),
│   │       │                                  # build_attn_metadata()
│   │       ├── pp_utils.py                    # PPGroup, MicroBatcher                  ← C084
│   │       │                                  # Pipeline parallel: micro-batch + 1F1B
│   │       ├── model_states/
│   │       │   ├── __init__.py
│   │       │   ├── interface.py               # ModelState (ABC)
│   │       │   └── default.py                 # DefaultModelState
│   │       ├── lora_utils.py                  # LoRA LRU cache
│   │       ├── spec_decode/                   # Spec decode worker
│   │       │   └── utils.py
│   │       └── structured_outputs.py          # Grammar bitmask at GPU
│   │
│   ├── executor/                              # Execution backend
│   │   ├── __init__.py
│   │   ├── abstract.py                        # Executor (ABC) — execute_model(),
│   │   │                                      # initialize_cache()
│   │   ├── uniproc_executor.py                # UniProcExecutor
│   │   ├── multiproc_executor.py              # MultiprocExecutor
│   │   │                                      # + TP×PP×EP×CP worker layout           ← C086
│   │   └── ray_utils.py                       # RayPlacementGroup, RayActorManager    ← C091
│   │
│   ├── attention/                             # Attention backends
│   │   ├── __init__.py
│   │   ├── backend.py                         # AttentionType (DECODER/ENCODER/...),
│   │   │                                      # AttentionBackend (ABC),
│   │   │                                      # AttentionImpl (ABC), AttentionImplBase,
│   │   │                                      # AttentionMetadataBuilder (ABC),
│   │   │                                      # CommonAttentionMetadata
│   │   ├── selector.py                        # get_attn_backend()
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── registry.py                    # AttentionBackendEnum
│   │   │   ├── torch_attn.py                 # TorchAttentionBackend (SDPA fallback)
│   │   │   ├── flash_attn.py                 # FlashAttentionBackend (PyTorch)
│   │   │   └── triton_attn.py                # TritonAttentionBackend
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── paged_attn.py                 # Python 参考实现
│   │       ├── triton_reshape_and_cache_flash.py
│   │       ├── triton_decode_attention.py
│   │       ├── triton_prefill_attention.py
│   │       └── ring_attention.py             # Ring Attention kernel (Triton)          ← C085
│   │
│   ├── sample/                                # Sampling pipeline
│   │   ├── __init__.py
│   │   ├── sampler.py                         # Sampler — forward(), sample(),
│   │   │                                      # compute_logprobs(), gather_logprobs(),
│   │   │                                      # greedy_sample(), apply_temperature(),
│   │   │                                      # apply_penalties(), apply_logits_processors()
│   │   ├── rejection_sampler.py               # RejectionSampler — forward(),
│   │   │                                      # rejection_sample(), parse_output()
│   │   ├── metadata.py                        # SamplingMetadata
│   │   ├── logits_processor/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py                   # LogitsProcessor ABC
│   │   │   ├── builtin.py                     # MinTokensLogitsProcessor
│   │   │   └── state.py                       # BatchUpdateBuilder
│   │   └── ops/
│   │       ├── __init__.py
│   │       ├── penalties.py                   # apply_all_penalties (Triton)
│   │       ├── bad_words.py                   # apply_bad_words
│   │       ├── logprobs.py                    # batched_count_greater_than
│   │       ├── topk_topp_sampler.py           # TopKTopPSampler
│   │       └── topk_topp_triton.py            # Triton top-k/top-p kernel
│   │
│   ├── spec_decode/                           # Speculative decoding
│   │   ├── __init__.py
│   │   ├── eagle.py                           # EagleProposer
│   │   ├── medusa.py                          # MedusaProposer
│   │   ├── ngram_proposer.py                  # NgramProposer
│   │   └── metadata.py                        # SpecDecodeMetadata
│   │
│   ├── structured_output/                     # Structured output
│   │   ├── __init__.py
│   │   ├── backend_types.py                   # StructuredOutputBackend (ABC)
│   │   └── backend_xgrammar.py                # XgrammarBackend
│   │
│   ├── pool/                                  # Pooling
│   │   ├── __init__.py
│   │   └── pooling_runner.py                # GPUModelRunner pooling path (encode)
│   │
│   └── metrics/
│       ├── __init__.py
│       └── collector.py                       # SchedulerStats, IterationStats
│
├── model_executor/                            # Model loading bridge
│   ├── __init__.py
│   ├── model_loader/                        # load_model()
│   ├── models/
│   │   └── opt.py
│   └── layers/
│       ├── __init__.py
│       ├── linear.py                          # ColumnParallelLinear, RowParallelLinear
│       ├── fused_moe/                         # MoE layer                                ← C081
│       │   ├── __init__.py
│       │   ├── fused_moe.py                   # fused_moe kernel (Triton)
│       │   ├── layer.py                       # MoE layer (router + experts)
│       │   │                                   # + EP dispatch/combine integration        ← C082
│       │   └── router/                         # TopK routing + load balancing
│       ├── quantization/                      # Quantization
│       │   ├── __init__.py
│       │   ├── base_config.py                  # QuantizationConfig
│       │   ├── fp8.py                         # FP8 dispatch + simplified kernel
│       │   └── auto_gptq.py                    # GPTQ dispatch (stub)
│       └── attention/
│           ├── __init__.py
│           └── attention.py                   # Attention(nn.Module)
│
├── triton_kernels/                            # GPU 计算基础
│   ├── gemm.py                                # Triton tiled GEMM kernel
│   └── flash_attention.py                    # Triton FlashAttention kernel
│
├── mla/                                       # MLA 研究
│   ├── __init__.py
│   ├── mla_layer.py                           # MLA decoder layer 简化实现
│   └── analysis.md                            # MLA 计算分析与优化路径
│
├── lora/                                      # LoRA
│   ├── __init__.py
│   ├── layers.py                              # BaseLayerWithLoRA,
│   │                                          # ColumnParallelLinearWithLoRA
│   └── request.py                             # LoRARequest
│
├── multimodal/                                # MultiModal registry only
│   ├── __init__.py
│   └── registry.py                            # MultiModalRegistry
│
├── distributed/                               # Distributed Runtime
│   ├── __init__.py
│   ├── parallel_state.py                      # TP/PP/EP/CP group management,
│   │                                          # all_reduce(), all_gather(), broadcast()
│   │                                          # GroupCoordinator, init_tp/pp/ep/cp_group()
│   ├── communication_op.py                    # collective ops wrapper
│   ├── elastic_ep/                             # EP: All-to-All dispatch/combine       ← C082
│   │   ├── __init__.py
│   │   └── dispatch.py                         # EP dispatch/combine utilities
│
├── deploy/                                    # Cluster Deployment
│   ├── autoscaling.py                         # AutoscalingController                   ← C093
│   └── k8s/
│       ├── deployment.yaml                    # K8s Deployment spec                     ← C092
│       ├── service.yaml                       # K8s Service spec
│       └── hpa.yaml                           # K8s HPA spec
│
├── engine/                                    # Engine public layer
│   └── arg_utils.py                           # EngineArgs
│
├── entrypoints/                               # Entry points
│   └── llm.py                                 # LLM — generate(), encode()
│
├── runtime/                                   # Legacy runtime (test compat)
│   ├── __init__.py
│   ├── model_loader/, tokenizer.py, model.py
│
└── tests/
```

**数量对照**

| 来源 | my_vllm/ .py 文件数 |
|------|---------------------|
| C032 基线 | ~51 |
| C070 目标 | ~85 |
| C100 目标 | ~90 |
| vLLM V1 `v1/` 目录 | ~220 |

C100 以更少的文件数覆盖更高密度的「理解」模块——分布式 Runtime（TP/PP/EP/CP 完整实现）+ 集群调度理解（Ray/K8s/Autoscaling）+ 现代模型能力（MoE/LoRA/FP8/MLA）。

---

## 二、组件层次与职责边界

### C100 目标（Runtime 主干 + 分布式 Runtime + 现代模型能力 + 集群调度）

```
┌─ 集群调度层 ─────────────────────────────────────────────────────────────┐
│                                                                          │
│  AutoscalingController ← C093                                           │
│    │ decide: queue_depth > threshold → scale_up / scale_down             │
│    │                                                                     │
│    ├─→ Ray (C091): RayActorManager + PlacementGroup                     │
│    │     RayWorker × N: each a GPU process with model loaded             │
│    │                                                                     │
│    └─→ K8s (C092): Deployment + HPA (queue_depth metric)                │
│          Pod × N: each with GPU resource limits + readiness probe        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─ API / SDK Layer ────────────────────────────────────────────────────────┐
│  entrypoints/llm.py:LLM.generate(prompt) / encode(text)                  │
│  engine/arg_utils.py:EngineArgs → VllmConfig                             │
└──────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─ Engine Layer ───────────────────────────────────────────────────────────┐
│  engine/core_client.py:EngineClient (ABC)                                │
│    ├── InprocClient — 同进程                                              │
│    └── SyncMPClient — ZMQ REP/REQ                                        │
│                                                                          │
│  v1/engine/async_llm.py:AsyncLLM                                         │
│    ├── generate() → AsyncGenerator[RequestOutput]                        │
│    └── encode() → AsyncGenerator[PoolingRequestOutput]                   │
└──────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─ EngineCore (v1/engine/core.py) ─────────────────────────────────────────┐
│  step() → step_with_batch_queue()                                        │
│    │                                                                     │
│    ├─▶ Scheduler (v1/core/sched/scheduler.py)                            │
│    │   │   schedule(throttle_prefills) → SchedulerOutput                 │
│    │   │   update_from_output(scheduler_output, model_runner_output)     │
│    │   │   SchedulingPolicy (FCFS / Priority) + Preemption               │
│    │   │                                                                 │
│    │   └─ KVCacheCoordinator (v1/core/kv_cache_coordinator.py)           │
│    │       ├─ find_longest_cache_hit(block_hashes)                       │
│    │       ├─ 5 种 SingleTypeKVCacheManager                              │
│    │       └─ BlockPool (CoW + BlockHashToBlockMap)                      │
│    │                                                                     │
│    └─▶ Executor (v1/executor/multiproc_executor.py)                      │
│          │                                                               │
│          │  ┌─ 分布式 Runtime ─────────────────────────────────┐        │
│          │  │                                                  │        │
│          ├── TP (C083): ColumnParallelLinear + RowParallelLinear│        │
│          │     all_reduce / all_gather across GPUs              │        │
│          │                                                     │        │
│          ├── PP (C084): Micro-batch pipeline                   │        │
│          │     layer_i on GPU_0 → send → GPU_1 recv → layer_j  │        │
│          │     1F1B-lite inference schedule                    │        │
│          │                                                     │        │
│          ├── EP (C082): Expert Placement across GPUs           │        │
│          │     All-to-All token dispatch/combine               │        │
│          │                                                     │        │
│          ├── CP (C085): Ring Attention for long context        │        │
│          │     seqlen split → ring send/recv KV tiles          │        │
│          │                                                     │        │
│          └── Multi-GPU (C086): TP×PP×EP×CP topology            │        │
│          └──────────────────────────────────────────────────────┘        │
│                                                                          │
│          └─▶ Worker (v1/worker/gpu_worker.py)                            │
│                │ execute_model(SchedulerOutput) → ModelRunnerOutput      │
│                │                                                         │
│                ├─ RequestState.add_request()                             │
│                ├─ BlockTables.compute_slot_mappings()                    │
│                ├─ InputBatch + CachedRequestState                        │
│                │                                                         │
│                ├─ CudagraphDispatcher.dispatch()                         │
│                ├─ Model.forward()                                        │
│                │   ├─ AttentionLayer.forward()                           │
│                │   │   ├─ QKV proj (ColumnParallel across TP group)      │
│                │   │   ├─ [if CP]: ring_attention_kernel()               │
│                │   │   └─ AttentionImpl.forward() → Triton PagedAttn     │
│                │   │                                                     │
│                │   └─ MoELayer.forward()                                 │
│                │       ├─ Router: expert logits                          │
│                │       ├─ [if EP]: all_to_all_dispatch()                 │
│                │       ├─ fused_moe_kernel: parallel expert compute      │
│                │       └─ [if EP]: all_to_all_combine()                  │
│                │                                                         │
│                └─ Sampler.forward(logits, sampling_metadata)             │
│                      ├─ apply_logits_processors()                        │
│                      ├─ apply_penalties()                                │
│                      ├─ greedy_sample() / top_k / top_p                  │
│                      └─ compute_logprobs()                               │
└──────────────────────────────────────────────────────────────────────────┘

输出侧:
  OutputProcessor.process_outputs(engine_core_outputs)
    ├─ LogprobsProcessor.update_from_output()
    ├─ IncrementalDetokenizer.update()
    └─ RequestOutputCollector → async streaming → client
```

---

## 三、核心数据结构（与 vLLM 对齐）

### 1. ParallelConfig（`config/parallel.py`）

```python
@dataclass
class ParallelConfig:
    """TP/PP/EP/CP 配置。"""
    tensor_parallel_size: int = 1       # TP: 模型权重切分
    pipeline_parallel_size: int = 1     # PP: 模型层切分
    expert_parallel_size: int = 1       # EP: expert 分布
    context_parallel_size: int = 1      # CP: 序列切分
    world_size: int = 1                 # 总 GPU 数
```

### 2. GroupCoordinator（`distributed/parallel_state.py`）

```python
class GroupCoordinator:
    """管理一个并行 group 的通信。
    
    每个并行维度 (TP/PP/EP/CP) 有独立的 GroupCoordinator。
    """
    def __init__(self, group, device, world_size, rank):
        self.group = group              # torch.distributed.ProcessGroup
        self.world_size = world_size
        self.rank = rank

    def all_reduce(self, tensor): ...
    def all_gather(self, tensor, dim=-1): ...
    def broadcast(self, tensor, src=0): ...
    def send(self, tensor, dst): ...
    def recv(self, tensor, src): ...

# 全局 group accessors
_TP: GroupCoordinator | None = None
_PP: GroupCoordinator | None = None
_EP: GroupCoordinator | None = None
_CP: GroupCoordinator | None = None

def get_tp_group() -> GroupCoordinator: ...
def get_pp_group() -> GroupCoordinator: ...
def get_ep_group() -> GroupCoordinator: ...
def get_cp_group() -> GroupCoordinator: ...
```

### 3. EP All-to-All（`distributed/elastic_ep/`）

```python
def expert_parallel_dispatch(
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor,
    router_weights: torch.Tensor,
    num_local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-to-All dispatch tokens to expert GPUs。
    返回: (dispatched_tokens, sorted_indices, dispatch_mask)
    """

def expert_parallel_combine(
    expert_output: torch.Tensor,
    sorted_indices: torch.Tensor,
) -> torch.Tensor:
    """All-to-All combine expert outputs back to original GPUs。"""
```

### 4. PP Group（`v1/worker/gpu/pp_utils.py`）

```python
class PPGroup:
    def __init__(self, pp_size, pp_rank, layers_per_stage): ...
    def send_forward(self, tensor, dst): ...
    def recv_forward(self, src): ...

class MicroBatcher:
    def __init__(self, batch_size, num_micro_batches): ...
    def split_batch(self, input_batch): ...
    def get_schedule(self): ...
```

### 5. CP Ring Attention（`v1/attention/ops/ring_attention.py`）

```python
class CPGroup:
    def __init__(self, cp_size, cp_rank): ...
    def ring_step(self, K, V, K_recv_buf, V_recv_buf): ...

@triton.jit
def ring_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, Lse_ptr,
    seqlen_local: int, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
): ...

def ring_attention_forward(Q, K, V, cp_group: CPGroup) -> torch.Tensor:
    """Ring Attention 前向。cp_size 步 ring 通信。"""
```

### 6. AutoscalingController（`deploy/autoscaling.py`）

```python
@dataclass
class ScalingDecision:
    action: str          # "scale_up" | "scale_down" | "noop"
    target_replicas: int
    reason: str

class AutoscalingController:
    def __init__(self, config: AutoscalingConfig): ...
    def evaluate(self, metrics) -> ScalingDecision: ...
```

### 7. Ray Deployment（`v1/executor/ray_utils.py`）

```python
@ray.remote(num_gpus=1)
class RayWorker:
    def __init__(self, vllm_config, rank, world_size): ...
    def execute_model(self, scheduler_output): ...

class RayPlacementGroup:
    """STRICT_SPREAD / STRICT_PACK / PACK。"""

class RayExecutor(Executor):
    def __init__(self, vllm_config, placement_group): ...
    def execute_model(self, scheduler_output): ...
```

### 8. 通信开销模型（`C086 分析`）

```
通信开销模型 (per token per layer):

TP: all_reduce(local_output)
    量: O(hidden_dim)
    频: 每层
    总: O(num_layers × hidden_dim)

PP: send(activation) + recv(activation)
    量: O(hidden_dim × seqlen)
    频: per micro-batch
    总: O(num_micro_batches × hidden_dim × seqlen)

EP: all_to_all(tokens)
    量: O(top_k × hidden_dim × seqlen)
    频: per MoE layer
    总: O(num_moe_layers × k × hidden_dim × seqlen)

CP: ring send(K_tile) + ring recv(K_tile)
    量: O(num_heads × head_dim × seqlen/cp_size) per step
    频: cp_size steps per attention
    总: O(cp_size × num_heads × head_dim × seqlen/cp_size)
      = O(num_heads × head_dim × seqlen) = 与序列长度线性

兼容矩阵:
  TP × PP: ✅  每个 PP stage 内部做 TP
  TP × EP: ✅  每个 EP rank 内部做 TP
  TP × CP: ✅  每个 CP rank 内部做 TP
  PP × CP: ⚠️  需要跨 stage 同步 ring state
  EP × CP: ⚠️  All-to-All 与 ring 通信重叠复杂
```

---

## 四、核心数据流（C100 与 vLLM 对齐）

```
┌─────────────────────────────────────────────────────────────────────────┐
│ EngineCore.step() / step_with_batch_queue()                             │
│                                                                         │
│  1. scheduler_output = scheduler.schedule(throttle_prefills)            │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ Scheduler: Priority/Preempt/Async                            │     │
│     │ KVCacheCoordinator: find_longest_cache_hit + allocate        │     │
│     │ Spec Decode proposer + Grammar compilation (if applicable)   │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  2. model_output = model_executor.execute_model(scheduler_output)       │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ MultiprocExecutor: dispatches to TP×PP×EP×CP workers        │     │
│     │                                                             │     │
│     │ Worker.execute_model(scheduler_output):                      │     │
│     │                                                              │     │
│     │   ┌─ PP pipeline ─────────────────────────────────────────┐ │     │
│     │   │ MicroBatcher.split_batch → N micro-batches              │ │     │
│     │   │ for each micro-batch:                                   │ │     │
│     │   │   if pp_rank > 0: recv_forward() from prev stage       │ │     │
│     │   │                                                        │ │     │
│     │   │   ┌─ TP (inside PP stage) ──────────────────────────┐  │ │     │
│     │   │   │ Model.forward():                                  │  │ │     │
│     │   │   │                                                  │  │ │     │
│     │   │   │   ┌─ CP (if cp_size > 1) ───────────────────┐   │  │ │     │
│     │   │   │   │ ring_attention_forward(Q, K, V, cp_group)│   │  │ │     │
│     │   │   │   │   for step in range(cp_size):            │   │  │ │     │
│     │   │   │   │     O_local = flash_attn(Q, K, V)        │   │  │ │     │
│     │   │   │   │     send/recv K,V to/from next rank      │   │  │ │     │
│     │   │   │   │     O = merge(O, O_local, LSE)           │   │  │ │     │
│     │   │   │   └──────────────────────────────────────────┘   │  │ │     │
│     │   │   │                                                  │  │ │     │
│     │   │   │   Attention: QKV proj (ColumnParallel)           │  │ │     │
│     │   │   │     → AttentionImpl.forward()                    │  │ │     │
│     │   │   │     → all_reduce(RowParallelLinear output)       │  │ │     │
│     │   │   │                                                  │  │ │     │
│     │   │   │   ┌─ EP (if ep_size > 1) ───────────────────┐   │  │ │     │
│     │   │   │   │ MoELayer.forward():                       │   │  │ │     │
│     │   │   │   │   1. Router: compute expert logits        │   │  │ │     │
│     │   │   │   │   2. all_to_all_dispatch(tokens)          │   │  │ │     │
│     │   │   │   │   3. fused_moe_kernel(local experts)      │   │  │ │     │
│     │   │   │   │   4. all_to_all_combine(outputs)          │   │  │ │     │
│     │   │   │   └──────────────────────────────────────────┘   │  │ │     │
│     │   │   │                                                  │  │ │     │
│     │   │   │   [if LoRA]: inject adapter weights              │  │ │     │
│     │   │   │   [if FP8]: dynamic input quantization           │  │ │     │
│     │   │   └──────────────────────────────────────────────────┘  │ │     │
│     │   │                                                        │ │     │
│     │   │   if pp_rank < pp_size - 1: send_forward() to next    │ │     │
│     │   │   stage                                                │ │     │
│     │   └────────────────────────────────────────────────────────┘ │     │
│     │                                                              │     │
│     │   Sampler.forward(logits, sampling_metadata)                │     │
│     │   [if spec_decode]: RejectionSampler.forward()              │     │
│     └──────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  3. engine_outputs = scheduler.update_from_output(scheduler_output,     │
│       model_runner_output)                                              │
│     ┌─────────────────────────────────────────────────────────────┐     │
│     │ check_stop → finish_requests → KVCacheManager.free()        │     │
│     │ → dict[int, EngineCoreOutputs]                              │     │
│     └─────────────────────────────────────────────────────────────┘     │
│                                                                         │
│  4. OutputProcessor.process_outputs(engine_core_outputs)                │
│     ├─ LogprobsProcessor.update_from_output()                           │
│     ├─ IncrementalDetokenizer.update() → check_stop_strings()           │
│     └─ RequestOutputCollector → streaming → client                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 五、逐文件对比（C070 → C100 → vLLM）

| C100 文件 | C070 状态 | C100 vs vLLM |
|-----------|-----------|----------------|
| `config/parallel.py` | **不存在** | ParallelConfig (tp/pp/ep/cp_size，对齐 vLLM) |
| `distributed/parallel_state.py` | TP stub | + GroupCoordinator + init_tp/pp/ep/cp_group (对齐 vLLM) |
| `distributed/communication_op.py` | 不存在 | collective ops wrapper (对齐 vLLM) |
| `distributed/elastic_ep/` | **不存在** | EP All-to-All dispatch/combine (对齐 vLLM `fused_moe/layer.py` all-to-all) |
| `v1/executor/ray_utils.py` | **不存在** | RayPlacementGroup + RayWorker + RayExecutor (对齐 vLLM `engine/utils.py` + `executor/ray_executor.py`) |
| `model_executor/layers/fused_moe/` | **不存在** | fused_moe kernel + MoELayer + routing (对齐 vLLM `layers/fused_moe/`) |
| `v1/attention/ops/ring_attention.py` | **不存在** | Ring Attention kernel + CPGroup (对齐 vLLM `attention/ops/common.py`) |
| `v1/worker/gpu/pp_utils.py` | **不存在** | PPGroup + MicroBatcher (对齐 vLLM `worker/gpu/pp_utils.py`) |
| `v1/worker/gpu_worker.py` | execute_model | + PP send/recv handling (对齐 vLLM) |
| `v1/worker/gpu_model_runner.py` | execute_model/sample | + PP stage-aware + CP ring dispatch (对齐 vLLM) |
| `model_executor/layers/quantization/fp8.py` | **不存在** | FP8 逐通道量化 kernel (对齐 vLLM) |
| `model_executor/layers/quantization/base_config.py` | **不存在** | QuantizationConfig (对齐 vLLM) |
| `lora/layers/` | **不存在** | BaseLayerWithLoRA + ColumnParallelLinearWithLoRA (对齐 vLLM) |
| `lora/request.py` | **不存在** | LoRARequest (对齐 vLLM) |
| `multimodal/registry.py` | **不存在** | MultiModalRegistry (对齐 vLLM) |
| `deploy/autoscaling.py` | **不存在** | AutoscalingController + ScalingDecision |
| `deploy/k8s/deployment.yaml` | **不存在** | K8s Deployment + Service + HPA spec |
| `engine/arg_utils.py` | **不存在** | EngineArgs (对齐 vLLM) |
| `entrypoints/llm.py` | **不存在** | LLM.generate() / encode() (对齐 vLLM) |

---

## 六、架构差距总结

| 维度 | C070 基线 | C100 目标 | vLLM 参考 | 差距率 |
|------|-----------|-----------|-----------|--------|
| **引擎层** | step() 闭环 | EngineClient 两层 + OutputCollector | 多 client + DP/EP + sleep/wake | 20% |
| **调度** | FCFS + update | Priority/Preempt/Async + grammar_bitmask | KV/EC connector, spec decode sched | 20% |
| **KV Cache** | single manager | KVCacheCoordinator, 5 SingleTypeManager, CoW | 9 种 manager + KV events | 15% |
| **Model Runner** | HF model | ForwardContext, Triton PagedAttn, CUDA Graph, MoE, LoRA, PP/CP | CUDA kernel 全路径, full cudagraph | 20% |
| **Attention** | 1 backend | 3 backends + registry + 3 Triton kernels + Ring Attention | 30+ backends, flashinfer, flex_attention | 30% |
| **Sampling** | greedy + top-k | 完整 SamplingMetadata + RejectionSampler | FP64 gumbel | 10% |
| **Spec Decode** | 无 | Eagle/Medusa/NGram + RejectionSampler | dFlash/Gemma4 | 25% |
| **Structured Output** | 无 | XGrammar | Outlines/Guidance | 15% |
| **TP** | stub | TP=2 real NCCL (all_reduce/all_gather) | TP=8, full device communicator | 25% |
| **PP** | 无 | Micro-batch + 1F1B + bubble analysis | PP=8, full pipeline schedule | 30% |
| **EP** | 无 | All-to-All dispatch/combine + TP composition | MoE DP, full EP infrastructure | 35% |
| **CP** | 无 | Ring Attention + online LSE merge | USP, full CP integration | 30% |
| **Config** | 3 config 类 | 4 config 类 (VllmConfig + Model + Cache + Parallel) | 31 config 类 | 15% |
| **LoRA** | 无 | LoRALinear + Runner mixin | Full LoRA, sharded, mm LoRA | 30% |
| **MoE** | 无 | Router + TopK + fused kernel + EP | Full MoE infrastructure (30 files) | 35% |
| **量化** | 无 | FP8 dispatch + kernel + GPTQ stub | FP8/INT4/NVFP4, Marlin, Cutlass | 40% |
| **MLA** | 无 | MLA decoder layer + 分析 | MLA 20 files | 55% |
| **集群调度** | 无 | Ray (actor lifecycle) + K8s (deployment/HPA) + Autoscaling | Full Ray/K8s operator | 40% |
| **SDK** | run.py | LLM.generate() / encode() | OpenAI API, gRPC, Anthropic API, CLI | 25% |

**平均差距**: ~25%（Runtime 主干 10-15% / 分布式 Runtime 25-35% / 现代模型能力 25-35% / 集群调度 40%）

---

## 七、C100 故意保留的差距（C101+ 生态适配）

| 领域 | 具体内容 | 保留原因 |
|------|----------|----------|
| Attention Backend (额外 27+) | flashinfer, flex_attention, ROCm(3), cpu, turboquant 等 | Backend 抽象已掌握 |
| Triton Ops (额外 15+) | prefix_prefill, merge_attn_states, fp8_mqa_logits 等 | CUDA 优化 |
| TP > 2 | TP=4, TP=8 full scaling | TP=2 已覆盖核心通信模式 |
| PP > 2 | PP=4, PP=8 full pipeline schedule | PP=2 已覆盖 micro-batch + send/recv 机制 |
| EP 完整 (30 files) | MoE DP, full EP dispatch, KV/EC transfer | EP=2 已覆盖 All-to-All 核心语义 |
| CP 完整 | USP, full CP integration, All-to-All variant | Ring Attention 已覆盖序列并行的核心算法 |
| 分布式完整 | Ray executor 完整, device communicator, weight transfer, elastic EP | 资源管理 + 进程调度, 平台工程 |
| Quantization 完整 | AWQ/INT4/NVFP4 CUDA kernel, Marlin, Cutlass | FP8 已覆盖校准路径 |
| MultiModal 完整 | Audio/Video/OCR, full renderer, vision backends | Runtime 不变 |
| K8s 完整 | K8s Operator, Custom Metrics API, Prometheus Adapter | 已理解 deployment + HPA 模式 |
| 其余 Config (28 类) | compilation, device, diffusion 等 | 产品化配置管理 |

---

## 八、C100 完成后判断标准

1. **EngineClient 两层**: `InprocClient` → `SyncMPClient` 工作
2. **SchedulerInterface**: Priority + Preempt + Async + grammar_bitmask 完整
3. **KVCacheCoordinator**: 5 种 SingleTypeManager + CoW 完整
4. **ForwardContext**: 完整 `set_forward_context → get_forward_context → AttentionImpl.forward` 注入链
5. **Triton PagedAttention**: 3 个 kernel 工作 + Ring Attention kernel 工作
6. **CUDA Graph**: batch=[1,2,4,8] graph capture/replay
7. **Spec Decode**: Eagle/Medusa/NGram + RejectionSampler 端到端
8. **Structured Output**: XGrammar fill_bitmask → Sampler 集成
9. **Pooling**: encode 工作
10. **Config**: VllmConfig(4 config) 聚合 + validate
11. **TP=2 real NCCL**: all_reduce/all_gather/broadcast，与单 GPU 输出一致
12. **PP=2**: micro-batch pipeline 端到端，bubble 计算正确
13. **EP=2**: All-to-All dispatch/combine，expert load 分布正确
14. **CP=2**: Ring Attention 端到端，长序列与单 GPU 输出一致
15. **Multi-GPU (C086)**: TP×PP 2×2 组合验证通过
16. **LoRA**: LoRALinear + multi-adapter LRU + runner integration
17. **MoE**: Router + TopK + fused kernel + EP 端到端
18. **FP8 Quantization**: FP8 kernel 数值误差 < 1e-2 vs FP16
19. **MLA**: decoder layer 实现 + absorb/decoupled RoPE 验证
20. **FlashAttention kernel**: online softmax 数值验证 < 1e-3
21. **Ray**: RayExecutor 替换 MultiprocExecutor，TP=2 运行正常
22. **K8s**: deployment.yaml + hpa.yaml 语义正确，readiness probe 等待模型加载
23. **Autoscaling**: AutoscalingController 正确评估 scale_up/scale_down 条件
24. **LLM SDK**: LLM.generate() / encode() 工作
25. **每个 Commit 可运行**: `pytest` 全绿

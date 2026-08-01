# After C032 — my-vllm 与 vLLM V1 完整差距清单

my-vllm `v1/` 目录 ~50 个 .py 文件，参考 vllm `v1/` 目录 ~220 个 .py 文件。
以下按目录列出所有缺失模块和功能。

---

## 一、`v1/engine/` — 引擎层

| 缺失模块 | 职责 |
|----------|------|
| `coordinator.py` | DP 多卡协调器（ZMQ 进程收集多 DP rank 统计、发布负载均衡状态） |
| `core_client.py` | 引擎客户端抽象层：`EngineCoreClient`(ABC)、`InprocClient`、`SyncMPClient`、`AsyncMPClient`、`DPAsyncMPClient`、`DPLBAsyncMPClient` |
| `parallel_sampling.py` | `ParentRequest` 类处理 n>1 并行采样扇入/扇出，含子请求追踪、聚合输出、采样参数克隆 |
| `tensor_ipc.py` | `TensorIpcSender` / `TensorIpcReceiver` — 跨进程零拷贝 Tensor 传输（torch.multiprocessing.Queue） |
| `utils.py` | `CoreEngineProcManager`（spawn/local 引擎进程生命周期）、`CoreEngineActorManager`（Ray actor 引擎管理 + placement groups）、`SignalCallback`、`EngineZmqAddresses`、`launch_core_engines()`、`wait_for_engine_startup()` |
| `logprobs.py` | `LogprobsProcessor` — 增量 logprobs 处理（prompt logprobs + sample logprobs），合并逐步结果 |
| `__init__.py` 内容 | `EngineCoreRequest`（msgspec.Struct，20+字段：prompt_token_ids, mm_features, sampling_params, lora, priority, data_parallel_rank, trace_headers, cache_salt 等）、`EngineCoreOutput`、`EngineCoreOutputs`、`EngineCoreRequestType`、`UtilityOutput`、`FinishReason` IntEnum (STOP/LENGTH/ABORT/ERROR/REPETITION)、`EngineCoreEvent`/`EngineCoreEventType`、`PauseMode` |
| `core.py` 缺失功能 | ZMQ 输入输出队列、batch queue（pipeline parallelism）、KV cache 初始化/内存 profiling（`profile_run()`）、structured output manager、KV connector / EC connector 集成、sleep/wake_up（GPU offloading levels 0-2）、pause/resume 调度器、LoRA add/remove/list/pin、prefix cache reset、encoder cache reset、MMP receiver cache、collective RPC、tensor IPC receiver、DP 引擎协调、shutdown 协议、graceful drain、signal handling、engine core monitor thread |
| `async_llm.py` 缺失功能 | background output_handler（asyncio task）、streaming input（async generator）、`encode()`（pooling）、`pause_generation()` / `resume_generation()`、`scale_elastic_ep()`、`wait_for_requests_to_drain()`、LoRA 异步方法、weight transfer（init/start/update/finish）、Prometheus shutdown、stat logger manager、profiler、renderer |
| `llm_engine.py` 缺失功能 | `from_engine_args()` / `from_vllm_config()`、renderer 集成、stat logging、DP group 初始化、`apply_model()`、`collective_rpc()`、bytecode hook 清理、LoRA 管理、encoder cache reset、sleep/wake_up、prefix cache reset、`get_metrics()` |
| `input_processor.py` 缺失功能 | 多模态输入（`MultiModalFeatureSpec`、`MultiModalBudget`）、encoder-decoder 输入分割、`InputPreprocessor` 集成、renderer 集成、cache salt、LoRA request attachment、`assign_request_id()`（外部 vs 内部 ID 映射）、`data_parallel_rank` 路由、speculative config 字段 |
| `output_processor.py` 缺失功能 | `RequestOutputCollector`（async queue + delta streaming 聚合）、stop-string 检测、logprobs 处理、trace header 传播、`PoolingRequestOutput`、parallel sampling 子请求聚合、iteration stats 收集、多模态 cache 统计、engine core timestamp 追踪 |
| `detokenizer.py` 缺失功能 | `FastIncrementalDetokenizer`（tokenizers DecodeStream，>=0.22.0）+ `SlowIncrementalDetokenizer`，stop-string 检测、leading space skipping |

---

## 二、`v1/core/` — 调度器 & KV Cache 核心

| 缺失模块 | 职责 |
|----------|------|
| `encoder_cache_manager.py` | `EncoderCacheManager` + `EncoderDecoderCacheManager` — 多模态 encoder 输出缓存（vision embeddings），分配时触发 eviction，按 hash 跨请求共享 |
| `kv_cache_coordinator.py` | `KVCacheCoordinator` (ABC) + `HybridKVCacheCoordinator` (903行) — 协调多种 attention type（FullAttention / SlidingWindow / Mamba / MLA / CrossAttention 等）的 KV cache block 分配，处理不同 block sizes、细粒度 hash lookup、prefix cache retention intervals、按 spec 管理器委派 |
| `kv_cache_metrics.py` | `BlockMetricsState`（block 生命周期、访问历史、重用间隔追踪）+ `KVCacheMetricsCollector`（抽样 KV cache residency 指标） |
| `single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager` (ABC, 1882行) + 10+ 具体管理器：`FullAttentionManager`、`SlidingWindowManager`、`MambaManager`、`CrossAttentionManager`、`MLAAttentionManager`、`ChunkedLocalAttentionManager`、`SlidingWindowMLAManager`、`HiddenStateCacheManager`、`RSWAManager`、`SinkFullAttentionManager`、`TQFullAttentionManager` |
| `sched/interface.py` | `SchedulerInterface` (ABC, 254行) — 完整调度器契约：`schedule(throttle_prefills)`、`get_grammar_bitmask()`、`update_from_output()`、`update_draft_token_ids()`、`finish_requests()`、`get_num_unfinished_requests()`、`has_requests()`、`pause_state` property、`set_pause_state()`、`reset_prefix_cache()`、`reset_encoder_cache()`、`get_request_counts()`、`make_stats()`、`shutdown()`、`get_kv_connector()`、`get_ec_connector()` |
| `sched/async_scheduler.py` | 异步调度器（speculative decoding 支持） |
| `sched/utils.py` | `check_stop()`、`remove_all()` |
| `sched/scheduler.py` 缺失功能 | priority 调度、preemption（recompute/swapped）、KV connector / EC connector 集成、structured output grammar 编译与 bitmask、encoder cache 查找、multimodal budget 管理、draft token 追踪、DP prefill 节流、streaming input 更新、LoRA 调度、KV event 发布、routed expert 管理 |
| `sched/output.py` 缺失功能 | `GrammarOutput`、`ScheduledEncoderInputStats` |
| `block_pool.py` 缺失功能 | `BlockHashToBlockMap`（多 block hash 碰撞处理）、CoW（Copy-on-Write）通过 `_maybe_evict_cow_blocks()`、`gc_blocks()` 垃圾回收、`KVConnector` 集成、`KVCacheMetricsCollector` 集成 |
| `kv_cache_manager.py` 缺失功能 | 多 group block 分配、sliding window 支持、CoW 感知分配、`get_num_common_prefix_blocks()` |
| `kv_cache_utils.py` 缺失内容 | `KVCacheBlockCopy` dataclass、`BlockHash` / `BlockHashWithGroupId` 类型系统、`generate_block_hash_extra_keys()`、`get_request_block_hasher()`、`init_none_hash()`、`resolve_kv_cache_block_sizes()`、`get_kv_cache_configs()`、`generate_scheduler_kv_cache_config()`、`get_kv_cache_capacity()` |

---

## 三、`v1/worker/` — 模型执行器

### 缺失文件（20+ 个）

| 模块 | 职责 |
|------|------|
| `gpu_input_batch.py` | `CachedRequestState`（35+字段）+ `InputBatch` — 多 group block tables、logprobs 累积、pooling states、LoRA requests、multimodal features、spec decode metadata、XDRoPE/MRoPE positions、reasoning config、slot mapping |
| `block_table.py` | Triton 核函数替代 Python 循环：`MultiGroupBlockTable`、`SlotMappingMode` 枚举、Triton `copy_blocks()`、CP interleave 支持、`FusedStagedWriter` |
| `gpu/states.py` | `StagedWriteTensor` + `UvaBackedTensor`（UVA 大张量节省 GPU 内存）、`num_computed_prefill_tokens`、`draft_tokens`、CPU mirror arrays |
| `gpu/attn_utils.py` | AttentionGroup 初始化、backend 绑定、per-layer attn_metadata 构建、`CommonAttentionMetadata` 创建 |
| `gpu/buffer_utils.py` | `StagedWriteTensor`、`UvaBackedTensor`、`FusedStagedWriter` |
| `gpu/cudagraph_utils.py` | CUDA graph 工具函数 |
| `gpu/spec_decode/` | 投机解码 worker 端文件 |
| `gpu/structured_outputs.py` | GPU 端 structured output 处理 |
| `gpu/warmup.py` | 模型 warmup 例程 |
| `gpu/mm/` | 多模态 worker 端文件 |
| `gpu/kv_connector.py` | KV connector worker 端 |
| `gpu/lora_utils.py` | LoRA 工具 |
| `gpu/metrics/` | worker 端指标 |
| `gpu/pool/` | pooling worker |
| `gpu/pcp_manager.py` | PCP manager |
| `gpu/pp_utils.py` | pipeline parallel 工具 |
| `gpu/shutdown.py` | worker 关闭协议 |
| `gpu/async_utils.py` | 异步工具 |
| `gpu/cp_utils.py` | context parallel 工具 |
| `gpu/dp_utils.py` | data parallel 工具 |
| `gpu/eplb_utils.py` | expert parallel load balancing |
| `cpu_model_runner.py` + `cpu_worker.py` + `cpu/` | CPU 执行路径 |
| `dp_utils.py` | 数据并行工具（跨 DP rank batch 协调） |
| `ubatching.py` + `ubatch_utils.py` + `gpu_ubatch_wrapper.py` | micro-batch 拆分（降低峰值内存） |
| `workspace.py` | GPU workspace 临时张量管理 |
| `lora_model_runner_mixin.py` | LoRA 混入 |
| `kv_connector_model_runner_mixin.py` | KV connector 混入 |
| `ec_connector_model_runner_mixin.py` | encoder cache connector 混入 |
| `encoder_cudagraph.py` + `encoder_cudagraph_defs.py` | encoder CUDA graph |
| `mamba_utils.py` | Mamba 状态管理工具 |
| `startup_plan.py` | worker 布局的启动计划 |
| `utils.py` | worker 通用工具函数 |
| `xpu_model_runner.py` + `xpu_worker.py` | Intel XPU 支持 |
| `tpu_input_batch.py` | TPU input batching |

### `gpu_model_runner.py` 缺失功能

CUDA graph capture/replay（`CUDAGraphWrapper` / `BreakableCUDAGraphWrapper` / `CudagraphDispatcher`）、`ForwardContext` / `BatchDescriptor` 注入、speculative decoding + draft token 处理、encoder-decoder / 多模态 encoder 支持、MoE `RoutedExpertsCapturer`、offloader（GPU→CPU）、`EplbState`（expert parallelism）、LoRA integration with `LoRAMapping`、Mamba SSU backend、MRotaryEmbedding / XRotaryEmbedding、DP attention、UBatch wrapper、`CompilationCounter`、多模态 pruning、async scheduling support

---

## 四、`v1/executor/` — 执行后端

| 缺失模块 | 职责 |
|----------|------|
| `multiproc_executor.py` (1089行) | 多进程 worker 管理、worker failure 检测/回调、shared memory 广播（`MessageQueue`）、`KVOutputAggregator`、DP/TP/PP/PCP/DCP group 管理、异步 `execute_model()`/`sample_tokens()`、GPU memory profiling、`init_kv_output_aggregator()`、`determine_available_memory()` |
| `ray_executor.py` (628行) | Ray 分布式执行：`RayWorkerMetaData`、placement group 调度、zero-copy 输出处理 |
| `ray_executor_v2.py` | 新版 Ray executor |
| `ray_utils.py` | `RayWorkerWrapper`、`FutureWrapper` |
| `ray_env_utils.py` | Ray worker 环境变量处理 |
| `vllm_net_devices.py` | 分布式执行的网络设备配置 |

---

## 五、`v1/attention/` — Attention 后端

### 缺失后端（15+ 个）

| 后端 | 说明 |
|------|------|
| `flashinfer.py` | FlashInfer backend + TRT-LLM 风格 KV cache |
| `triton_attn.py` | Triton unified attention kernel |
| `flex_attention.py` | Torch FlexAttention backend |
| `mamba_attn.py` / `mamba1_attn.py` / `mamba2_attn.py` | Mamba SSM 后端 |
| `linear_attn.py` | 线性注意力 |
| `cpu_attn.py` | CPU 注意力 |
| `mla/` (20 文件) | 完整 MLA 后端：`flashmla.py`、`flashinfer_mla.py`、`flashinfer_mla_sparse.py`、`cutlass_mla.py`、`triton_mla.py`、`flashattn_mla.py`、`sparse_swa.py`、`indexer.py`、`compressor_utils.py` 等 |
| `rocm_attn.py` + `rocm_aiter_fa.py` + `rocm_aiter_unified_attn.py` | AMD ROCm 后端 |
| `turboquant_attn.py` | 量化 KV cache 后端 |
| `gdn_attn.py` / `hpc_attn.py` / `short_conv_attn.py` | 专用注意力后端 |
| `flash_attn_diffkv.py` / `triton_attn_diffkv.py` | Differential KV cache 后端 |
| `fa_utils.py` | FlashAttention 公用工具 |
| `registry.py` | 后端注册表 + `AttentionBackendEnum` |

### 缺失 Triton ops（19+ 个核函数）

| ops 文件 | 说明 |
|----------|------|
| `chunked_prefill_paged_decode.py` | chunked prefill + paged decode 融合 kernel |
| `prefix_prefill.py` | prefix prefill kernel |
| `triton_decode_attention.py` | Triton decode attention kernel |
| `triton_prefill_attention.py` | Triton prefill attention kernel |
| `triton_unified_attention.py` | 统一注意力 kernel |
| `triton_unified_attention_diffkv.py` | diff KV 统一注意力 kernel |
| `triton_reshape_and_cache_flash.py` | reshape + cache 写入 kernel |
| `triton_fp8_mqa_logits.py` | FP8 MQA logits kernel |
| `triton_merge_attn_states.py` | 合并注意力状态 kernel |
| `triton_turboquant_decode.py` | 量化 decode kernel |
| `triton_turboquant_store.py` | 量化 store kernel |
| `merge_attn_states.py` | 注意力状态合并 |
| `common.py` | 公用 attention ops |
| `dcp_alltoall.py` | DCP all-to-all |
| `flashmla.py` | FlashMLA ops |
| `vit_attn_wrappers.py` | ViT 注意力包装 |
| `xpu_mla_sparse.py` | XPU MLA sparse |
| `rocm_aiter_mla_sparse.py` | ROCm MLA sparse |
| `int4_per_token_head.py` | INT4 per-token head |

### `backend.py` 缺失

`AttentionType` 枚举 (DECODER/ENCODER/ENCODER_ONLY/ENCODER_DECODER)、`supported_dtypes`、`supported_kv_cache_dtypes`、`forward_includes_kv_cache_update`、`get_supported_kernel_block_sizes()`、`get_kv_cache_shape()`、`get_kv_cache_layout()`、`get_kv_cache_stride_order()`、`AttentionMetadataBuilder`、`AttentionMetadata` 子类、`QuantKey` 量化类型

---

## 六、`v1/sample/` — 采样

| 缺失模块 | 职责 |
|----------|------|
| `logits_processor/interface.py` | LogitsProcessor ABC |
| `logits_processor/builtin.py` | `MinTokensLogitsProcessor` 等内置处理器 |
| `logits_processor/state.py` | `BatchUpdateBuilder` — batch 状态更新管理 |
| `rejection_sampler.py` (953行) | `RejectionSampler` (nn.Module) — 投机解码 acceptance/rejection 算法、bonus token 处理、Triton draft validation 核函数、`PLACEHOLDER_TOKEN_ID`、`GREEDY_TEMPERATURE`、`MAX_SPEC_LEN` |
| `thinking_budget_state.py` | 推理模型 thinking budget 追踪 |
| `ops/bad_words.py` | `apply_bad_words` op |
| `ops/logprobs.py` | `batched_count_greater_than` op |
| `ops/penalties.py` | `apply_all_penalties` Triton kernel（repetition + frequency + presence） |
| `ops/topk_topp_sampler.py` | `TopKTopPSampler` 类 |
| `ops/topk_topp_triton.py` | Triton fused top-k/top-p kernel |
| `sampler.py` 缺失功能 | `LogprobsMode` (raw_logprobs/raw_logits)、`predict_bonus_token`（spec decode）、FP64 gumbel、bad words / min tokens / logit bias 过滤、full logprobs gathering pipeline |
| `metadata.py` 缺失字段 | `min_p`、`bad_words`、`min_tokens`、`allowed_tokens`、`all_random` mask、`max_num_logprobs`、`generators` |

---

## 七、`v1/serial_utils.py`

参考 628 行 vs 当前 28 行（JSON stub）。缺失：

- `msgspec` + `msgpack` 序列化引擎
- `OOBTensorConsumer` / `OOBTensorProvider` ABC（带外张量 IPC）
- `CUSTOM_TYPE_PICKLE` / `CUSTOM_TYPE_CLOUDPICKLE` / `CUSTOM_TYPE_RAW_VIEW` 编码类型
- 多模态字段序列化（`MMF_CLASS_TO_FACTORY`）
- `_custom_ext_type_encode` / `_custom_ext_type_decode`（numpy/torch 张量 raw views）
- cloudpickle fallback
- `VLLM_INSECURE_SERIALIZATION` 安全序列化控制

---

## 八、`v1/metrics/` — 指标

| 缺失模块 | 职责 |
|----------|------|
| `loggers.py` (1402行) | `StatLoggerBase` ABC、`PrometheusStatLogger`、`AggregateStatLoggerBase`、`StatLoggerManager`、插件式 stat logger factory、per-engine 指标：TTFT（首 token 时间）、TPOT（per-output-token 时间）、ITL（token 间延迟）、request rate、throughput、GPU cache 使用率、prefix cache 命中率、waiting/running 计数、prefill/decode token 计数、hidden states 大小、KV cache 利用率、LoRA 状态、wait reasons（capacity/deferred）、sleep state |
| `prometheus.py` | `setup_multiprocess_prometheus()`、`get_prometheus_registry()`、`unregister_vllm_metrics()`、`shutdown_prometheus()` |
| `stats.py` | `SchedulerStats`、`IterationStats`、`CachingMetrics`、`PrefixCacheStats`、`SchedulerIterationDetails`、`PromptTokenStats`、`BaseCacheStats`、`KVCacheEvictionEvent`、`RequestStateStats`、`LoRARequestStates`、`MultiModalCacheStats`、`PrefillStats`、`SpecDecodingStats` |
| `perf.py` | `PerfStats`、`ModelMetrics`、`PerfMetricsLogging`、`PerfMetricsProm` |
| `ray_wrappers.py` | Ray 特定指标包装 |
| `reader.py` | `Metric` 类型、`get_metrics_snapshot()` |
| `utils.py` | `create_metric_per_engine()` |

---

## 九、`v1/kv_offload/` + `v1/simple_kv_offload/`

**完整缺失**两个目录。

| 目录 | 内容 |
|------|------|
| `kv_offload/` | `base.py` (offload backend ABC)、`config.py`、`cpu/` (host memory backend)、`factory.py` (创建 offload backend)、`file_mapper.py` (磁盘持久化映射)、`tiering/` (分层策略) |
| `simple_kv_offload/` | `copy_backend.py` (GPU↔CPU 拷贝)、`cuda_mem_ops.py` (GPU 内存操作)、`manager.py` (offload 生命周期管理)、`metadata.py`、`worker.py` |

---

## 十、`model_executor/` + `forward_context.py` — 模型加载与执行上下文

| 缺失 | 说明 |
|------|------|
| `forward_context.py` | `BatchDescriptor` dataclass（CUDA graph 分发：num_tokens, num_reqs, uniform, has_lora, num_active_loras）、`set_forward_context()` context manager、`get_forward_context()`、`is_forward_context()` — Attention 接入 model forward 的关键机制 |
| `model_executor/layers/attention/` | `Attention` 和 `MLAAttention` layer 类——替代 HF attention 并分发到 V1 backend、`AttentionLayerBase` ABC、量化感知 attention（fp8, int4, nvfp4） |
| `unified_attention_with_output` | Triton fused QKV 投影 + attention 计算的 custom op |
| `compilation/` | `CUDAGraphWrapper`、`BreakableCUDAGraphWrapper`、`CompilationCounter`、cudagraph monitoring |

---

## 十一、`v1/` 顶级缺失目录

| 目录 | 内容 |
|------|------|
| `v1/spec_decode/` (18文件) | 投机解码 proposer：Eagle (`eagle.py`)、Medusa (`medusa.py`)、n-gram (`ngram_proposer.py`)、dFlash (`dflash.py`)、Gemma4 (`gemma4.py`)、custom proposer、suffix decoding、dynamic scheduling、`metrics.py`、`utils.py` |
| `v1/structured_output/` (8文件) | JSON schema / grammar 结构化输出：Outlines backend、XGrammar backend、LM Format Enforcer backend、Guidance backend、`backend_types.py`、`request.py`、`utils.py` |
| `v1/pool/` (4文件) | pooling 模型（embedding）、late interaction（ColBERT 风格） |
| `v1/cudagraph_dispatcher.py` (350行) | CUDA graph 运行时调度器 |
| `v1/kv_cache_spec_registry.py` (209行) | KVCacheSpec 类型和 manager 的可插拔注册表 |
| `v1/utils.py` | `CpuGpuBuffer`、`StagedWriteTensor`、`UvaBackedTensor`、`compute_iteration_details()`、`record_function_or_nullcontext()`、`tensor_data()`、`copy_slice()`、`get_engine_client_zmq_addr()`、`shutdown()` |

---

## 十二、`config/` — 配置系统

当前 42 行 `EngineConfig` vs 参考 31 个配置文件（`VllmConfig` 聚合）。缺失：

| 配置类 | 关键字段 |
|--------|----------|
| `VllmConfig` | 顶层聚合器（2450行） |
| `CacheConfig` | `block_size`、`num_gpu_blocks`、`cache_dtype` / `kv_cache_dtype`、`kv_cache_memory_bytes`、`kv_cache_size_tokens`、`kv_cache_max_concurrency`、sliding window |
| `SchedulerConfig` | `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`enable_chunked_prefill`、`schedule_policy`、`async_scheduling`、`stream_interval`、`get_scheduler_cls()` |
| `CompilationConfig` | `CUDAGraphMode` (FULL/PIECEWISE/NONE)、`CompilationMode`、cudagraph specialization 选项 |
| `ParallelConfig` | TP、PP、DP、CP、EP、DCP、DP coordinator、elastic EP、placement groups、NUMA binding |
| `LoRAConfig` | LoRA 模块和请求类型 |
| `SpeculativeConfig` | 投机解码 proposer 配置 |
| `StructuredOutputsConfig` | grammar/JSON schema backend 选择 |
| `ObservabilityConfig` | Prometheus、tracing 配置 |
| `QuantizationConfig` | FP8/INT8/INT4/NVFP4/AWQ/GPTQ |
| `MultiModalConfig` | 视觉/音频 encoder 配置 |
| `KVTransferConfig` | KV connector 传输配置 |
| `ECTransferConfig` | encoder cache 传输配置 |
| `OffloadConfig` | KV offload 配置 |
| `ReasoningConfig` | 推理模型配置 |
| `AttentionConfig` | 注意力后端选择 |
| `DeviceConfig` | 设备选择 |
| `LoadConfig` | 模型加载选项 |
| `KernelConfig` | kernel 调优 |
| `MambaConfig` | Mamba SSM 参数 |
| `ModelConfig` | HF model name、dtype、trust_remote_code |
| `PoolerConfig` | pooling 模型配置 |
| `ProfilerConfig` | profiler 选项 |
| `KVEventsConfig` | KV event 发布配置 |
| `DiffusionConfig` | 扩散模型配置 |
| `WeightTransferConfig` | 权重传输配置 |
| `EncoderCacheManagerConfig` | encoder cache manager 配置 |

---

## 十三、`v1/kv_cache_interface.py` — KV Cache 规格接口

参考 1002 行 vs 当前简化版。缺失：

- `KVCacheSpec` 完整 ABC + 所有子类型：`FullAttentionSpec`、`SlidingWindowSpec`、`MLAAttentionSpec`、`MambaSpec`、`HiddenStateCacheSpec`、`CrossAttentionSpec`、`ChunkedLocalAttentionSpec` 等
- `KVCacheGroupSpec` — 多 attention group 组合
- `KVCacheConfig` — KV cache 完整配置
- `KVQuantMode` — 量化模式

---

## 十四、顶层缺失模块

| 模块 | 说明 |
|------|------|
| `engine/arg_utils.py` | `EngineArgs` + `AsyncEngineArgs` — CLI 参数解析 |
| `engine/protocol.py` | `EngineClient` 协议 — API server 使用的引擎接口 |
| `entrypoints/openai/` | OpenAI 兼容 API server（chat completions, completions, embeddings） |
| `entrypoints/llm.py` | `LLM` 类 — 离线推理 API |
| `entrypoints/launcher.py` | 服务器启动器 |
| `entrypoints/cli/` | CLI 工具 |
| `entrypoints/serve/` | 生产服务器（含 Elastic EP、scaling） |
| `entrypoints/grpc_server.py` | gRPC 服务器 |
| `entrypoints/anthropic/` | Anthropic 兼容 API |
| `entrypoints/generate/` | 生成入口 |
| `entrypoints/chat_utils.py` | 对话工具 |
| `distributed/` | KV transfer connectors、EC transfer connectors、parallel state（TP/PP/EP/DCP/PCP groups）、device communicators、weight transfer（RL training） |
| `lora/` | LoRA layers（`lora/layers.py`）、request types、LoRA config |
| `quantization/` | FP8、INT8、INT4、NVFP4、AWQ、GPTQ 量化后端 |
| `multimodal/` | `MultiModalRegistry`、`MultiModalFeatureSpec`、`MultiModalBudget`、encoder budget management |
| `compilation/` | CUDA graph wrapping、breakable cudagraphs、compilation counting |
| `plugins/` | stat logger plugins、general plugins |
| `tracing/` | OpenTelemetry tracing 集成 |
| `renderers/` | `BaseRenderer` — 多模态 prompt 渲染 |
| `utils/` (39文件) | `network_utils.py`（ZMQ socket、TCP/IPC 地址）、`system_utils.py`（进程管理）、`hashing.py`（`sha256_cbor`、`xxhash_cbor`）、`mem_utils.py`、`gc_utils.py`（`freeze_gc_heap()`）、`async_utils.py`（`cancel_task_threadsafe()`）、`torch_utils.py`、`collection_utils.py`、`numa_utils.py` 等 |

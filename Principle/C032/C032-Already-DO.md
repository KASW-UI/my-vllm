# C032-Already-DO — 对照 vLLM 参考实现的 C032 完成进度

共 14 个领域，每个领域标注 DONE（C032 已完成）或 UNDO（仍未完成）。

---

## 一、`v1/engine/` — 引擎层

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `coordinator.py` — DP 多卡协调器 | UNDO |
| `core_client.py` — 引擎客户端抽象层（Inproc/MP/DP clients） | UNDO |
| `parallel_sampling.py` — n>1 并行采样 | UNDO |
| `tensor_ipc.py` — 跨进程零拷贝 Tensor 传输 | UNDO |
| `utils.py` — 引擎进程生命周期管理（spawn/Ray/ZMQ） | UNDO |
| `logprobs.py` — 增量 logprobs 处理 | UNDO |
| `__init__.py` — `EngineCoreRequest`(msgspec, 20+字段)、`FinishReason` IntEnum 等 | UNDO |
| `core.py` — **EngineCore.step() 控制流对齐** | **DONE** |
| `core.py` — ZMQ 队列、batch queue、KV cache profiling、structured output、sleep/wake_up、pause/resume、LoRA、prefix cache reset、DP 协调、shutdown 等 | UNDO |
| `async_llm.py` — background output_handler、streaming input、`encode()`、pause/resume、LoRA、weight transfer、Prometheus 等 | UNDO |
| `llm_engine.py` — `from_engine_args()`、`collective_rpc()`、`apply_model()`、LoRA、sleep 等 | UNDO |
| `input_processor.py` — 多模态、encoder-decoder、renderer、cache salt、LoRA、speculative config 等 | UNDO |
| `output_processor.py` — `RequestOutputCollector`(async queue + delta streaming)、stop-string、logprobs、parallel sampling 聚合等 | UNDO |
| `detokenizer.py` — `FastIncrementalDetokenizer`、stop-string detection、leading space skip | UNDO |

**C032 细节**：
- `core.py:step()` 重构：`schedule() → execute_model() → scheduler.update_from_output() → _finish_request() → output_processor.process_outputs()`，对齐参考 vLLM 的控制流骨架
- 其余引擎层能力均不在 C032 范围内

---

## 二、`v1/core/` — 调度器 & KV Cache 核心

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `encoder_cache_manager.py` — 多模态 encoder 输出缓存 | UNDO |
| `kv_cache_coordinator.py` — 多 attention type 协调器（903 行） | UNDO |
| `kv_cache_metrics.py` — block lifetime / cache 命中率统计 | UNDO |
| `single_type_kv_cache_manager.py` — 10+ 种具体管理器 | UNDO |
| `sched/interface.py` — `SchedulerInterface` ABC（254 行） | UNDO |
| `sched/async_scheduler.py` — 异步调度器 | UNDO |
| `sched/utils.py` — `check_stop()`、`remove_all()` | UNDO |
| `sched/scheduler.py` — priority 调度、preemption、KV/EC connector、structured output grammar、encoder cache、multimodal budget、draft token、DP throttling、streaming、LoRA、KV events | UNDO |
| `sched/scheduler.py` — **`update_from_output()` + `_find_request()`** | **DONE** |
| `sched/output.py` — `GrammarOutput`、`ScheduledEncoderInputStats` | UNDO |
| `block_pool.py` — `BlockHashToBlockMap`、CoW(`_maybe_evict_cow_blocks`)、`gc_blocks()`、`KVConnector` 集成、MetricsCollector | UNDO |
| `kv_cache_manager.py` — 多 group 分配、sliding window、CoW 感知分配 | UNDO |
| `kv_cache_utils.py` — `KVCacheBlockCopy`、`BlockHashWithGroupId`、hash extra keys、`resolve_kv_cache_block_sizes()`、`generate_scheduler_kv_cache_config()` | UNDO |

**C032 细节**：
- `scheduler.py` 新增 `update_from_output()`（检测 EOS/max_tokens → 标记 finish）和 `_find_request()`
- `outputs.py` 新增 `EngineCoreOutputs` dataclass（`finished_requests` + `sampled_token_ids`）
- 其余调度/KV cache 核心能力均不在 C032 范围内

---

## 三、`v1/worker/` — 模型执行器

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `gpu_input_batch.py` — `CachedRequestState`(35+字段)、多 group block tables、logprobs、LoRA、multimodal、spec decode 等 | UNDO |
| `block_table.py` — Triton kernels 替代 Python 循环、`MultiGroupBlockTable`、`SlotMappingMode`、`copy_blocks()`、`FusedStagedWriter` | UNDO |
| `gpu/states.py` — `StagedWriteTensor`、`UvaBackedTensor`、`num_computed_prefill_tokens`、`draft_tokens`、CPU mirror | UNDO |
| `gpu/attn_utils.py` — AttentionGroup 初始化、per-layer metadata 构建 | UNDO |
| `gpu/buffer_utils.py` — `FusedStagedWriter` | UNDO |
| `gpu/cudagraph_utils.py` — CUDA graph 工具 | UNDO |
| `gpu/spec_decode/` — 投机解码 worker | UNDO |
| `gpu/structured_outputs.py` — GPU 端 structured output | UNDO |
| `gpu/warmup.py` — 模型 warmup | UNDO |
| `gpu/mm/` — 多模态 worker | UNDO |
| `gpu/kv_connector.py` — KV connector worker | UNDO |
| `gpu/lora_utils.py` — LoRA | UNDO |
| `gpu/metrics/` / `gpu/pool/` / `gpu/pcp_manager.py` / `gpu/pp_utils.py` / `gpu/shutdown.py` / `gpu/async_utils.py` / `gpu/cp_utils.py` / `gpu/dp_utils.py` / `gpu/eplb_utils.py` — 其余 gpu 子模块 | UNDO |
| `cpu_model_runner.py` + `cpu_worker.py` + `cpu/` — CPU 路径 | UNDO |
| `dp_utils.py` — 数据并行 | UNDO |
| `ubatching.py` + `ubatch_utils.py` + `gpu_ubatch_wrapper.py` — micro-batch | UNDO |
| `workspace.py` — GPU workspace | UNDO |
| `lora_model_runner_mixin.py` / `kv_connector_model_runner_mixin.py` / `ec_connector_model_runner_mixin.py` | UNDO |
| `encoder_cudagraph.py` — encoder CUDA graph | UNDO |
| `mamba_utils.py` — Mamba 状态 | UNDO |
| `startup_plan.py` + `utils.py` | UNDO |
| `xpu_model_runner.py` + `xpu_worker.py` + `tpu_input_batch.py` — 其他硬件 | UNDO |
| `gpu_model_runner.py` — **移除 legacy bridge methods（`_prefill_one` / `_decode_one_legacy`）** | **DONE** |
| `gpu_model_runner.py` — **提取 `_write_kv_to_physical_cache()`** | **DONE** |
| `gpu_model_runner.py` — **解码 KV 写入改用 `reshape_and_cache_pytorch`** | **DONE** |
| `gpu_model_runner.py` — CUDA graph、spec decode、encoder-decoder、MoE、offloader、LoRA、Mamba、UBatch 等 | UNDO |
| `gpu_worker.py` — **移除 legacy `execute_prefill` / `execute_decode`** | **DONE** |
| `gpu_worker.py` — 其余 | UNDO |

**C032 细节**：
- `gpu_model_runner.py`：删除 `_prefill_one()` 和 `_decode_one_legacy()` 两个 bridge 方法及 `KVCache` / `greedy_sample` 导入。`execute_model()` 完全通过 `Sampler(nn.Module)` + 物理 KV cache 运行
- `gpu_model_runner.py`：`_write_prefill_to_cache()` 重命名为 `_write_kv_to_physical_cache()`
- `gpu_model_runner.py:decode`：解码 KV 写入从 direct tensor assignment 改为 `reshape_and_cache_pytorch(k_last, v_last, kv_cache, slot_tensor)`，与 prefill 写入统一
- `gpu_worker.py`：删除 `execute_prefill()` / `execute_decode()` 及 `KVCache` 导入

---

## 四、`v1/executor/` — 执行后端

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `multiproc_executor.py`(1089行) — 多进程 worker、failure detection、shared memory、KVOutputAggregator、DP/TP/PP/PCP/DCP、async `sample_tokens()` | UNDO |
| `ray_executor.py` + `ray_executor_v2.py` — Ray 分布式执行 | UNDO |
| `ray_utils.py` + `ray_env_utils.py` — Ray 工具 | UNDO |
| `vllm_net_devices.py` — 网络设备配置 | UNDO |

**C032 细节**：未涉及 executor 层变更。

---

## 五、`v1/attention/` — Attention 后端

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `flashinfer.py` — FlashInfer backend | UNDO |
| `triton_attn.py` — Triton unified attention | UNDO |
| `flex_attention.py` — Torch FlexAttention | UNDO |
| `mamba_attn.py` / `mamba1_attn.py` / `mamba2_attn.py` — Mamba SSM | UNDO |
| `linear_attn.py` — 线性注意力 | UNDO |
| `cpu_attn.py` — CPU 注意力 | UNDO |
| `mla/`(20文件) — MLA 后端 | UNDO |
| `rocm_attn.py` 等 AMD 后端 | UNDO |
| `turboquant_attn.py` — 量化 KV cache | UNDO |
| `flash_attn_diffkv.py` / `triton_attn_diffkv.py` — Diff KV | UNDO |
| `registry.py` — 后端注册表 | UNDO |
| 19+ Triton ops（chunked_prefill、prefix_prefill、decode/prefill attention、unified attention、turboquant 等） | UNDO |
| `backend.py` — `AttentionType` 枚举、`supported_dtypes`、`get_kv_cache_layout()`、`AttentionMetadataBuilder`、量化类型 | UNDO |

**C032 细节**：未涉及 attention 层变更。物理 KV cache 写入/读取通过 `paged_attn.py` 的 Python 实现进行，未替换 HF attention。

---

## 六、`v1/sample/` — 采样

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `logits_processor/interface.py` — ABC | UNDO |
| `logits_processor/builtin.py` — `MinTokensLogitsProcessor` | UNDO |
| `logits_processor/state.py` — `BatchUpdateBuilder` | UNDO |
| `rejection_sampler.py`(953行) — 投机解码 rejection | UNDO |
| `thinking_budget_state.py` — reasoning budget | UNDO |
| `ops/`(5文件) — bad_words、logprobs、penalties(Triton)、topk_topp | UNDO |
| `sampler.py` — `LogprobsMode`、`predict_bonus_token`、FP64 gumbel、bad words/min tokens/logit bias | UNDO |
| `metadata.py` — `min_p`、`bad_words`、`min_tokens`、`allowed_tokens`、`all_random`、`max_num_logprobs` | UNDO |

**C032 细节**：采样层未变更。`greedy_sample()` 函数（`generation/sampler.py`）已删除，由现有 `v1/sample/sampler.py` 中的 `Sampler(nn.Module)` 替代。`GPUModelRunner` 已使用 `Sampler` 而非旧函数。

---

## 七、`v1/serial_utils.py`

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `msgspec` + `msgpack` 序列化引擎 | UNDO |
| `OOBTensorConsumer` / `OOBTensorProvider`（带外张量 IPC） | UNDO |
| `CUSTOM_TYPE_PICKLE` / `CLOUDPICKLE` / `RAW_VIEW` 编码 | UNDO |
| 多模态字段序列化 | UNDO |
| cloudpickle fallback | UNDO |

**C032 细节**：未涉及。当前为 JSON stub。

---

## 八、`v1/metrics/` — 指标

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `loggers.py`(1402行) — Prometheus：TTFT, TPOT, ITL, throughput, GPU cache, prefix cache hit rate, LoRA, wait reasons | UNDO |
| `prometheus.py` — 多进程 Prometheus setup/shutdown | UNDO |
| `stats.py` — `SchedulerStats`、`IterationStats`、`CachingMetrics`、`PrefixCacheStats` 等 | UNDO |
| `perf.py` — 性能指标 | UNDO |
| `ray_wrappers.py` / `reader.py` / `utils.py` | UNDO |

**C032 细节**：未涉及。

---

## 九、`v1/kv_offload/` + `v1/simple_kv_offload/`

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `kv_offload/` — offload backend、cpu/、factory、file_mapper、tiering | UNDO |
| `simple_kv_offload/` — copy_backend、cuda_mem_ops、manager、metadata、worker | UNDO |

**C032 细节**：未涉及。

---

## 十、`model_executor/` + `forward_context.py` — 模型加载与执行上下文

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `forward_context.py` — `BatchDescriptor`、`set_forward_context()`、`get_forward_context()` | UNDO |
| `layers/attention/` — `Attention`/`MLAAttention` layer 替代 HF attention | UNDO |
| `unified_attention_with_output` — Triton fused QKV + attention custom op | UNDO |
| `compilation/` — `CUDAGraphWrapper`、`CompilationCounter` | UNDO |
| `model_loader.py` — **内联 `load_model` 实现，不再依赖 `runtime/`** | **DONE** |
| `models/opt.py` — **修复失效导入（`from transformers import OPTForCausalLM`）** | **DONE** |

**C032 细节**：
- `model_executor/model_loader.py`：从 `from runtime.model_loader import load_model` 改为内联实现，接口不变
- `model_executor/models/opt.py`：从 `from runtime.models.opt import OPTForCausalLM`（模块不存在）改为 `from transformers import OPTForCausalLM`

---

## 十一、`v1/` 顶级缺失目录

| 参考 vLLM 的目录 | C032 状态 |
|------------------|-----------|
| `v1/spec_decode/`(18文件) — Eagle、Medusa、n-gram、dFlash、Gemma4 等 | UNDO |
| `v1/structured_output/`(8文件) — Outlines、XGrammar、LM Format Enforcer | UNDO |
| `v1/pool/`(4文件) — pooling、late interaction | UNDO |
| `v1/cudagraph_dispatcher.py`(350行) | UNDO |
| `v1/kv_cache_spec_registry.py`(209行) | UNDO |
| `v1/kv_cache_interface.py` — 完整 `KVCacheSpec` + 子类型 + `KVCacheGroupSpec` + `KVCacheConfig` + `KVQuantMode` | UNDO |
| `v1/utils.py` — `CpuGpuBuffer`、`StagedWriteTensor`、`UvaBackedTensor` 等 | UNDO |

**C032 细节**：未涉及。

---

## 十二、`config/` — 配置系统

| 参考 vLLM 的能力 | C032 状态 |
|------------------|-----------|
| `VllmConfig` — 顶层聚合器（2450 行） | UNDO |
| `CacheConfig` — `cache_dtype`、`kv_cache_memory_bytes`、sliding window | UNDO |
| `SchedulerConfig` — `enable_chunked_prefill`、`schedule_policy`、`async_scheduling` | UNDO |
| `CompilationConfig` — `CUDAGraphMode` | UNDO |
| `ParallelConfig` — TP/PP/DP/CP/EP/DCP | UNDO |
| `LoRAConfig`、`SpeculativeConfig`、`StructuredOutputsConfig`、`ObservabilityConfig`、`QuantizationConfig`、`MultiModalConfig`、`KVTransferConfig`、`OffloadConfig`、`ReasoningConfig`、`AttentionConfig`、`DeviceConfig`、`LoadConfig`、`KernelConfig`、`MambaConfig`、`ModelConfig`、`PoolerConfig`、`ProfilerConfig`、`WeightTransferConfig`、`EncoderCacheManagerConfig` | UNDO |

**C032 细节**：保留现有 42 行 `EngineConfig`，未扩展到参考的 31 配置类。

---

## 十三、顶层缺失模块

| 参考 vLLM 的模块 | C032 状态 |
|------------------|-----------|
| `engine/arg_utils.py` — `EngineArgs` + `AsyncEngineArgs` | UNDO |
| `engine/protocol.py` — `EngineClient` 协议 | UNDO |
| `entrypoints/openai/` — OpenAI 兼容 API server | UNDO |
| `entrypoints/llm.py` — `LLM` 离线 API | UNDO |
| `entrypoints/launcher.py` / `cli/` / `serve/` / `grpc_server.py` / `anthropic/` / `generate/` / `chat_utils.py` | UNDO |
| `distributed/` — KV/EC transfer、parallel state、device communicator、weight transfer | UNDO |
| `lora/` — LoRA layers + request types | UNDO |
| `quantization/` — FP8/INT8/INT4/NVFP4/AWQ/GPTQ | UNDO |
| `multimodal/` — `MultiModalRegistry` | UNDO |
| `compilation/` — CUDA graph wrapping | UNDO |
| `plugins/` — stat logger / general plugins | UNDO |
| `tracing/` — OpenTelemetry | UNDO |
| `renderers/` — 多模态 prompt 渲染 | UNDO |
| `utils/`(39文件) — network_utils、system_utils、hashing、mem_utils、gc_utils、async_utils、torch_utils、numa_utils 等 | UNDO |

**C032 细节**：未涉及。

---

## C032 总结

| 类别 | DONE 数量 | UNDO 数量 | DONE 内容 |
|------|-----------|-----------|-----------|
| `v1/engine/` | 1 | 14 | `core.py:step()` 控制流对齐 |
| `v1/core/` | 1 | 15 | `scheduler.update_from_output()` + `EngineCoreOutputs` |
| `v1/worker/` | 3 | 25 | 移除 legacy bridge methods、提取 `_write_kv_to_physical_cache`、解码 KV 写入统一 |
| `v1/executor/` | 0 | 5 | — |
| `v1/attention/` | 0 | 18 | — |
| `v1/sample/` | 0 | 12 | — |
| `v1/serial_utils.py` | 0 | 1 | — |
| `v1/metrics/` | 0 | 8 | — |
| `v1/kv_offload/` | 0 | 2 | — |
| `model_executor/` | 2 | 4 | `model_loader.py` 内联实现、`models/opt.py` 修复 |
| `v1/` 顶级目录 | 0 | 7 | — |
| `config/` | 0 | 21 | — |
| 顶层模块 | 0 | 14 | — |
| **合计** | **7** | **146** | |

C032 完成了 C031 未收尾的清理工作（6 项）和引擎控制流对齐（1 项），其余 146 项能力差距留给 C033 及后续阶段。

### C032 交付物

| 项目 | 说明 |
|------|------|
| 修改 V1 源文件 | 7 个（`gpu_model_runner.py`, `gpu_worker.py`, `core.py`, `scheduler.py`, `outputs.py`, `model_loader.py`, `opt.py`) |
| 删除遗留文件 | 3 个（`cache/kv_cache.py`, `generation/sampler.py`, `batch/batch_executor.py`) |
| 重写文件 | 2 个（`runtime/model.py` LLM shim, `run.py` V1 管线） |
| 测试更新 | 4 个（`test_C010` 重写, 3 个 benchmark 归档） |
| 文档 | 4 个（C032.md, snapshot, Already-DO, UNDO） |
| Tag | `C032-cleanup` |

---

## C032 项目文件树

```
my-vllm/
├── run.py                                   # CLI entry (V1 pipeline)
├── pyproject.toml
│
├── my_vllm/
│   ├── __init__.py
│   ├── config.py                            # EngineConfig (42 lines)
│   │
│   ├── v1/                                  # === V1 Core Architecture ===
│   │   ├── __init__.py
│   │   ├── request.py                       # Request + RequestStatus (11 states)
│   │   ├── outputs.py                       # SamplerOutput, ModelRunnerOutput, EngineCoreOutputs
│   │   ├── utils.py                         # ConstantList
│   │   ├── serial_utils.py                  # JSON-based stub (28 lines)
│   │   ├── kv_cache_interface.py            # KVCacheSpec
│   │   │
│   │   ├── engine/                          # Engine layer
│   │   │   ├── __init__.py
│   │   │   ├── core.py                      # EngineCore (schedule→execute→update_from_output)
│   │   │   ├── llm_engine.py                # LLMEngine public API
│   │   │   ├── async_llm.py                 # AsyncLLM (asyncio.Queue)
│   │   │   ├── input_processor.py           # InputProcessor (prompt→tokenize→Request)
│   │   │   ├── output_processor.py          # OutputProcessor + RequestOutput
│   │   │   ├── detokenizer.py               # IncrementalDetokenizer
│   │   │   └── exceptions.py                # EngineGenerateError, EngineDeadError
│   │   │
│   │   ├── core/                            # Scheduler & KV Cache core
│   │   │   ├── __init__.py
│   │   │   ├── block_pool.py                # BlockPool (ref_cnt, hash cache, LRU free queue)
│   │   │   ├── kv_cache_manager.py          # KVCacheManager (allocate_slots, get_computed_blocks)
│   │   │   ├── kv_cache_utils.py            # KVCacheBlock, FreeKVCacheBlockQueue
│   │   │   └── sched/
│   │   │       ├── __init__.py
│   │   │       ├── scheduler.py             # FCFS Scheduler (chunked prefill, update_from_output)
│   │   │       ├── output.py                # SchedulerOutput, NewRequestData, CachedRequestData
│   │   │       └── request_queue.py         # FCFSRequestQueue
│   │   │
│   │   ├── worker/                          # Model execution
│   │   │   ├── __init__.py
│   │   │   ├── worker_base.py               # WorkerBase (ABC)
│   │   │   ├── gpu_worker.py                # Worker (GPU impl, no legacy methods)
│   │   │   ├── gpu_model_runner.py          # GPUModelRunner (batched prefill+decode, _write_kv_to_physical_cache)
│   │   │   └── gpu/
│   │   │       ├── __init__.py
│   │   │       ├── input_batch.py           # InputBatch dataclass
│   │   │       ├── block_table.py           # BlockTables (GPU tensor)
│   │   │       ├── states.py                # RequestState (GPU tensor per-request state)
│   │   │       └── model_states/
│   │   │           ├── __init__.py
│   │   │           ├── interface.py         # ModelState (ABC)
│   │   │           └── default.py           # DefaultModelState
│   │   │
│   │   ├── executor/                        # Execution backend
│   │   │   ├── __init__.py
│   │   │   ├── abstract.py                  # Executor (ABC)
│   │   │   └── uniproc_executor.py          # UniProcExecutor
│   │   │
│   │   ├── attention/                       # Attention backends
│   │   │   ├── __init__.py
│   │   │   ├── backend.py                   # AttentionBackend + AttentionImpl + CommonAttentionMetadata
│   │   │   ├── metadata.py                  # AttentionMetadata base
│   │   │   ├── selector.py                  # get_attn_backend()
│   │   │   └── backends/
│   │   │       ├── __init__.py
│   │   │       ├── torch_attn.py            # TorchAttention (SDPA fallback)
│   │   │       ├── flash_attn.py            # FlashAttention stub
│   │   │       └── ops/
│   │   │           ├── __init__.py
│   │   │           └── paged_attn.py        # Python: reshape_and_cache, gather_cached_kv
│   │   │
│   │   ├── sample/                          # Sampling pipeline
│   │   │   ├── __init__.py
│   │   │   ├── sampler.py                   # Sampler (nn.Module: greedy, top-k, top-p, penalties)
│   │   │   ├── metadata.py                  # SamplingMetadata
│   │   │   └── logits_processor.py          # apply_repetition/frequency/presence_penalty
│   │   │
│   │   └── metrics/                         # Metrics collection
│   │       ├── __init__.py
│   │       └── collector.py                 # MetricsCollector
│   │
│   ├── model_executor/                      # Model loading bridge
│   │   ├── __init__.py
│   │   ├── model_loader.py                  # load_model (inline impl)
│   │   ├── models/
│   │   │   └── opt.py                       # OPT placeholder
│   │   └── layers/
│   │       ├── __init__.py
│   │       └── linear.py                    # ColumnParallelLinear, RowParallelLinear (TP stubs)
│   │
│   ├── runtime/                             # Legacy runtime (kept for test compat)
│   │   ├── __init__.py
│   │   ├── model_loader.py                  # load_model (original impl)
│   │   ├── tokenizer.py                     # load_tokenizer
│   │   ├── model.py                         # Minimal LLM shim (HF only, no legacy deps)
│   │   └── main.py                          # Legacy CLI entry
│   │
│   ├── distributed/                         # Distributed support (TP stubs)
│   │   ├── __init__.py
│   │   └── parallel_state.py                # TP rank/world_size, all_gather, all_reduce
│   │
│   ├── server/                              # HTTP server (C014 legacy)
│   │   ├── __init__.py
│   │   ├── api_server.py                    # stdlib http.server
│   │   └── async_llm.py                     # Legacy async wrapper
│   │
│   ├── batch/                               # (empty, __init__.py only)
│   ├── cache/                               # (empty, __init__.py only)
│   ├── engine/                              # (empty, __init__.py only)
│   ├── scheduler/                           # (empty, __init__.py only)
│   ├── worker/                              # (empty, __init__.py only)
│   ├── executor/                            # (empty, __init__.py only)
│   ├── generation/                          # (empty, __init__.py only)
│   └── metrics/                             # (empty, __init__.py only)
│
├── examples/
│   ├── benchmark.py
│   ├── run.py
│   └── server.py
│
├── tests/
│   ├── legacy/                              # Archived legacy benchmarks
│   │   ├── benchmark_c003.py
│   │   ├── benchmark_C003.py
│   │   └── benchmark_C005.py
│   ├── test_C001_forward.py
│   ├── test_C002_generate.py
│   ├── test_C003_kv_cache.py
│   ├── test_C004_request.py
│   ├── test_C005_batch.py
│   ├── test_C006_dynamic.py
│   ├── test_C007_scheduler.py
│   ├── test_C008_kv_manager.py
│   ├── test_C010_model_runner.py            # Rewritten for V1 API
│   ├── test_C013_tp.py
│   ├── test_C014_llm_engine.py
│   ├── test_C015_exceptions.py
│   ├── test_C015_request.py
│   ├── test_C015_utils.py
│   ├── test_C016_request_queue.py
│   ├── test_C016_scheduler_output.py
│   ├── test_C017_batch_prefill.py
│   ├── test_C017_executor.py
│   ├── test_C018_block_pool.py
│   ├── test_C018_free_block_queue.py
│   ├── test_C018_kv_cache_manager.py
│   ├── test_C019_block_table.py
│   ├── test_C020_request_state.py
│   ├── test_C021_model_state.py
│   ├── test_C022_attention.py
│   ├── test_C023_paged_attn.py
│   ├── test_C024_sampler.py
│   ├── test_C025_output_processor.py
│   ├── test_C026_input_processor.py
│   ├── test_C027_async_scheduler.py
│   ├── test_C028_executor.py
│   ├── test_C029_prefix_cache.py
│   ├── test_C030_chunked_prefill.py
│   ├── test_C031_runtime.py
│   └── test_tp.py
│
├── docs/
│   └── snapshots/
│       ├── ... (C001-C031 snapshots)
│       └── C032.md
│
└── Principle/
    ├── Commit-Standard.md
    ├── C000-C014/
    ├── C015-C031/
    ├── C031-Final.md
    ├── C032/
    │   ├── C032.md
    │   ├── C032-Already-DO.md
    │   └── After-C032-UNDO.md
    └── C033-C050/
```

**统计**：`my_vllm/` 下 51 个 .py 文件（不含空 `__init__.py`），`tests/` 下 36 个测试文件。

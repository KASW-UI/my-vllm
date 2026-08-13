# Target FileTree

> `my_vllm/` 作为 `vllm/` 子集的最终文件树。

```
my_vllm/                          → vllm/
  __init__.py
  config/                         → vllm/config/
    __init__.py
    vllm.py        (VllmConfig, EngineConfig=alias)
    model.py       (ModelConfig)
    cache.py       (CacheConfig)
    scheduler.py   (SchedulerConfig)
    parallel.py    (ParallelConfig)
  distributed/                    → vllm/distributed/
    __init__.py
    parallel_state.py  (GroupCoordinator, initialize_model_parallel)
    elastic_ep/
      __init__.py
      dispatch.py
  model_executor/                 → vllm/model_executor/
    __init__.py
    model_loader.py
    layers/
      __init__.py
      linear.py      (ColumnParallelLinear, RowParallelLinear)
      fused_moe/
        __init__.py
        fused_moe.py
        layer.py
        router/
          __init__.py
          fused_moe_router.py
    models/
      __init__.py
      opt.py          (OPT-125m)
  v1/                             → vllm/v1/
    __init__.py
    request.py         (Request, RequestStatus, SamplingParams)
    outputs.py         (SamplerOutput, ModelRunnerOutput)
    serial_utils.py    (MsgpackEncoder, MsgpackDecoder)
    utils.py           (ConstantList)
    kv_cache_interface.py  (KVCacheSpec hierarchy)
    engine/
      __init__.py      (EngineCoreRequest, EngineCoreOutput, EngineCoreOutputs, FinishReason)
      core.py          (EngineCore)
      llm_engine.py    (LLMEngine)
      async_llm.py     (AsyncLLM)
      input_processor.py
      output_processor.py
      detokenizer.py
      exceptions.py
    core/
      __init__.py
      block_pool.py    (BlockPool, BlockHashToBlockMap)
      kv_cache_manager.py    (KVCacheManager, KVCacheBlocks)
      kv_cache_utils.py      (KVCacheBlock, FreeKVCacheBlockQueue, KVCacheBlockCopy)
      kv_cache_metrics.py    (BlockMetricsState, KVCacheMetricsCollector)
      kv_connector.py        (KVConnectorBase_V1, NoopConnector)
      sched/
        __init__.py
        interface.py          (SchedulerInterface ABC)
        scheduler.py          (Scheduler)
        output.py             (SchedulerOutput, NewRequestData, CachedRequestData)
        request_queue.py      (SchedulingPolicy, FCFSRequestQueue, PriorityRequestQueue)
        utils.py              (check_stop, remove_all)
    executor/
      __init__.py
      abstract.py     (Executor ABC)
      uniproc_executor.py   (UniProcExecutor)
      multiproc_executor.py (MultiprocExecutor)
      ray_executor.py       (RayDistributedExecutor)
      ray_utils.py
    worker/
      __init__.py
      worker_base.py      (WorkerBase)
      gpu_worker.py       (Worker)
      gpu_model_runner.py (GPUModelRunner)
      gpu/
        __init__.py
        block_table.py    (BlockTables)
        input_batch.py    (InputBatch)
        states.py         (RequestState)
        pp_utils.py       (PPHandler)
        model_states/
          __init__.py
          interface.py    (ModelState)
          default.py
    attention/
      __init__.py
      backend.py
      metadata.py         (CommonAttentionMetadata)
      selector.py         (get_attn_backend)
      backends/
        __init__.py
        torch_attn.py
        flash_attn.py
        ops/
          __init__.py
          paged_attn.py
    sample/
      __init__.py
      sampler.py          (Sampler nn.Module)
      logits_processor.py
      metadata.py         (SamplingMetadata)
    metrics/
      __init__.py
      collector.py
      stats.py            (IterationStats, SchedulerStats)
      loggers.py          (LoggingStatLogger)
      prometheus.py
    entrypoints/
      __init__.py
      api_server.py
  run.py                   (CLI entry point)
```

## 删除的目录

| 目录 | 原因 |
|------|------|
| `my_vllm/batch/` | 遗留存根 |
| `my_vllm/cache/` | 遗留存根 |
| `my_vllm/executor/` | 遗留存根 |
| `my_vllm/generation/` | 遗留存根 |
| `my_vllm/scheduler/` | 遗留存根 |
| `my_vllm/worker/` | 遗留存根 |
| `my_vllm/metrics/` | 冗余桥接层 |
| `my_vllm/runtime/model.py` | 由 model_executor 覆盖 |
| `my_vllm/runtime/main.py` | 由 run.py + v1/entrypoints/ 覆盖 |

# Chunked Prefill + Paged Decode 调度分析

## Hybrid Batch 调度原理

一个 batch 同时包含 prefill tokens 和 decode tokens：

```
Step N:
  ┌─ Prefill tokens: req_0 [t0..t15] (chunked), req_3 [t32..t47] (cached)
  └─ Decode tokens:  req_1 t17, req_2 t5

  GPU forward:
    Prefill:  reshape_and_cache → attention (Q: new tokens, KV: all history)
    Decode:   gather_cached_kv → attention (Q: 1 token, KV: full history)
```

## Scheduler 决策

### Token Budget 分配

```
max_num_batched_tokens = 2048

1. Running chunked prefill requests continue (highest priority)
   → ensures prefill progress doesn't stall

2. New requests from waiting queue
   → allocate blocks → compute cache hits → chunk if needed

3. Priority ordering: 
   - decode requests always run (1 token each, very cheap)
   - prefill requests fill remaining budget
```

### Chunked Prefill (C030)

长 prompt 拆分成多个 chunk，每次 step 执行一段：

```
prompt_len = 2048, token_budget = 1024

Step 1: tokens 0..1023     (prefill chunk 1, 1024 tokens)
Step 2: tokens 1024..2047   (prefill chunk 2, 1024 tokens)
Step 3: → decode begins
```

好处：
- 避免单个长 prompt 独占 GPU 导致其他请求延迟
- 允许 decode 请求穿插在 chunk 之间（interleaving）

### Prefix Cache 交互

```
Chunked prefill + prefix cache:
  - get_computed_blocks() checks cache hits before allocating
  - num_computed_tokens tracks how many tokens are already cached
  - allocate_slots() only allocates blocks for uncached tokens

Example:
  prompt_len = 2048, cached_tokens = 1024 (from previous request)
  → num_computed_tokens = 1024, block_ids includes cached blocks
  → remaining = 1024, chunked if > token_budget
```

## 性能 Tradeoff 分析

### TTFT vs Throughput

| 策略 | TTFT | Throughput | 适用场景 |
|---|---|---|---|
| 更多 prefill budget | 更低（prompt 更快完成） | 更低（decode 变慢） | 交互式对话 |
| 更多 decode budget | 更高（wait 时间增长） | 更高（更高并发） | 批量推理 |
| Balanced (default) | 适中 | 适中 | 混合负载 |

### Slot Mapping 策略

```
Prefill:
  slot_mapping[t] = block_id[t] * block_size + offset[t]
  → reshape_and_cache: write K/V to physical cache

Decode:
  slot_mapping[0] = block_id[last] * block_size + offset[last]
  → read via gather_cached_kv
```

### Preemption 交互 (C052)

```
KV budget exhausted → _try_preempt_for_kv_budget()
  - Select low-priority running requests
  - Free their KV blocks → retry allocation
  - Preempted requests: num_computed_tokens = 0 → re-prefill
```

## 为什么不实现完整 Hybrid Batch 引擎

C030 已有 chunked prefill 基础。C107 是分析文档，目标是理解调度决策的逻辑——
这是分布式 PD 分离下调度策略的理论基础。

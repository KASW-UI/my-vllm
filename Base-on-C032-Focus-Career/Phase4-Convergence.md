# Phase 4: 收敛与验证

> 基线: C102 (KV Connector) | 目标: 全链路 benchmark + chunked prefill 调度分析
> 包含 2 个 P0 commits: C107, C098
>
> **真实 vLLM 路径映射**: Benchmark 参考真实 vLLM `benchmarks/` 目录（顶层，不在 v1 内）。分析文档为教育项目特有。

---

## 执行顺序

| # | Commit | 依赖 | 验证 |
|---|---|---|---|
| 22 | C107: Scheduler: Chunked prefill + paged decode analysis | C030, C052 | `python run.py` |
| 23 | C098: Benchmark: throughput, TTFT/TPOT/ITL, prefill, decode, MoE, EP, TP, PP, CP | 全部 | `python benchmark/run.py` |

---

## C107 — Chunked Prefill + Paged Decode 调度分析

**来源**: `Principle/C101-C120/C101-C120.md` #C107 节

### 目标

深度分析 hybrid batch（chunked prefill + paged decode 混合）的调度策略和性能 tradeoff。

### 文件

```
docs/analysis/
  chunked_prefill_analysis.md   (新增: 分析文档)

v1/core/sched/scheduler.py     (可选: 增加调度策略变体注释/参数)
```

### 核心分析内容

**Hybrid Batch 调度原理**:
```
一个 batch 同时包含:
  - Prefill tokens (来自新请求或 chunked 请求的下一段)
  - Decode tokens (来自正在生成 token 的请求)

Scheduler 决策:
  1. token_budget 分配: 多少给 prefill，多少给 decode
  2. chunked prefill: 长 prompt 拆分成多个 chunk，每次执行一段
  3. 优先级: decode 请求优先（保证低延迟），剩余 budget 给 prefill
```

**性能 tradeoff 分析**:
```
TTFT vs Throughput:
  - 更多 prefill budget → 更低 TTFT（prompt 更快完成），但 decode 变慢
  - 更多 decode budget → 更高 throughput（更多请求并发），但 TTFT 变高
  
Slot Mapping 策略:
  - prefill: 新数据写入 KV cache → reshape_and_cache
  - decode: 读取全部 history KV → gather_cached_kv + attention
```

**为什么不实现完整的 hybrid batch 引擎**:
C030 已有 chunked prefill 基础，C107 是分析而非实现。目标是理解 hybrid batch 的调度决策逻辑——这是分布式 PD 分离下调度策略的理论基础。

### 依赖

C030 (chunked prefill) + C052 (preemption)。

### 验证

`python run.py` — 分析文档与现有代码行为一致。

### AGENTS.md update

- `Package layout` section: 新增 `docs/analysis/` 目录
- `Key architecture facts` section: 无变更（分析文档，不改运行时代码）

---

## C098 — Benchmark Suite

**来源**: `Principle/C071-C100/C070-C100.md` #C098 节

### 目标

全链路性能建模——验证从 C035 到 C098 的所有能力。

### 文件

```
benchmark/
  __init__.py
  throughput.py        # requests/sec 吞吐量
  latency.py           # TTFT, TPOT, ITL
  prefill.py           # prefill latency (single + batch)
  decode.py            # per-token decode latency
  moe.py               # MoE vs dense MLP throughput (C081)
  ep.py                # EP scaling: 1 vs 2 vs 4 GPUs (C082)
  tp.py                # TP=2 speedup (C083)
  pp.py                # PP=2 throughput + bubble analysis (C084)
  cp.py                # CP=2 long context latency (C085)
  kv_connector.py      # KV transfer latency + overhead (C102)
  run_all.py           # 运行全部 benchmark
```

### 核心实现

```python
# benchmark/throughput.py
def benchmark_throughput(
    num_prompts: int = 100,
    prompt_len: int = 128,
    max_tokens: int = 50,
) -> dict:
    return {
        "requests_per_sec": float,
        "tokens_per_sec": float,
        "ttft_avg_ms": float,
        "tpot_avg_ms": float,
        "itl_avg_ms": float,
    }

# benchmark/ep.py
def benchmark_ep_scaling(
    num_experts: int = 8,
    gpu_counts: list[int] = [1, 2, 4],
) -> dict:
    return {
        1: throughput_single_gpu,
        2: throughput_ep_2,
        4: throughput_ep_4,
    }
```

### 依赖

C035-C098 全部 P0 阶段。

### 验证

`python benchmark/run_all.py` — 所有 benchmark 正常输出。

### AGENTS.md update

- `Package layout` section: `benchmark/` 注释扩展为 `throughput, latency (TTFT/TPOT/ITL), prefill, decode, moe, ep, tp, pp, cp, kv_connector`
- `Key architecture facts` section: 无变更

---

## Phase 4 完成后的能力

- ✅ Chunked prefill + paged decode 调度分析文档
- ✅ 全链路 benchmark: throughput, TTFT/TPOT/ITL, prefill, decode, MoE, EP, TP, PP, CP, KV Connector
- ✅ **可以进入 Level 4 实战**: 改 Scheduler / 改 KV 驱逐 / Nsight profiling / KV 迁移实验

## 核心目标

核心不是：

> 继续生成新的 mini-vLLM。

而是：

> **让 Codex 基于真实 vLLM V1 最终实现，对已经完成的 Runtime 进行连续 Git Commit 演进，使其从"研究版 Runtime"逐步成长为接近真实 vLLM V1 的 Production Runtime。**

项目：

```
/home/kasw/文档/GitHub/GPU-Project/my-vllm
```

参考：

```
/home/kasw/文档/GitHub/GPU-Project/vllm
```
---

# Role

你是一名长期维护 LLM Serving Runtime 的核心工程师。

现在项目已经完成：

```
Commit001~Commit014
```

整个 Runtime 已经拥有：

- Engine
    
- Scheduler
    
- Executor
    
- Worker
    
- ModelRunner
    
- KV Cache
    
- Tensor Parallel
    
- OpenAI API
    
- Async Engine
    

整个项目已经可以正常提供推理服务。

你的任务不再是新增架构。

你的任务是：

**模拟一个真实工程团队，在 Runtime 架构冻结之后，持续迭代工程实现，使其逐渐逼近真实 vLLM V1。**

整个开发过程必须保持：

```
真实
渐进
可运行
每次 Commit 都可测试
```

而不是：

```
一次性重构
一次性复制 vLLM
```
---

# IMPORTANT: 你的任务不是复制 vLLM


禁止：

- 直接照搬 vLLM 文件结构
- 直接复制类名
- 直接复制源码
- 一开始设计完整架构


你的目标：

模拟真实工程演进。


也就是说：

最终版本应该接近 vLLM 的思想，

但是每一个中间版本必须是：

1. 可以运行
2. 有真实缺陷
3. 缺陷推动下一次重构


Commit014 已完成架构冻结。

因此：

禁止：

新增：

```
Scheduler

Executor

Worker

ModelRunner

KVCacheManager

BlockPool

BlockTable

EngineCore

AsyncLLM

API Server
```

这些组件已经存在。

之后所有 Commit：

只能：

- 重构
    
- 补全
    
- 工程优化
    
- 数据流完善
    
- 控制流完善
    
- 工程能力增强
    

不能：

新增新的核心组件。

---

# Reference Source

唯一参考：

```
vllm/v1/

vllm/model_executor/
```

重点参考：

```
engine/

core/

worker/

executor/

attention/

sample/
```

不是复制代码。

而是：

参考：

- 控制流
    
- 数据流
    
- API
    
- 工程组织
    
- 数据结构
    
---

# Principle(Most Important)

vLLM V1：/home/kasw/文档/GitHub/GPU-Project/vllm
每次 Commit必须参考/home/kasw/文档/GitHub/GPU-Project/vllm实现，在 Commit 014 完成后：
- 核心控制流与 vLLM V1 对齐
- 核心数据流与 vLLM V1 对齐
- 主要组件边界与 vLLM V1 对齐
Commit 014 之后不再新增核心架构
每个 Commit 必须是真实 Git commit，而不是阶段文档。
每个 Commit 后保留一个可运行状态。
后续 Commit 必须基于前一个 Commit 的代码修改，而不是重新生成项目。

# Principle 0

Commit014 已满足：

✅ 控制流基本对齐

✅ 数据流基本对齐

✅ 组件边界基本对齐

Commit015 开始：

不得新增核心架构。

只能不断缩小：

```
my-vllm

↓

vLLM V1
```

之间：

工程实现上的差距。

---

# Principle 1

不要为了接近 vLLM 而复制源码。

例如：

禁止：

```
直接复制 scheduler.py
```

正确：

例如：

发现：

```
Scheduler 返回 Request[]
```

导致：

Executor 无法获得：

- batch metadata
    
- kv metadata
    
- request metadata
    

于是：

新增：

```
SchedulerOutput
```

这是一次合理 Commit。

---

# Principle 2

每一个 Commit 必须来自真实工程问题。

例如：

错误：

```
实现 SchedulerOutput
```

正确：

```
Executor 开始需要更多调度信息，
因此 Scheduler 返回结构化输出。
```

---

# Principle 3

每次 Commit 必须保持：

```
git checkout <commit>

pytest

全部通过
```

不能：

中间状态不可运行。

---

# Principle 4

每次 Commit 必须保留缺陷。

例如：

实现：

```
SchedulerOutput
```

允许：

仍然：

```
execute_model()

一次一个request
```

下一次：

才重构 Batch Execute。

---

# Principle 5

不要引入：

CUDA Kernel

FlashAttention Kernel

Triton Kernel

FP8

MoE

SpecDecode

这些属于：

后续大型功能。

Commit015~030：

重点：

Runtime Engineering。

---

# Development Goal

整个阶段目标：

让 Runtime：

从：

```
研究版本
```

逐渐成长为：

```
Production Runtime
```

不是：

新增功能。

而是：

工程质量提升。

---

# Development Output Format


整个项目由 Git Commit 组成。


共17个 Commit。


每次只生成一个 Commit。


不要一次生成全部。


第一次输出：

只生成 Commit 015。


之后我输入：

继续 Commit 016


你再继续。


---

# 每个 Commit 输出格式


必须严格按照：


# Commit XXX


## Git Commit Message


格式：



```
xxx: xxx
```




例如：



```
runtime: add minimal single request inference
```




---

# 1. 本版本目标


一句话。


例如：



控制流正式与 vLLM 对齐




---

# 2. 为什么需要这个版本


描述：

当前系统有什么需求。


不要描述未来架构。


例如：



目前只有模型权重，
需要验证模型是否能够完成一次forward。




---

# 3. 当前版本项目结构


展示真实目录：

例如：



```
my_vllm/

src/


model.py
runtime.py


main.py

tests/
```




---

# 4. 完整代码


这是最重要部分。


必须生成真实代码。


要求：


每个文件：

必须完整。


格式：

## src/model.py


完整代码


禁止：

* ...
* TODO
* placeholder
* "保持不变"
* 伪代码

---

# 5. 运行方式

必须提供：

安装依赖：


pip install ...


运行：


python main.py


---

# 6. 测试方式

必须提供：

例如：


python tests/test_generate.py


并说明：

预期输出。

---

# 7. 当前版本限制

必须列出：

当前设计缺陷。

例如：


限制：

- 只能单请求
- 没有KV Cache
- 没有batch


---

# 8. Git Diff

必须说明：

新增文件：


+
xxx.py


修改文件：


M
xxx.py


删除文件：


D
xxx.py


---

# 9. 下一 Commit 为什么出现

只允许描述：

下一个问题。

不要提前写代码。

例如：


Commit016:

Scheduler 返回：
```
list[Request]
```
无法表达：
* prefill
* decode
* finished
* new blocks


---

# Evolution History

严格按照下面顺序。

注意：

这是问题演进顺序，

不是模块完成顺序。


---

C015-C030 应该按照：

**控制流 → Batch 数据流 → Worker状态 → KV物理化 → Attention抽象 → Kernel路径**




---


真实依赖应该是：

```
HF past_key_values
        |
        v
GPU KV Cache Tensor
        |
        v
BlockPool
        |
        v
BlockTable
        |
        v
Attention Metadata
        |
        v
PagedAttention
```



---

# C015

# C015 Introduce v1 Runtime Namespace and EngineCore

目标：

建立 vLLM V1 架构边界。

当前：

```
my_vllm/

engine/
    llm_engine.py
    state_manager.py


scheduler/

worker/

executor/
```

问题：

所有东西在旧 namespace：

```
engine
scheduler
worker
```

没有 v1 runtime 层。

---

## 修改

新增：

```
my_vllm/

v1/

├── __init__.py
├── outputs.py        （ModelRunnerOutput 最小占位）
├── utils.py           （ConstantList）
├── request.py         ← migrate from engine/request.py

├── engine/
│
│   ├── core.py        ← migrate from engine/state_manager.py (rename→EngineCore)
│   ├── llm_engine.py  ← migrate from engine/llm_engine.py
│   └── exceptions.py

└── core/
    └── sched/
        └── output.py  （SchedulerOutput 占位，C016 完善）

```

迁移：

```
engine/state_manager.py  → v1/engine/core.py        (StateManager → EngineCore)
engine/llm_engine.py      → v1/engine/llm_engine.py
engine/request.py         → v1/request.py
```

⚠️ C015 只完成控制循环容器迁移，**不改变内部执行模型**：

EngineCore.step() 内部此时仍然是：

```
Scheduler.schedule() → list[Request]
for each new request: _prefill()    ← 逐 request，HF past_key_values
for each running request: _decode_step()
update_state()
```

Batch 执行留到 C017。如果 C015 就改写执行模型，C017 失去演进意义。

```
scheduler/scheduler.py     — C016 才迁移到 v1/core/sched/scheduler.py
worker/                    — C017 才迁移
executor/                  — C017 才迁移
cache/                     — C018 才迁移
```

执行流：

```
EngineCore.step()


        |

        v


Scheduler.schedule()           ← 仍调用旧 scheduler，返回 list[Request]


        |

        v


Executor 逐 request 执行        ← 尚未结构化


        |

        v


update_state()
```

C015 同时引入占位数据结构（C016 完善）：

```
v1/outputs.py          → ModelRunnerOutput (最小定义)
v1/core/sched/output.py → SchedulerOutput (占位)
```

旧路径保留过渡（后续 commit 迁移）：

```
scheduler/scheduler.py     — C016 迁移到 v1/core/sched/
worker/                    — C017 迁移到 v1/worker/
executor/                  — C017 迁移到 v1/executor/
cache/                     — C018 迁移到 v1/core/
```

---

当前能力：

✅ v1 namespace建立

✅ EngineCore控制循环

缺陷：

Scheduler仍返回list[Request]

---

# C016

# Scheduler Output Protocol

对应：

```
v1/core/sched/output.py
```

问题：

当前：

```python
schedule()

return List[Request]
```

无法表达：

- prefill
    
- decode
    
- token budget
    
- block信息
    

---

新增：

```
v1/core/sched/

output.py
```

迁移：

```
scheduler/scheduler.py          (C014 返回 list[Request])
        ↓
v1/core/sched/scheduler.py      (schedule() → SchedulerOutput)

engine/waiting_queue.py          (C014 list-waiting)
engine/running_queue.py          (C014 list-running)
        ↓
v1/core/sched/request_queue.py   (FCFSRequestQueue)
```

定义：

```python
SchedulerOutput:

scheduled_new_reqs

scheduled_cached_reqs

finished_req_ids

num_scheduled_tokens

total_num_scheduled_tokens

preempted_req_ids

new_block_ids_to_zero
```

---

当前能力：

✅ Scheduler和Executor接口稳定

缺陷：

仍逐request执行

---

# C017

# C017 Execution Batch Framework

合并原 C017+C018。

对应：

```
v1/worker/gpu/input_batch.py
```

问题：

当前：

```
for request:

    execute_prefill()

    execute_decode()
```

GPU无法batch。

---

迁移（旧 worker/executor → v1/worker/ + v1/executor/）：

```
worker/model_runner.py       → v1/worker/gpu_model_runner.py
                                 (接口改为 execute_model(SchedulerOutput))
worker/gpu_worker.py         → v1/worker/gpu_worker.py
                                 (execute_prefill/decode → execute_model)
worker/worker_base.py        → v1/worker/worker_base.py
                                 (抽象方法改为 execute_model)
executor/uniproc_executor.py → v1/executor/uniproc_executor.py
                                 (新增 execute_model() 方法)
```

新增：

```
v1/executor/abstract.py      (Executor ABC，含 execute_model)
v1/worker/gpu/input_batch.py
```

结构（此阶段仅基础字段，不包含 slot_mapping — 该字段依赖 C019 BlockTable 的 GPU tensor 版本）：

```python
InputBatch:

request_ids

input_ids

positions

seq_lens

```

执行：

```
SchedulerOutput

        |

        v

InputBatch

        |

        v

ModelRunner
```

---

当前能力：

✅ batch decode

✅ batch prefill

缺陷：

KV仍是HF past_key_values

InputBatch 后续 commit 扩展：
  C019 后 → +slot_mapping, +block_table_tensor
  C022 后 → +query_start_loc, +is_prefilling, +num_computed_tokens_np

---

# C018

# C018 Physical KV Cache Runtime

注意：

不是新增cache。

C014已有：

```
cache/

block.py
block_pool.py
kv_cache_manager.py
```

但是错误：

```
HF past_key_values
```

⚠️ 这是整个 Runtime 从 HuggingFace wrapper 转向 Serving Engine 的分水岭。之前本质仍是 `transformers` inference；之后 Runtime 自己管理 GPU 显存。

---

迁移：

```
my_vllm/cache/


        ↓


my_vllm/v1/core/

```

修改：

```
cache/block.py    → v1/core/kv_cache_utils.py
                      (KVCacheBlock: +ref_cnt, +block_hash, +is_null)
                      (FreeKVCacheBlockQueue: LRU eviction 双向链表)

cache/block_pool.py   → v1/core/block_pool.py
                          (使用 FreeKVCacheBlockQueue 替代 _free_ids list)
                          (+cached_block_hash_to_block — C029 完全启用)

cache/kv_cache_manager.py → v1/core/kv_cache_manager.py
                               (接口改为 allocate_slots / get_computed_blocks)
                               (新增 KVCacheBlocks 数据结构)

新增：

v1/kv_cache_interface.py   (KVCacheSpec — Attention/Worker/KVCacheManager 之间架构边界)
```

KV：

从：

```
dict[str, KVCache]
```

变成：

```
physical KV tensor pool

+

block allocator
```

---

当前能力：

✅ KV Cache脱离HF

缺陷：

request不知道block映射

---

# C019

# Paged KV Block Table

对应：

```
worker/block_table.py
```

问题：

现在：

```
request

↓

block list
```

不是paged attention需要的：

```
logical block

↓

physical block
```

---

迁移：

```
cache/block_table.py         (Python list 版)
        ↓
v1/worker/gpu/block_table.py (GPU tensor 版: BlockTables + slot_mappings)
```

实现：

```
Request

token 0-15

        |

        block 3


token16-31

        |

        block 8
```

---

当前能力：

✅ Paged KV 数据结构

缺陷：

GPU request state不存在

---

# C020

# C020 GPU Request State

替代原 C019/C020。

对应：

```
v1/worker/gpu/states.py
```

问题：

现在：

```
ModelRunner:

dict[str, state]
```

---

新增：

```
v1/worker/gpu/

states.py
```

实现：

```python
RequestState:

req_id_to_index

all_token_ids

num_computed_tokens

last_sampled_tokens

```

---

当前能力：

✅ GPU request 生命周期独立

缺陷：

模型初始化仍混乱

---

# C021

# C021 Model Initialization State

对应：

```
v1/worker/gpu/model_states/
```

不是：

```
runtime_state
kv_state
```

---

新增：

```
v1/worker/gpu/model_states/

interface.py

default.py
```

职责：

```
load model

initialize weights

dtype

device
```

---

当前能力：

✅ Model lifecycle独立

缺陷：

Attention仍HF

---

# C022

# C022 Attention Backend Interface

对应：

```
v1/attention/backend.py
```

问题：

当前：

```
HF Attention
```

---

新增：

```
v1/attention/

backend.py

metadata.py

selector.py
```

CommonAttentionMetadata 字段来自前置 commit：

```
来源                  → 字段
─────────────────────────────────────────
C017 InputBatch       → query_start_loc, seq_lens, positions
C019 BlockTable       → block_table_tensor, slot_mapping
C020 RequestState     → num_computed_tokens, is_prefilling
```

接口：

不是：

```
prefill()

decode()
```

而是：

```python
forward(
 layer,
 query,
 key,
 value,
 kv_cache,
 attn_metadata,
 output
)
```

---

当前能力：

✅ Attention抽象存在

缺陷：

没有paged kernel

---

# C023

# C023 PagedAttention Runtime

对应：

```
v1/attention/backends/ops/paged_attn.py
```

实现：

PyTorch版本：

```
Q

+

BlockTable

+

KV tensor


↓

attention output
```

---

当前能力：

✅ Runtime真正使用Paged KV

缺陷：

慢

---

# C024

# Sampling Pipeline

对应：

```
v1/sample/
```

移动（从 gpu_model_runner.py 拆出 sampler 职责）：

```
v1/worker/gpu_model_runner.py  → 移除 sampler 逻辑

v1/sample/
│
├── sampler.py              (nn.Module: greedy + top-k/top-p)
├── logits_processor.py     (penalties, logit bias)
└── metadata.py             (SamplingMetadata)
```

流程：

```
logits

↓

processor

↓

sample

↓

token
```

---

当前能力：

✅ Model forward和sampling分离

---

# C025

# C025 Output Processing Pipeline

对应：

```
v1/engine/output_processor.py

v1/engine/detokenizer.py
```

拆：

```
Engine

↓

OutputProcessor

↓

Detokenizer
```

---

当前能力：

✅ streaming输出基础

---

# C026

# C026 Input Processing Pipeline

对应：

```
v1/engine/input_processor.py
```

现在：

```
API

↓

tokenizer

↓

Request
```

改：

```
Prompt

↓

InputProcessor

↓

Request
```

---

当前能力：

✅ API薄化

---

# C027

# C027 Async Runtime Scheduler

对应：

```
v1/core/sched/async_scheduler.py
```

替换：

```
queue.Queue + sleep
```

实现：

```
request arrive

↓

async queue

↓

schedule
```

---

当前能力：

✅ 真async runtime

---

# C028

# Executor Abstraction

对应：

```
executor/
```

新增：

```
abstract.py

multiproc_executor.py
```

接口：

```python
execute_model()
```

---

当前能力：

✅ executor扩展点

---

# C029

# Prefix Cache

对应：

```
v1/core/kv_cache_manager.py   (get_computed_blocks)
v1/core/block_pool.py         (cached_block_hash_to_block + get_cached_block)
```

依赖：

C018 物理 KV 和 C019 BlockTable。

增强：

```
v1/core/block_pool.py:
  + cached_block_hash_to_block: dict[bytes, KVCacheBlock]
  + get_cached_block(block_hash) → KVCacheBlock | None
  + cache_full_blocks(request, blocks, ...)

v1/core/kv_cache_manager.py:
  + get_computed_blocks(request) → (KVCacheBlocks, num_new_computed_tokens)

v1/request.py:
  + block_hashes: list[bytes]
```

不是新增独立 prefix_cache.py — vLLM V1 将 prefix cache 集成到 BlockPool 的 hash→block 映射和 KVCacheManager 的 computed_blocks 查询中。

实现：

```
prompt hash
  ↓
cached blocks (BlockPool.get_cached_block)
  ↓
reuse KV (KVCacheManager.get_computed_blocks, 计入 num_new_computed_tokens)
```

---

当前能力：

✅ Prefix caching

---

# C030

# Chunked Prefill

对应：

```
v1/core/sched/scheduler.py    (schedule() 中 token-budget 分块)
v1/request.py                  (is_prefill_chunk 属性)
```

不是新增独立 chunked_prefill.py — vLLM V1 将 chunked prefill 作为 Scheduler 内部的调度策略。

增强：

```
v1/core/sched/scheduler.py:
  schedule() 中引入 token_budget:
    - 每次 prefill 最多分配 max_num_batched_tokens 个 token
    - 长 prompt 分多个 chunk
    - 每 chunk 后插入 decode step

v1/request.py:
  + is_prefill_chunk → bool    (还有未处理的 prompt token)
```

实现：

```
long prompt (4096)

chunk1: 512 tokens
  ↓
decode step (穿插 running requests)
  ↓
chunk2: 512 tokens
  ↓
decode step
  ↓
...
```

---

当前能力：

✅ vLLM调度模型

---

# C031

# C031 Runtime Completeness Alignment

最后收尾：

```
迁移 server/api_server.py → server/   (或替换为 FastAPI 版)
重构 metrics/ → v1/metrics/          (prometheus, stats)

新增：
v1/serial_utils.py                   (MsgpackEncoder/Decoder, IPC 支持)
```

说明：

`exceptions.py` 已在 C015 创建，不重复。
`AsyncLLM` 已在 C027 完成异步改造，不重复。
`OpenAI API` C014 已有 `server/api_server.py`，本阶段可选升级为 FastAPI。

---

# 最终 C031 架构

```
v1/

engine/

 EngineCore
 InputProcessor
 OutputProcessor


core/

 sched/
   Scheduler
   SchedulerOutput

 KVCacheManager
 BlockPool


worker/

 GPUWorker

 GPUModelRunner

 InputBatch

 RequestState


attention/

 AttentionBackend

 PagedAttention


sample/

 Sampler


executor/

 Executor

 UniProcExecutor

 MultiProcExecutor
```

---




并满足：

```
Control Flow ≈ vLLM V1

Data Flow ≈ vLLM V1

Component Boundary ≈ vLLM V1
```

但是保留：

```
Kernel

Distributed

CUDA Graph

Spec Decode

Quantization

Production Reliability
```

这些工程规模差距。


但是：

只能通过commit逐渐出现。

---

# 禁止行为

禁止：

1. 输出大量架构分析

2. 输出Mermaid图

3. 输出学习路线

4. 一次输出多个Commit

5. 生成不能运行的代码

6. 提前引入未来模块

7. 直接复制vLLM

---

# 开始

现在只生成：

Commit 015

不要生成Commit016以后内容。





---

这个版本和你之前最大的区别：

- **不是让 Codex “设计一个 mini-vLLM”**
- **不是让 Codex “按模块实现 vLLM”**
- 而是让它模拟：

> “一个工程师写着写着遇到问题，于是代码自然长成 vLLM V1 的过程”

这样你后面用 Git 看：

```bash
git log --oneline
````

看到的才会像真实项目：

```
001 runtime
002 generation
003 kv cache
004 request
...
```

而不是：

```
add scheduler
add worker
add executor
```

这种教学项目。

模型：opt-125m

关于vllm最终版本参数：

|参数|推荐值|原因|
|---|---|---|
|gpu-memory-utilization|**0.90**|3B 比较宽裕，可以适当提高|
|max-model-len|**4096**|足够观察 KV Cache，避免浪费显存|
|max-num-batched-tokens|**4096**|Batch 收益明显|
|max-num-seqs|**32**|Scheduler 调试方便|
|block-size|**16**|与 vLLM 默认一致|
|dtype|**bfloat16**（优先）/ float16|BF16 更稳定|
|tensor-parallel-size|**1**|单卡开发|
|pipeline-parallel-size|**1**|单卡开发|
|enforce-eager|**True**（开发）|方便调试|
|enable-prefix-caching|**False**（开发）|避免影响观察 Prefill/KV Cache|
|cpu-offload-gb|**0**|不需要|
|quantization|**None**|全量模型|
|kv-cache-dtype|**auto**|保持默认|
|attention-backend|**flash_attn**|推荐|



```
vllm/v1/
├── engine/          (14 files)
│   ├── core.py              ← EngineCore — 核心调度循环
│   ├── core_client.py       ← EngineCore 客户端抽象
│   ├── llm_engine.py        ← LLMEngine — 外部入口
│   ├── async_llm.py         ← AsyncLLM — async 包装
│   ├── coordinator.py       ← 协调多进程 / DP
│   ├── detokenizer.py       ← token→text
│   ├── output_processor.py  ← 输出后处理
│   ├── input_processor.py   ← 输入预处理
│   ├── logprobs.py / parallel_sampling.py / tensor_ipc.py
│   └── exceptions.py / utils.py
│
├── core/            (7 + sched/)
│   ├── block_pool.py               ← KVCacheBlock 池
│   ├── kv_cache_manager.py         ← KV cache 分配/释放
│   ├── kv_cache_coordinator.py     ← 异构 KV cache 协调
│   ├── kv_cache_utils.py / kv_cache_metrics.py
│   ├── single_type_kv_cache_manager.py
│   ├── encoder_cache_manager.py
│   └── sched/
│       ├── scheduler.py            ← 核心调度器
│       ├── async_scheduler.py      ← async variant
│       ├── interface.py / output.py / request_queue.py / utils.py
│
├── worker/          (30+ files)
│   ├── worker_base.py              ← Worker 抽象
│   ├── gpu_worker.py / cpu_worker.py / xpu_worker.py
│   ├── gpu_model_runner.py         ← GPU ModelRunner
│   ├── cpu_model_runner.py
│   ├── block_table.py              ← BlockTable (per-request)
│   ├── gpu/
│   │   ├── model_runner.py         ← 实际 GPU 推理执行
│   │   ├── input_batch.py          ← 批输入构建
│   │   ├── block_table.py / attn_utils.py / states.py
│   │   ├── model_states/           ← 模型状态管理
│   │   ├── sample/                 ← GPU 端采样
│   │   └── spec_decode/            ← GPU 端推测解码
│   └── workspace.py / startup_plan.py / utils.py
│
├── executor/        (9 files)
│   ├── abstract.py          ← Executor 接口
│   ├── uniproc_executor.py  ← 单进程 Executor
│   ├── multiproc_executor.py← 多进程 Executor
│   └── ray_executor.py / ray_executor_v2.py / ray_utils.py
│
├── attention/       (30+ files)
│   ├── backend.py / selector.py
│   ├── backends/            ← 各种 attention 后端
│   │   ├── flash_attn.py / flashinfer.py / flex_attention.py
│   │   ├── triton_attn.py / mamba_attn.py / ...
│   │   └── mla/             ← MLA 专用 (DeepSeek)
│   └── ops/                 ← 手写 CUDA/Triton kernel 操作
│       ├── paged_attn.py / triton_decode_attention.py
│       ├── chunked_prefill_paged_decode.py
│       └── ...
│
├── sample/          (sampler + logits_processor + ops)
├── spec_decode/     (推测解码 proposer)
├── kv_offload/      (KV cache CPU offload)
├── structured_output/
├── metrics/
├── pool/            (embedding pooling)
├── request.py
├── outputs.py
├── kv_cache_interface.py
└── serial_utils.py / utils.py
2. vllm/model_executor 完整树 (45 dirs, 464 files)
model_executor/
├── layers/
│   ├── attention/          ← Attention 层实现 (MLA, cross, encoder, etc.)
│   ├── linear.py           ← 各类 Linear (含 TP Column/RowParallelLinear)
│   ├── fused_moe/          ← MoE 融合 kernel
│   ├── mamba/              ← Mamba mixer
│   ├── quantization/       ← 量化方案 (fp8, gptq, awq, marlin, ...)
│   ├── rotary_embedding/   ← 所有 RoPE 变体
│   ├── layernorm.py / activation.py / conv.py / mla.py / mhc.py
│   └── vocab_parallel_embedding.py
│
├── models/                 ← 150+ 模型适配
│   ├── llama.py / qwen2.py / deepseek_v2.py / mistral.py / ...
│   └── transformers/       ← 通用 Transformers 模型实现
│
├── kernels/                ← 手写 CUDA/Triton kernels
│   ├── linear/             ← 各类 quantized linear kernels
│   └── mhc/                ← Multi-Head Convolution kernels
│
├── model_loader/           ← 模型加载 (sharded, tensorizer, etc.)
├── offloader/              ← GPU-CPU offload
├── warmup/                 ← 预热 (JIT, CUDA graphs)
└── parameter.py / utils.py / custom_op.py
```


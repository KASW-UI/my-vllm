核心不是“生成一个简化 vLLM 教程”，而是：

> **让 Codex 基于真实 vLLM V1 最终代码形态，反向生成一个由 Git commit 驱动的、问题演进式的 LLM Serving Runtime 开发历史。**

项目位置：/home/kasw/文档/GitHub/GPU-Project/my-vllm
vllm-v1位置：/home/kasw/文档/GitHub/GPU-Project/vllm

---


# Role

你是一名资深 LLM Serving Runtime 工程师。

你的任务不是写教程。

你的任务是：

模拟一个工程团队从零开发一个 LLM 推理服务框架的真实 Git 演进过程。


最终目标：

通过连续 Git Commit，
逐步演化出一个结构接近 vLLM V1 的简化版 LLM Serving Engine。


---

# Reference Source

唯一最终参考：

vLLM V1源码。


版本：

```
commit:

190be7dad
```


branch:

```
main
```


参考范围：

```
vllm/v1/

vllm/model_executor/

```

包括：

- EngineCore
- Scheduler
- Request
- KV Cache Manager
- BlockPool
- ModelRunner
- Worker
- Executor
- Attention Backend
- Tensor Parallel


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


---

# 核心开发原则


## Principle 0

vLLM V1：/home/kasw/文档/GitHub/GPU-Project/vllm
每次 Commit必须参考/home/kasw/文档/GitHub/GPU-Project/vllm实现，在 Commit 014 完成后：
- 核心控制流与 vLLM V1 对齐
- 核心数据流与 vLLM V1 对齐
- 主要组件边界与 vLLM V1 对齐
Commit 014 之后不再新增核心架构
每个 Commit 必须是真实 Git commit，而不是阶段文档。
每个 Commit 后保留一个可运行状态。
后续 Commit 必须基于前一个 Commit 的代码修改，而不是重新生成项目。


## Principle 1

不要按照模块实现。


错误：



```
Model
↓
KV Cache
↓
Scheduler
↓
Worker
↓
Executor
```




正确：

按照工程问题演进。


例如：



```
能跑模型
↓
需要生成token
↓
生成太慢
↓
引入KV Cache
↓
无法管理多个请求
↓
引入Request
↓
GPU利用率低
↓
Batch
↓
动态请求混乱
↓
Scheduler
```




---

# Principle 2

每个 Commit 都必须是一个完整版本。


不是：

"实现Scheduler模块"


而是：

"当前系统遇到了请求管理混乱问题，因此抽离Scheduler"


---

# Principle 3

不要提前设计未来结构。


例如：

Commit 001:

禁止出现：



```
scheduler.py
worker.py
executor.py
block_manager.py
```




因为此时没有需求。


---

# Principle 4

每个 Commit 必须保留缺陷。


例如：

KV Cache版本：

允许：



每层一个连续tensor



禁止：

马上实现：



```
BlockPool
BlockTable
PagedAttention
```




因为碎片问题还没有出现。


---

# Development Output Format


整个项目由 Git Commit 组成。


共13个 Commit。


每次只生成一个 Commit。


不要一次生成全部。


第一次输出：

只生成 Commit 001。


之后我输入：

继续 Commit 002


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



实现最基础的单请求LLM推理。




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


Commit002:

当前只能forward，
无法完成token-by-token生成，
因此需要引入generation loop。


---

# Evolution History

严格按照下面顺序。

注意：

这是问题演进顺序，

不是模块完成顺序。

---

# Commit 001

# Minimal Runtime

## 目标

运行 HuggingFace LLM，完成最小推理闭环。

## 新增

```
runtime/
├── tokenizer.py
├── model_loader.py
├── model.py
└── main.py
```

实现：

- tokenizer加载
    
- checkpoint加载
    
- model forward
    
- logits输出
    

执行链：

```
prompt
 |
tokenizer
 |
input_ids
 |
model.forward()
 |
logits
```

## 参考 vLLM

```
model_executor/models
model_loader
```

## 当前能力

✅ 单prompt  
✅ 单次forward

## 缺陷

❌ 无文本生成  
❌ 无decode loop  
❌ 无状态保存  
❌ 无request概念  
❌ 无batch

不要：

- scheduler
    
- worker
    
- executor
    

---

# Commit 002

# Generation Loop

## 问题

forward只能得到一次logits，无法生成文本。

## 新增

```
generation/
└── sampler.py
```

实现：

- autoregressive loop
    
- greedy sampling
    
- token append
    

执行：

```
input_ids

while not EOS:

    forward()

    logits

    sample()

    append token
```

## 参考

```
vLLM sample
```

## 当前能力

✅ 单请求生成文本

## 缺陷

❌ 每步重复计算历史token

---

# Commit 003

# KV Cache

## 问题

decode阶段重复计算attention历史。

## 新增

```
cache/
└── kv_cache.py
```

实现：

```
Prefill:

prompt
 |
attention
 |
K,V cache


Decode:

new token
 |
attention + KV cache
```

实现：

- prefill
    
- decode
    
- KV storage
    

使用：

连续内存KV cache。

## 不实现

- block
    
- paging
    
- allocator
    

## 当前能力

✅ 单请求高效decode

## 缺陷

❌ cache无法管理多个请求

---

# Commit 004

# Request Object

## 问题

runtime只能处理一个隐含prompt。

## 新增

```
engine/
└── request.py
```

定义：

```
Request

id
prompt
input_ids
output_tokens
status
```

状态：

```
WAITING

RUNNING

FINISHED
```

## 当前能力

✅ 请求生命周期存在

## 缺陷

❌ 一次只能执行一个request

---

# Commit 005

# Static Batch

## 问题

GPU无法充分利用。

## 新增

```
batch/
└── batch_executor.py
```

实现：

```
Request[]

      |
      v

Batch

      |
      v

model.forward(batch)
```

支持：

- batch input
    
- batch logits
    

## 当前能力

✅ 多request同时forward

## 缺陷

静态batch：

```
request A: 100 tokens

request B: 10 tokens

必须等待A
```

问题：

- padding浪费
    
- 无动态加入退出
    

---

# Commit 006

# Dynamic Generation State

## 问题

静态batch无法处理不同生命周期request。

例如：

```
Request A
finish at step 10


Request B
finish at step 100
```

## 新增

```
engine/
├── waiting_queue.py
├── running_queue.py
└── state_manager.py
```

管理：

```
WAITING

    |
    v

RUNNING

    |
    v

FINISHED
```

实现：

- waiting request管理
    
- running request管理
    
- finished request回收
    
- 动态batch更新
    

## 当前能力

✅ 动态request生命周期

## 缺陷

scheduler逻辑仍混在engine：

```
if waiting:
    add

if running:
    continue

if finished:
    remove
```

---

# Commit 007

# Scheduler

## 问题

Engine职责膨胀：

```
Engine

├── request管理
├── batch构造
├── 状态转换
├── 调度策略
```

## 新增

```
scheduler/
└── scheduler.py
```

抽离：

```
Engine

 |
 Scheduler

 |
 Batch
```

实现：

接口：

```python
schedule()

add_request()

finish_request()
```

策略：

- FCFS
    
- max batch size
    

参考：

```
vLLM v1 core/sched
```

## 当前能力

✅ request调度独立

## 缺陷

❌ 不知道KV cache资源

---

# Commit 008

# KV Cache Manager

## 问题

Scheduler决定执行request，但是不知道显存是否足够。

## 新增

```
cache/
├── kv_cache_manager.py
└── allocator.py
```

实现：

```
Scheduler

      |
      v

KVCacheManager

      |
      v

Can allocate?
```

新增：

- allocate
    
- free
    
- request-cache绑定
    

## 当前能力

✅ scheduler感知资源

## 缺陷

❌ 连续cache导致碎片

---

# Commit 009

# Paged KV Cache

## 问题

连续KV cache：

```
request A
████████

request B
████

释放后:

██  ██
```

产生碎片。

## 新增

```
cache/
├── block.py
├── block_table.py
├── block_pool.py
└── allocator.py
```

实现：

- block
    
- block table
    
- block allocation
    

参考：

```
vLLM block_pool.py
kv_cache_manager.py
```

## 当前能力

✅ 多请求共享KV显存

---

# Commit 010

# ModelRunner

## 问题

Engine负责执行细节。

拆分：

```
Engine

Scheduler

ModelRunner
```

新增：

```
worker/
└── model_runner.py
```

负责：

- input preparation
    
- model forward
    
- logits处理
    

参考：

```
gpu_model_runner.py
```

---

# Commit 011

# Worker

## 问题

Model执行环境需要隔离。

新增：

```
worker/
├── worker.py
└── worker_base.py
```

负责：

- device初始化
    
- memory管理
    
- model生命周期
    

---

# Commit 012

# Executor

## 问题

多个Worker无法统一管理。

新增：

```
executor/
├── executor.py
└── gpu_executor.py
```

实现：

```
Executor

 |
 +-- Worker0

 |
 +-- Worker1
```

---

# Commit 013

# Tensor Parallel

## 问题

单GPU无法加载大模型。

新增：

```
distributed/

layers/
├── column_parallel_linear.py
└── row_parallel_linear.py
```

实现：

- tensor parallel
    
- NCCL通信
    

参考：

```
model_executor/layers/linear.py

distributed
```

---

# Commit 014

# Production Runtime

## 问题

研究runtime无法服务。

新增：

```
server/
├── api_server.py
├── async_engine.py

metrics/
```

实现：

- Async API
    
- OpenAI compatible API
    
- CUDA Graph
    
- metrics
    
- monitoring
    



---

# Coding Constraints

## venv

使用：

/home/kasw/workspace/gpu-infra/.venv

## Model

优先：

HuggingFace模型。

不要自己实现完整Transformer。

---

# Architecture Constraints

最终代码应该逐渐形成：


```
Engine

    Scheduler

    Executor

        Worker

            ModelRunner

                Model

                Attention

                KV Cache
```


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

Commit 001

不要生成Commit002以后内容。





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

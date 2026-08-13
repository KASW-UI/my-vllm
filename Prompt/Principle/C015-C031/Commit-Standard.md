明确要求：

1. **一个 commit = 一个架构演进阶段**
2. **每个 commit 必须可运行**
3. **必须生成阶段快照**
4. **必须打 tag**
5. **禁止跨阶段修改**
---


# Git Commit Development Rules

这个项目不是普通软件开发，而是用于学习 vLLM Runtime 演进。

请严格按照以下 Git 规范提交代码。

---

## 1. Commit 粒度

每个 commit 必须代表一个明确的 Runtime 演进阶段。

规则：

一个 commit 只能解决一个核心问题。

禁止：

- 一个 commit 同时引入多个架构变化
- 提前实现未来阶段功能
- 为了方便重构整个项目


例如：

正确：

```
C003: Add continuous KV Cache
```

只包含：

- KV Cache storage
- prefill/decode separation


错误：

C003:

- KV Cache
- Scheduler
- Worker
- Executor


---

# 2. Commit 命名规范

格式：

```
Cxxx: Module: Description
```


例如：

```
C001: Runtime: load HuggingFace model

C002: Generation: add autoregressive decoding

C003: KVCache: add continuous KV cache

C005: Batch: add static batching

C007: Scheduler: extract scheduler component
```


---

# 3. 每个 Commit 必须包含阶段文档

每完成一个 commit，新增：

```
docs/snapshots/Cxxx.md
```


格式：


# Cxxx: Title


## Goal

这个阶段解决的问题。


## New Features

新增能力。


## File Tree

当前完整项目结构。


例如：

```

src/

├── engine/
├── cache/
├── model/

````


## Runtime Flow

使用 mermaid 描述当前执行流程。


## Current Capability

当前支持：

- xxx


## Current Limitation

明确还缺少：

- xxx


## Next Stage

下一阶段解决的问题。



---

# 4. 每个 Commit 必须保持可运行

提交前必须验证：

```bash
python examples/run.py
````

或者：

```bash
pytest
```

确保：

```
git checkout Cxxx
```

后：

项目仍然可以运行。

---

# 5. 阶段 Tag

每个架构阶段完成后创建 tag。

格式：

```
C001-runtime
C002-generation
C003-kvcache
C004-request
...
```

例如：

```bash
git tag C003-kvcache
```

---

# 6. 不允许修改历史阶段

完成：

C003

之后：

进入：

C004

禁止：

为了 C004 方便：

修改 C003 的设计。

如果发现问题：

新增：

C004 fix commit。

---

# 7. 每个阶段必须输出

提交代码后，同时输出：

1. git commit message

2. 修改文件列表

3. 当前文件树

4. 当前 Runtime 架构图

5. 当前支持能力

6. 当前限制

7. Benchmark结果

---

# 8. 项目目标

最终形成类似：

```
C001 Minimal Runtime

↓

C002 Generation

↓

C003 KV Cache

↓

C004 Request

↓

C005 Static Batch

↓

C006 Dynamic State

↓

C007 Scheduler

↓

C008 KV Manager

↓

C009 Paged KV Cache

↓

C010 ModelRunner

↓

C011 Worker

↓

C012 Executor

↓

C013 Tensor Parallel

↓

C014 Production Runtime

```
每一个阶段都必须能够独立查看和运行。

---

另外还有一个关键点：

> **不要直接 squash commits。保持线性 Git history。**

禁止 squash。
保持每个 Cxxx commit 独立存在，因为 Git history 本身就是学习材料。
你的 Git 最终应该像一本“推理系统架构进化教材”，而不是普通项目提交记录。

# Loop 补齐计划：EngineCore busy loop + 请求队列解耦

> 目标：把 my-vllm 从「调用者同步驱动（`generate()` 里 `while has_work(): step()`）」升级为 vLLM V1 的「EngineCore 常驻 busy loop + input/output 队列解耦 + EngineCoreClient 抽象」。
> 对应 vLLM `v1/engine/core.py` 的 `run_busy_loop` / `_process_input_queue` / `_process_engine_step`、`v1/engine/core_client.py` 的 `EngineCoreClient`、`v1/engine/llm_engine.py` 的 client 接线。
> 编号：从当前 HEAD `C123` 继续，`C124` 起。每步 = 1 commit，须 `python run.py` / `pytest` 通过。
>
> **参考基准**：真实 vLLM V1 源码位于 `/home/kasw/文档/GitHub/GPU-Project/vllm`。下文每步的「参考 vLLM 文件」给出精确路径 + 关键符号 + 提取范围，所有新增代码须从中直接提取，不发明 API。

---

## 0. 现状诊断（loop 为什么还没形成）

| 缺失 | 证据 | 后果 |
|---|---|---|
| EngineCore 是「被动一步执行器」 | `my_vllm/v1/engine/core.py` 只有 `step()`，无 `run_busy_loop()` | 无常驻循环 |
| loop 由调用者拉 | `my_vllm/v1/engine/llm_engine.py:117` `while self.has_work(): self.step()` | 同步阻塞，请求提交与 step 耦合 |
| LLMEngine 直接持有 EngineCore | `llm_engine.py:40` `self._engine_core = EngineCore(...)` | 无 client 抽象层 |
| AsyncLLM 自带 naive loop | `my_vllm/v1/engine/async_llm.py:68` `run_loop()` 自己 drain queue + step | 不是 vLLM 的 busy loop |
| 无 input/output 队列 | 全仓库无 `input_queue` / `output_queue` | 请求提交与执行不解耦 |

**根本因果链**：没有 `run_busy_loop` + 队列，就没有「常驻 engine 主动推」的 loop；前端只能自己 `while step()` 拉。真实 vLLM 里这一步由 `EngineCoreProc.run_busy_loop`（后台）+ `EngineCoreClient`（前端代理）完成。

---

## 1. 目标架构

```
LLM / api_server（前端）
  → LLMEngine（持有 EngineCoreClient，不再直接持有 EngineCore）
      → EngineCoreClient
          ├─ InprocClient      —— 同步 step（offline，保留现有模式）
          └─ BusyLoopClient    —— 后台线程跑 run_busy_loop（server/async 模式）
              → EngineCore
                  ├─ input_queue   (ADD / ABORT / WAKEUP)
                  ├─ run_busy_loop():
                  │     while _handle_shutdown():
                  │         _process_input_queue()   # drain 请求进 scheduler
                  │         _process_engine_step()   # step → output_queue
                  └─ output_queue  (EngineCoreOutputs)
```

---

## 2. 分阶段计划

### C124 — EngineCore busy loop 原语（地基）

**目标**：给 `EngineCore` 加上 input/output 队列 + `run_busy_loop` + shutdown 生命周期。**本阶段不改变现有 `generate()` 行为**（`step()` 仍可用），只为 loop 提供底座。

**参考 vLLM 文件**：

| 文件 | 关键符号 | 提取范围 |
|---|---|---|
| `vllm/v1/engine/core.py` | `EngineCore` (L98)、`EngineShutdownState` (L989)、`has_work` (L1346)、`is_running` (L1354)、`run_busy_loop` (L1358)、`_process_input_queue` (L1368)、`_process_engine_step` (L1399)、`_handle_shutdown` (L1423)、`_handle_client_request` (L1471) | 去 DP（`engines_running`/`_has_global_unfinished_reqs`）、EEP、`batch_queue`、`aborts_queue`、UTILITY RPC、`idle_state_callbacks`；只保留 ADD / ABORT / WAKEUP 三个请求类型 + 单引擎语义 |
| `vllm/v1/engine/__init__.py` | `EngineCoreRequestType` (L252) | 只保留 `ADD` / `ABORT` / `WAKEUP` 三个成员（去 `START_DP_WAVE` / `UTILITY` / `EXECUTOR_FAILED`） |

**工作项**：
1. 在 `my_vllm/v1/engine/__init__.py`（或 `core.py`）加 `EngineCoreRequestType` 枚举：`ADD` / `ABORT` / `WAKEUP`。
2. `EngineCore.__init__` 增加：
   - `self.input_queue: queue.Queue[tuple[EngineCoreRequestType, Any]]`
   - `self.output_queue: queue.Queue[EngineCoreOutputs]`
   - `self.shutdown_state = EngineShutdownState.RUNNING`
3. `EngineCore` 增加方法（从参考文件直接提取，简化）：
   - `has_work()`：`return self.scheduler.has_requests()`（去 DP 的 `engines_running`/`batch_queue`）
   - `is_running()`：`return self.shutdown_state == EngineShutdownState.RUNNING`
   - `run_busy_loop()`：`while self._handle_shutdown(): self._process_input_queue(); self._process_engine_step()`
   - `_process_input_queue()`：`while not has_work() and is_running(): block on input_queue.get()` → `_handle_client_request`；再 drain 剩余请求（`get_nowait`）
   - `_process_engine_step()`：`outputs = self.step()` → 若 `outputs.outputs` 非空 `output_queue.put_nowait(outputs)`
   - `_handle_client_request(request_type, request)`：`ADD` → `self.scheduler.add_request(request)`；`ABORT` → `self.scheduler.finish_requests(request, FINISHED_ABORTED)`；`WAKEUP` → `pass`
   - `_handle_shutdown()`：`RUNNING`→`True`；`REQUESTED`→按 `shutdown_timeout` 决定 drain/abort 未完成请求，置 `SHUTTING_DOWN`→`False`
   - `shutdown()`：置 `REQUESTED` + `input_queue.put((WAKEUP, None))` 唤醒阻塞的 loop
4. **保持 `add_request()` / `step()` 原签名不变**（本阶段不接队列，Inproc 路径零改动）。

**验证**：单测起 `threading.Thread(target=run_busy_loop)`，`input_queue.put((ADD, req))` → loop 自动 step → `output_queue.get()` 拿到结果 → `shutdown()` 干净退出。`pytest tests/test_C124_busy_loop.py` 通过。

---

### C125 — EngineCoreClient 抽象

**目标**：建立 LLMEngine ↔ EngineCore 之间的 client 层：同步 `InprocClient`（保留现状）+ 线程版 `BusyLoopClient`（跑 busy loop）。

**参考 vLLM 文件**：

| 文件 | 关键符号 | 提取范围 |
|---|---|---|
| `vllm/v1/engine/core_client.py` | `EngineCoreClient` (L71)、`make_client` (L83)、`InprocClient` (L276)（`get_output` L289、`add_request` L297） | 去掉 ZMQ/MP/DP/LoRA/utility/reset/sleep 等；只保留 `add_request` / `get_output` / `abort_requests` / `shutdown`；`make_client` 只支持 `asyncio_mode=False` + 一个 `busy_loop` 开关 |

**工作项**：
1. 新建 `my_vllm/v1/engine/core_client.py`：
   - `EngineCoreClient(ABC)`：抽象 `add_request` / `get_output` / `abort_requests` / `shutdown`
   - `InprocClient(EngineCoreClient)`：直接持有 `EngineCore`；`get_output()` = `engine_core.step()`；`add_request()` = `engine_core.add_request()`（同步，等价现有 `LLMEngine.generate` 的驱动方式）
   - `BusyLoopClient(EngineCoreClient)`：构造时 `threading.Thread(target=engine_core.run_busy_loop, daemon=True).start()`；`add_request()` = `engine_core.input_queue.put((ADD, req))`；`get_output()` = `engine_core.output_queue.get()`
   - `make_client(busy_loop, vllm_config, executor_class, log_stats)`：`busy_loop=False` → `InprocClient`；`busy_loop=True` → `BusyLoopClient`
2. 单测：两种 client 都能跑通 `add_request → get_output` 同一流程。

**验证**：`pytest tests/test_C125_core_client.py` 通过。

---

### C126 — LLMEngine 接入 EngineCoreClient

**目标**：`LLMEngine` 不再直接 `EngineCore(...)`，改持 `EngineCoreClient`。**纯重构，行为不变**。

**参考 vLLM 文件**：

| 文件 | 关键符号 | 提取范围 |
|---|---|---|
| `vllm/v1/engine/llm_engine.py` | `LLMEngine.__init__` (L105 `EngineCoreClient.make_client`)、`add_request` (L276 `self.engine_core.add_request`)、`step` (L304 `self.engine_core.get_output`) | 只改「EngineCore 实例 → client」的持有关系；`step()` 内 `get_output()` 后接 `OutputProcessor.process_outputs()` 的既有流程不动 |

**工作项**：
1. `LLMEngine.__init__` 增加 `busy_loop: bool = False` 参数，用 `EngineCoreClient.make_client(...)` 得到 `self.engine_core`（client），替代 `self._engine_core = EngineCore(...)`。
2. `add_request` / `step` / `has_work` 改为通过 client 调用（`self.engine_core.add_request(...)` / `self.engine_core.get_output()`）。
3. `step()` 里 `req_map = {r.request_id: r for r in self.engine_core.scheduler.running}` 需改为经 client 暴露 `scheduler.running`（或让 InprocClient/BusyLoopClient 暴露一个 `running_requests()` 访问器）。
4. `generate()`（offline 同步）默认 `busy_loop=False` → `InprocClient`，行为与 C123 完全一致。

**验证**：`python run.py` 输出与 C123 完全一致；`pytest` 全绿。

---

### C127 — AsyncLLM + api_server 重接 busy loop

**目标**：`AsyncLLM` 删掉自带的 naive `run_loop`，改走 `BusyLoopClient`；`api_server` 用 `busy_loop=True` 的 LLMEngine。

**参考 vLLM 文件**：

| 文件 | 关键符号 | 提取范围 |
|---|---|---|
| `vllm/v1/engine/async_llm.py` | `AsyncLLM` (L70)、`add_request` (L280)、`generate` (L524)、`output_handler` (L656)、`check_health` (L900) | 接口对齐 `AsyncLLM.generate`；本阶段以「线程版 busy loop + `asyncio.to_thread(client.get_output)`」替代 ZMQ/进程，不引入 `AsyncMPClient` |
| `vllm/v1/engine/core_client.py` | `AsyncMPClient` (L467 附近) | 仅参考其「asyncio + busy loop 后台」语义，不实现 ZMQ |

**工作项**：
1. `AsyncLLM` 改为持有一个 `busy_loop=True` 的 `LLMEngine`（即 `BusyLoopClient`）：
   - `generate()` 提交后 `await asyncio.to_thread(self._engine.get_output_blocking)` 等结果，resolve per-request future
   - 删除 `run_loop` / `_drain_queue` / `_add_to_engine`（`async_llm.py:68/92/98`）
2. `my_vllm/entrypoints/openai/api_server.py`：`LLMEngine` 以 `busy_loop=True` 构造。
3. 保留 `check_health` / `errored` / `dead_error` 接口（`async_llm.py:107/114/118`）。

**验证**：`pytest tests/test_C027_async_scheduler.py` 通过（AsyncLLM 行为不回归）；`examples/server.py` 可起服务。

---

### C128 — 验证 + 收尾

**目标**：证明 loop 已形成，清理旧路径，更新文档。

**参考 vLLM 文件**：

| 文件 | 关键符号 | 用途 |
|---|---|---|
| `vllm/v1/engine/core.py` | `EngineCoreProc` (L995)、`run_busy_loop` (L2024) | 对照：真实 busy loop 在进程内的完整形态（my-vllm 用线程替代，确认语义等价） |

**工作项**：
1. 验证「请求队列解耦」：offline 走 `InprocClient`（同步 step），server/async 走 `BusyLoopClient`（常驻 loop），两者输出一致。
2. 确认 `EngineCore` 有 `run_busy_loop()`（`grep` 可查）；`LLMEngine` 通过 `EngineCoreClient` 而非直接 `EngineCore`。
3. 更新 `AGENTS.md`：`Key architecture facts` 加「EngineCore.run_busy_loop: input_queue → schedule → execute → output_queue」；`Package layout` 加 `v1/engine/core_client.py`。
4. 写 `docs/snapshots/C128.md`（Goal / New Features / File Tree / Runtime Flow / Capability / Limitation / Next Stage）。

**验证**：`python run.py` + `pytest` 全绿；`git grep run_busy_loop` 命中 `core.py`。

---

## 3. 关键设计决策

1. **线程版 busy loop，不做 ZMQ/多进程**：真实 vLLM 的 busy loop 跑在 `EngineCoreProc`（独立进程 + ZMQ）。my-vllm 当前是 uniproc、无 ZMQ 依赖，故用**后台线程**承载 `run_busy_loop`（`BusyLoopClient`）。这是「先形成 loop 的形」，ZMQ/进程解耦留到分布式阶段（P0-9/P0-10 之后）。

2. **保留 InprocClient（offline 同步）**：真实 vLLM 的 offline `LLM()` 本身就用 `InprocClient`（caller-driven `step_fn`），并不跑 busy loop。所以 `run.py` 走 InprocClient 是**忠实**的，不是「没做完」。

3. **`add_request` 双入口**：`EngineCore.add_request()` 保持「直接进 scheduler」（Inproc 用）；队列化由 `BusyLoopClient` 通过 `input_queue.put((ADD, req))` + `_handle_client_request` 完成——与 vLLM 的 `EngineCore`(直接) / `EngineCoreProc`(队列) 分层一致。

4. **范围排除**：DP wave 同步、EEP 伸缩、`aborts_queue` 独立队列、UTILITY RPC、ZMQ socket、`asyncio_mode` 均不在本计划内；只实现 ADD / ABORT / WAKEUP 三个请求类型。

---

## 4. 风险与顺序

| 风险 | 缓解 |
|---|---|
| C124 改 `EngineCore` 破坏现有 `generate()` | `step()`/`add_request()` 签名不动，busy loop 是**新增**方法，Inproc 路径零改动 |
| 线程 + CUDA 的线程安全 | busy loop 线程只做 step 编排；模型前向仍在同一线程串行（与 vLLM Inproc 单线程语义一致） |
| `output_queue.get()` 阻塞导致 AsyncLLM 卡死 | C127 用 `asyncio.to_thread` 避免阻塞事件循环 |
| C126 改持有关系导致 `scheduler.running` 访问失效 | 让 client 暴露 `running_requests()` 访问器，`step()` 内部不再直接 `.scheduler` |
| 一次改太多不可调试 | 严格一 commit 一步，每步可跑（C126 是纯重构、行为不变） |

---

## 5. Done 的定义

- [ ] `EngineCore` 有 `run_busy_loop()` + `input_queue`/`output_queue` + `shutdown()`
- [ ] `LLMEngine` 通过 `EngineCoreClient`（InprocClient / BusyLoopClient）访问 EngineCore
- [ ] `AsyncLLM` 不再自带 naive loop，改走 BusyLoopClient
- [ ] offline（`run.py`）走 InprocClient，server/async 走 BusyLoopClient，输出一致
- [ ] `git grep run_busy_loop` 命中 `my_vllm/v1/engine/core.py`；`git grep "EngineCore("` 不再命中 `llm_engine.py`（已改持 client）

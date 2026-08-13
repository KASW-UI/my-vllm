# Plan-A: Logic Alignment (逻辑对齐)

> 修复内部实现逻辑与真实 vLLM 的偏差。

## 概述

24 个 P0 阶段实现后，教育项目的内部逻辑存在以下偏差：

### 偏差清单

| ID | 偏差 | 真实 vLLM | 教育项目 | 状态 |
|----|------|-----------|---------|------|
| L1 | Config 传播 | VllmConfig 贯穿构造链 | 7-8 个独立参数手动传递 | ✅ 已修复 |
| L2 | WorkerBase | 普通类 | ABC + @abstractmethod | ✅ 已修复 |
| L3 | CachedRequestData | list 索引对齐 | dict[str, X] | ✅ 已修复 |
| L4 | ModelRunnerOutput | list[list[int]] + req_id_to_index | dict[str, list[int]] | ✅ 已修复 |
| L5 | EngineCore 构造 | executor_class: type 内部实例化 | executor 实例外传 | ✅ 已修复 |
| L6 | MsgpackEncoder | msgspec.msgpack + OOB tensor | json 存根 | ⏳ 待处理 |
| L7 | EngineCoreOutput | engine/__init__.py + FinishReason | outputs.py str None | ⏳ 待处理 |
| L8 | Request 字段 | 52 个字段 | 14 个字段 | ⏳ 待处理 |

### 已完成的修复

#### Step 1: Config 传播链（7 文件）

**变更文件：**
- `v1/config/vllm.py` — VllmConfig 添加 `parallel_config` 字段，删除 EngineConfig dataclass（保留别名）
- `v1/worker/worker_base.py` — 删除 ABC，构造签名改为 `(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)`
- `v1/worker/gpu_model_runner.py` — 构造签名改为 `(vllm_config: VllmConfig, device)`
- `v1/worker/gpu_worker.py` — 构造签名改为 `(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)`
- `v1/executor/uniproc_executor.py` — 构造签名改为 `(vllm_config: VllmConfig)`
- `v1/executor/multiproc_executor.py` — 构造签名改为 `(vllm_config: VllmConfig)`，_worker_main 同步
- `v1/core/sched/scheduler.py` — 构造签名改为 `(vllm_config: VllmConfig, kv_cache_manager)`

**可验证：** `python -c "from my_vllm.v1.engine.core import EngineCore; ..."`

#### Step 2: EngineCore 构造对齐（1 文件）

**变更文件：**
- `v1/engine/core.py` — 构造签名改为 `(vllm_config: VllmConfig, executor_class: type[Executor], log_stats)`

内部创建 executor 并调用 `_init_executor()`。

#### Step 3: 输出数据结构对齐（4 文件）

**变更文件：**
- `v1/core/sched/output.py` — CachedRequestData: `new_block_ids: list[list[int] | None]`, `num_computed_tokens: list[int]`, `num_output_tokens: list[int]`
- `v1/outputs.py` — ModelRunnerOutput: `req_id_to_index: dict[str, int]`, `sampled_token_ids: list[list[int]]`
- `v1/core/sched/scheduler.py` — schedule() 构建 index-aligned lists，update_from_output() 使用 index 访问
- `v1/worker/gpu_model_runner.py` — execute_model() 使用 enumerate(cached.req_ids) 访问
- `v1/engine/output_processor.py` — process_outputs() 使用 index 访问
- `v1/engine/core.py` — _collect_metrics() 使用 index 访问

#### Step 4: 入口适配（3 文件）

**变更文件：**
- `run.py` — 使用 VllmConfig + executor_class 模式
- `runtime/main.py` — 同步更新
- `examples/server.py` — 同步更新

### 待处理项目

| ID | 任务 | 预估文件数 |
|----|------|-----------|
| L6 | MsgpackEncoder 真实实现 | 1 文件 (serial_utils.py) |
| L7 | EngineCoreOutput 迁移到 engine/__init__.py | 2 文件 (outputs.py, engine/__init__.py) |
| L8 | Request 字段补全 | 1 文件 (request.py) |

### 验证

```
python run.py        # 离线推理
pytest tests/        # 测试套件
```

# Plan-B: Code Alignment (代码对齐)

> 修复文件路径/模块组织/API 命名与真实 vLLM 的偏差。

## 概述

教育项目的 `my_vllm/` 文件树不是 `vllm/` 的干净子集。

### 核心原则

`my_vllm/` 的每个模块路径 = `vllm/` 的对应路径：
- `my_vllm/config/` → `vllm/config/`
- `my_vllm/distributed/` → `vllm/distributed/`
- `my_vllm/model_executor/` → `vllm/model_executor/`
- `my_vllm/v1/` → `vllm/v1/`

---

## 变更清单

### C1: 目录迁移（3 组 mv + import 刷新）

| # | 当前路径 | 目标路径 | vLLM 对应 |
|---|---------|---------|-----------|
| C1a | `my_vllm/v1/config/` | `my_vllm/config/` | `vllm/config/` |
| C1b | `my_vllm/v1/distributed/` | `my_vllm/distributed/` | `vllm/distributed/` |
| C1c | `my_vllm/v1/model_executor/` | `my_vllm/model_executor/` | `vllm/model_executor/` |

**执行步骤：**
1. `git mv` 移动每个目录
2. 全局替换 import 路径（所有 `.py` 文件）
3. 更新 `v1/config/__init__.py` 为 re-export 桥接层（兼容过渡期）
4. 验证 `python run.py`

### C2: 删除遗留/多余目录

| 目录 | 原因 |
|------|------|
| `my_vllm/batch/` | 遗留存根 |
| `my_vllm/cache/` | 遗留存根 |
| `my_vllm/executor/` | 遗留存根，功能在 v1/executor/ |
| `my_vllm/generation/` | 遗留存根 |
| `my_vllm/scheduler/` | 遗留存根，功能在 v1/core/sched/ |
| `my_vllm/worker/` | 遗留存根，功能在 v1/worker/ |
| `my_vllm/metrics/` (顶层) | 冗余桥接层 |
| `my_vllm/runtime/model.py` | 由 model_executor 覆盖 |
| `my_vllm/v1/config/__init__.py` | (迁移后删除或改为 re-export) |

### C3: 补齐缺失内容

| 文件 | 内容 |
|------|------|
| `v1/engine/__init__.py` | 添加 `EngineCoreRequest`, `EngineCoreOutput`, `EngineCoreOutputs`, `FinishReason` |
| `v1/core/sched/utils.py` | 添加 `check_stop()`, `check_sequence_repetition()` |

### C4: Import 路径全局修正

所有内部 import 改为匹配新路径：

```
my_vllm.v1.config → my_vllm.config
my_vllm.v1.distributed → my_vllm.distributed
my_vllm.v1.model_executor → my_vllm.model_executor
my_vllm.v1.outputs.EngineCoreOutput → my_vllm.v1.engine.EngineCoreOutput
my_vllm.v1.outputs.EngineCoreOutputs → my_vllm.v1.engine.EngineCoreOutputs
```

### C5: Server 入口路径修正

| 当前 | 目标 |
|------|------|
| `my_vllm/server/` (顶层, 含 api_server.py) | `my_vllm/v1/entrypoints/` |

---

## 实施顺序

1. **C1**: 目录迁移（先做，其余依赖新路径）
2. **C2**: 删除遗留目录
3. **C3**: 补齐缺失内容
4. **C4**: 全局 import 路径刷新
5. **C5**: Server 入口路径修正

---

## 变更量估计

| 任务 | 文件数 | 类型 |
|------|--------|------|
| C1 | 3 个目录 + ~20 文件 import 刷新 | mv + 批量文本替换 |
| C2 | 9 个目录 | 删除 |
| C3 | 2 个文件 | 新增内容 |
| C4 | ~30 文件 | import 路径替换 |
| C5 | 1 个文件 | 路径调整 |

---

## 验证

```
python run.py           # 离线推理
python -c "import my_vllm.config; import my_vllm.distributed; import my_vllm.model_executor"
pytest tests/           # 测试套件
```

# Base-on-education-Focus-Vllm

> 将 `my_vllm/` 重构为 `vllm/` 的干净子集。

## 背景

`Base-on-C032-Focus-Career` 计划的 24 个 P0 阶段虽已全部实现，但实现过程中偏离了"vLLM 子集"的原始目标：
- 文件树路径不匹配（config/distributed/model_executor 压在 v1/ 下）
- 构造链使用扁平参数而非 VllmConfig 聚合
- 输出数据结构使用 dict 索引而非真实 vLLM 的 list 索引
- WorkerBase 错误地使用了 ABC

## 两份对齐计划

### Plan-A: 逻辑对齐（Logic Alignment）

修复内部实现逻辑的偏差：
1. Config 传播链：VllmConfig 贯穿所有构造函数
2. WorkerBase 从 ABC 改为普通类
3. 输出数据结构从 dict 索引改为 list 索引
4. EngineCore 接收 executor_class 而非实例

详见 `Plan-A-Logic-Alignment.md`

### Plan-B: 代码对齐（Code Alignment）

修复文件路径/模块组织/API 命名的偏差：
1. 目录迁移：config/distributed/model_executor 移至顶层
2. 删除遗留目录和冗余桥接层
3. 补齐缺失的 `__init__.py` 导出
4. 全局 import 路径修正

详见 `Plan-B-Code-Alignment.md`

### 目标文件树

详见 `Target-FileTree.md`

## 实施顺序

1. Plan-A Step 1-2: Config 传播 + EngineCore 重构
2. Plan-B C1: 目录迁移
3. Plan-A Step 3-4: 输出结构 + 入口适配
4. Plan-B C2-C5: 清理遗留 + 填充缺失

## 相关文档

- `Plan-A-Logic-Alignment.md` — 逻辑对齐（Config 传播、数据结构、类层级）
- `Plan-B-Code-Alignment.md` — 代码对齐（文件树、目录迁移、import 修正）
- `Target-FileTree.md` — 最终目标文件树

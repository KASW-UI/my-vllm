# my-vllm 岗位子集对齐计划

> 目标：最终产物 = 严格对应 **方舟 MaaS 岗位 JD** 的最小干净代码子集。
> 文件树 + 对象逻辑与真实 vLLM 对齐，但**绝不含无关功能、绝不留 dangling 引用**。
>
> 执行目录：`/home/kasw/文档/GitHub/GPU-Project/my-vllm`
> 参考代码（只读，绝不修改）：`/home/kasw/文档/GitHub/GPU-Project/vllm`
> 当前进度：C128（loop 已打通），本计划从"剪枝"重新起锚。

> **本文件是计划，不是执行记录。每个模块都标注了它参考的 vLLM 源文件路径。**

---

## 1. 总原则

1. **判据唯一**：一个模块是否保留，只看它是否支撑 JD 的关键词之一
   （性能 / 成本 / 稳定性、PD 分离、分布式 KV Cache、异构推理、弹性计算、
   GPU/CUDA、分布式系统；加分项：网络/RDMA）。
2. **干净删除**：删一个功能 = 删文件 + 删 import + 删 config 字段 + 删所有引用。
   不留 NoOp/ABC 占位，不留 dangling import —— 杜绝"藕断丝连"。
3. **不因"vLLM 有"就保留**：vLLM 有、但 JD 无关的，一律不进 `my_vllm/`。
4. 每个保留的文件都能一句话说清"它支撑 JD 的哪个关键词"，且能指出对应的 vLLM 源文件。

---

## 2. 最终决策（Keep / Delete）

| JD 关键词 | KEEP（vLLM 对应物） | DELETE（JD 无关） |
|---|---|---|
| 性能/成本/稳定性 | scheduler、metrics、profiler、CUDA graph、speculative、量化 FP8 | — |
| PD 分离 | `distributed/kv_transfer/` + `KVConnectorBase_V1` | — |
| 分布式 KV Cache | prefix-cache、block-hash、kv_connector 传输通路 | — |
| 异构推理 | multi-dtype KV（FP8）、GQA/MQA/MoE 架构、`platforms/cuda`、`config/device.py` | cpu/tpu/xpu/rocm |
| 弹性计算 | `distributed/elastic_ep/` | — |
| GPU/CUDA | attention kernel（paged_attn/triton）、`fused_moe`、`config/kernel.py`、`_custom_ops.py`、`custom_op.py` | — |
| 分布式系统 | `parallel_state`（TP/PP/DP/EP/CP）+ NCCL | — |
| 网络/RDMA（加分） | NIXL connector | — |
| **LoRA（用户指定）** | `lora/` + `config/lora.py` | — |
| **多租户（用户指定：不做）** | — | `config/multi_tenant.py` + scheduler 多租户逻辑 |
| **Structured Output（用户指定：不要）** | — | （当前无文件，永不新增） |
| 无 JD 对应 | — | Mamba/SSM、多模态、pooler/embedding/rerank、encoder-decoder、非 OPT 模型、awq/gptq/marlin/cutlass 等量化后端、entrypoints 业务面 |

**架构覆盖原则**：models/ 只保留 `opt.py`（Dense）。GQA/MQA 通过 `QKVParallelLinear` +
attention metadata/kernel 参数覆盖；MoE 通过 `fused_moe` 可复用层覆盖。
不保留 `llama.py` / `deepseek_v2.py` 骨架。

---

## 3. 阶段 0：剪枝（精确删除清单）

> 执行前先 `grep` 确认引用范围，逐项删除后 `python run.py` + `pytest` 必须通过。

| # | 删除目标 | 参考 vLLM 文件 | 说明 |
|---|---|---|---|
| 0.1 | `config/multi_tenant.py` | 无核心对应（vLLM 多租户是外部 plugin，非 core 文件） | 整文件删（`MultiTenantConfig`/`TenantConfig`） |
| 0.2 | `config/vllm.py` 中的 `multi_tenant_config` 字段 | 参考 `vllm/config/vllm.py`（无此字段） | 删字段 + import |
| 0.3 | `config/__init__.py` 中的多租户导出 | 参考 `vllm/config/__init__.py` | 删 `MultiTenantConfig`/`TenantConfig` 导出 |
| 0.4 | `v1/core/sched/scheduler.py` 多租户逻辑 | 参考 `vllm/v1/core/sched/scheduler.py`（无 multi-tenant budget 段） | 删 `self.multi_tenant_config` + C120 budget 方法（约 L446–536） |
| 0.5 | `model_executor/models/llama.py` | 参考 `vllm/model_executor/models/llama.py` | 删除，GQA 用可复用层覆盖 |
| 0.6 | `model_executor/models/deepseek_v2.py` | 参考 `vllm/model_executor/models/deepseek_v2.py` | 删除，MoE 用 `fused_moe` 覆盖 |
| 0.7 | `distributed/elastic_ep/state.py` | 无对应（vLLM 为 `elastic_state.py` / `standby_state.py` / `elastic_execute.py`） | 与 `elastic_state.py` 重复，删 |

> **0.4 是"干净删除"的样板**：多租户不是独立文件，它渗进了 scheduler 的
> budget 检查逻辑。删的时候必须连同调用点一起清掉，验证无 dangling。

---

## 4. 阶段 1：文件树修正（10 处，invented → 0）

| # | 现状（错误路径） | 参考 vLLM 正确归属 | 动作 |
|---|---|---|---|
| 1 | `v1/attention/metadata.py`（`CommonAttentionMetadata`） | `vllm/v1/attention/backend.py`（L412） | 合并进 `backend.py`，删文件 |
| 2 | `v1/attention/backends/ops/paged_attn.py` + `__init__.py` | `vllm/v1/attention/ops/paged_attn.py` | 目录移到 `attention/ops/` |
| 3 | `v1/attention/ops/triton_slot_mapping.py`（自创名） | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` + `triton_attention_helpers.py` | 删，逻辑并入已有 triton 文件 |
| 4 | `v1/core/kv_connector.py`（`KVConnectorBase_V1`） | `vllm/distributed/kv_transfer/kv_connector/v1/base.py`（L171） | 迁移 + 补 `factory.py` |
| 5 | `v1/sample/logits_processor.py`（单文件） | `vllm/v1/sample/logits_processor/{interface.py,builtin.py,state.py}` | 拆成包 |
| 6 | `model_executor/model_loader.py`（单文件） | `vllm/model_executor/model_loader/{__init__.py,base_loader.py,default_loader.py,utils.py,weight_utils.py}` | 拆成包（并入 P1-2） |
| 7 | `distributed/kv_transfer/kv_connector/v1/nixl/nixl_connector.py` | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` | 改名 `connector.py` |
| 8 | `distributed/elastic_ep/state.py` | 无对应 | 阶段 0.7 已删 |
| 9 | `config/multi_tenant.py` | 无对应 | 阶段 0.1 已删 |
| 10 | `v1/attention/backends/ops/__init__.py` | `vllm/v1/attention/ops/__init__.py` | 随 #2 一起归位 |

> 已确认 `_custom_ops.py` / `model_executor/custom_op.py` **有 vLLM 对应**
> （`vllm/_custom_ops.py`、`vllm/model_executor/custom_op.py`），不属于 invented，保留。

---

## 5. 阶段 2：补齐 P0 关键对象边界

| 缺失模块 | 参考 vLLM 文件 | P0 项 | 逻辑要求 |
|---|---|---|---|
| `v1/core/kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | P0-3 | Manager→Coordinator→Group 层级 |
| `v1/core/single_type_kv_cache_manager.py` | `vllm/v1/core/single_type_kv_cache_manager.py` | P0-3 | 单 group 实现 |
| `v1/core/sched/utils.py` | `vllm/v1/core/sched/utils.py` | P0-5 | scheduler 数据模型辅助 |
| `v1/core/sched/async_scheduler.py` | `vllm/v1/core/sched/async_scheduler.py` | P0-5 | async 接口骨架 |
| `v1/worker/gpu/model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | P0-1 | 与 `gpu_model_runner.py` 的共享 minimal runner 拆分 |

---

## 6. 阶段 3：补齐 Must-Implement 关键通路

| 缺失模块 | 参考 vLLM 文件 | 岗位对应 | 逻辑要求 |
|---|---|---|---|
| `distributed/kv_transfer/kv_connector/v1/base.py` | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | PD / 分布式 KV Cache | `KVConnectorBase_V1` |
| `distributed/kv_transfer/kv_connector/v1/factory.py` | `vllm/distributed/kv_transfer/kv_connector/v1/../factory.py` | 同上 | connector 工厂 |
| `distributed/kv_transfer/kv_connector/v1/example_connector.py` | `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` | 同上 | ≥1 真实 backend |
| `v1/worker/gpu/kv_connector.py` | `vllm/v1/worker/gpu/kv_connector.py` | 同上 | `KVConnector`/`ActiveKVConnector` 包装 |
| `v1/sample/logits_processor/` 包 | `vllm/v1/sample/logits_processor/{interface.py,builtin.py,state.py}` | Sampler 边界 | `LogitsProcessor(ABC)` + `builtin.py` |
| `v1/spec_decode/draft_model.py` | `vllm/v1/spec_decode/draft_model.py` | 性能 | draft→verify→accept 通路 |
| `v1/spec_decode/eagle.py` | `vllm/v1/spec_decode/eagle.py` | 性能 | 同上 |
| `v1/spec_decode/metrics.py` + `utils.py` | `vllm/v1/spec_decode/{metrics.py,utils.py}` | 性能 | 同上 |
| `v1/engine/parallel_sampling.py` | `vllm/v1/engine/parallel_sampling.py` | 多机/并行采样 | 接口对齐 |
| `v1/engine/logprobs.py` | `vllm/v1/engine/logprobs.py` | 并行采样 | 接口对齐 |
| `config/speculative.py` | `vllm/config/speculative.py` | 配置补全 | 字段名对齐 |
| `config/device.py` | `vllm/config/device.py` | 异构 | 字段名对齐 |
| `config/kernel.py` | `vllm/config/kernel.py` | GPU/CUDA | 字段名对齐 |
| `config/attention.py` | `vllm/config/attention.py` | GPU/CUDA | 字段名对齐 |
| FP8 量化路径 | `vllm/model_executor/layers/quantization/{fp8.py,kv_cache.py,utils/fp8_utils.py}` | 成本 / 异构 | KV 或 weight 量化落地一条 |

---

## 7. 阶段 4：分布式多机抽象（加分项，优先级最低）

| 缺失模块 | 参考 vLLM 文件 | 岗位对应 |
|---|---|---|
| `v1/engine/tensor_ipc.py` | `vllm/v1/engine/tensor_ipc.py` | 多机通信 |
| `v1/engine/coordinator.py` | `vllm/v1/engine/coordinator.py` | 多机编排 |
| `v1/worker/gpu/dp_utils.py` | `vllm/v1/worker/gpu/dp_utils.py` | DP |
| `v1/worker/dp_utils.py` | `vllm/v1/worker/dp_utils.py` | DP |
| `v1/worker/gpu/cp_utils.py` | `vllm/v1/worker/gpu/cp_utils.py` | CP |
| `v1/worker/cp_utils.py` | `vllm/v1/worker/cp_utils.py` | CP |
| `distributed/parallel_state.py`（已有，需打通） | `vllm/distributed/parallel_state.py` | TP/PP/DP/EP/CP + NCCL |
| block-hash / prefix-cache ↔ kv_connector 对接 | `vllm/v1/core/{block_pool.py,kv_cache_utils.py}` | 分布式 KV Cache |

> 目标：TP=2 / PP=2 / EP≥2 / CP 各一条端到端可运行通路。

---

## 8. LoRA 专项（用户指定 KEEP）

当前仅 `lora/__init__.py` + `lora/request.py` + `config/lora.py` 骨架。
补齐到 vLLM 结构 + 一条可运行路径：

| 缺失模块 | 参考 vLLM 文件 |
|---|---|
| `lora/layers/base.py` | `vllm/lora/layers/base.py` |
| `lora/layers/base_linear.py` | `vllm/lora/layers/base_linear.py` |
| `lora/layers/column_parallel_linear.py` | `vllm/lora/layers/column_parallel_linear.py` |
| `lora/layers/row_parallel_linear.py` | `vllm/lora/layers/row_parallel_linear.py` |
| `lora/layers/fused_moe.py` | `vllm/lora/layers/fused_moe.py` |
| `lora/lora_model.py` | `vllm/lora/lora_model.py` |
| `lora/model_manager.py` | `vllm/lora/model_manager.py` |
| `lora/worker_manager.py` | `vllm/lora/worker_manager.py` |
| `lora/ops/torch_ops/lora_ops.py` | `vllm/lora/ops/torch_ops/lora_ops.py` |
| `lora/ops/triton_ops/` | `vllm/lora/ops/triton_ops/` |
| `v1/worker/lora_model_runner_mixin.py` | `vllm/v1/worker/lora_model_runner_mixin.py` |

> 范围：Dense Linear 的一条 LoRA 前向路径即可证明理解，不追求全 backend。

---

## 9. 验收标准（每阶段）

1. `grep` 全库无被删模块的 import/引用（dangling = 0）。
2. `python run.py` + `pytest tests/` 通过。
3. Level 1：`my_vllm/` 每个文件路径 ∈ vLLM 文件树（invented = 0）。
4. Level 2：保留文件的 class/method 名、职责、数据归属与 vLLM 一致（代码体可简化）。
5. 每个保留文件可一句话说明它支撑 JD 的哪个关键词，且能指出对应 vLLM 源文件。

---

## 10. 执行顺序汇总

```
阶段 0  剪枝（多租户 / 非 OPT 模型 / 重复文件）   ← 先做
阶段 1  文件树修正 10 处（invented → 0）
阶段 2  P0 对象边界（KV Cache 层级 / scheduler / model_runner）
阶段 3  Must-Implement 通路（KV Connector / Sampler / Speculative / FP8）
阶段 4  分布式多机抽象（TP/PP/EP/CP / tensor_ipc）
LoRA 专项（与阶段 3 并行或穿插）
```

每个阶段 = 一个 commit（沿用 `Cxxx: Module: Description`）+ `docs/snapshots/Cxxx.md` + tag，
commit 后 `python run.py` / `pytest` 必须通过。

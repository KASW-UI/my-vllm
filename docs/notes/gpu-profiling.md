# GPU profiling 笔记 — Nsight Compute / Roofline / Occupancy

> C200 配套笔记。覆盖 JD 硬性 #4「GPU 性能分析经验」所需的方法论：
> 工具链（Nsight Compute / Systems）+ 性能模型（roofline）+ 调优
> 信号（occupancy / register / shared mem）。
>
> 配套脚本：
> - `examples/profile_attention.py` — `unified_attention` wall-clock 基准
> - `examples/profile_cache_write.py` — `triton_reshape_and_cache_flash` 基准

---

## 1. 为什么需要 profiling

在 LLM 推理优化里，"**优化 kernel**" 和 "**优化调度**" 是两条独立
路径，前者需要 GPU 性能分析能力，后者需要分布式 + batching 视角。

`my-vllm` 当前内核热点（按调用频次排序）：

| 热点 | 出现位置 | 主要成本 |
|---|---|---|
| `triton_unified_attention` | 每层 decode/prefill | FLOPs + KV 读带宽 |
| `triton_reshape_and_cache_flash` | 每个新 token 写入 | KV 写带宽 |
| `triton_paged_attn` (vLLM 旧接口，已合并到 unified) | legacy 路径 | — |
| `awq_gemm` / `cutlass_scaled_mm` | Linear 层 (量化路径) | GEMM TFLOPS |

**优化前必须先测量**。本文档统一 profiling 工具与指标，方便 commit
间对照。

---

## 2. 工具链

### 2.1 Nsight Compute (`ncu`)

NVIDIA 官方 kernel profiler。**每个 kernel 调用**出一份 report，含：

- **GPU SpeedOfLight (SOL)** — 实际算力占峰值百分比（≥80% 表示计算满载）
- **Memory Throughput** — 实际带宽占峰值（HBM/GDDR）百分比
- **Compute (SM) Throughput** — SM 占用率（pipeline issue 比率）
- **Achieved Occupancy** — 实际驻留 warp 数 / SM 理论最大
- **Registers Per Thread**, **Shared Memory Per Block** — 资源占用
- **Source Counters** — `SASS` 反汇编关联回 Python/Triton 源码

启动方式：

```bash
# 单 kernel profile
ncu --set basic --target-processes all \
    python examples/profile_attention.py --batch-sizes 8 --max-seq-lens 1024

# 完整段（按 range 过滤）
ncu --set full --kernel-name regex:.*unified_attention.* \
    python examples/profile_attention.py

# 重要 metric 提取
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active \
    python examples/profile_cache_write.py
```

### 2.2 Nsight Systems (`nsys`)

全程序 timeline profiler。**不深入单个 kernel**，但给出：

- kernel 启动开销（launch latency，small kernel 可能 30% 时间是 launch）
- GPU 空闲 gap（CPU 端 Python overhead 暴露）
- NCCL 集合通信时间线
- CUDA stream 同步点

启动方式：

```bash
nsys profile -t cuda,nccl --output=trace_qd \
    python examples/profile_attention.py
nsys stats trace_qd.nsys-rep --report cuda_gpu_kern_sum
```

### 2.3 PyTorch 内置（`torch.cuda.Event`）

精度低于 `ncu`，但**零依赖**、可写进 unit test。本仓库的两个 profile
脚本都用 `torch.cuda.Event` 做基础时序：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(iters):
    fn()
end.record()
end.synchronize()
return start.elapsed_time(end) / iters
```

**建议用法**：
- **回归检测**：CI 里跑 `profile_attention.py`，对 baseline CSV 跑
  ±5% 容差，超阈值报警 → 接入 CI（未来 C232 阶段）。
- **粗粒度瓶颈定位**：哪段 wall-clock 涨了，再上 `ncu` 看哪个
  metric 退化。

---

## 3. Roofline 模型

### 3.1 公式

给定 kernel：
- **算力**（FLOPS）：$F = \text{总浮点操作数}$
- **数据量**（Bytes）：$B = \text{总内存访问字节}$
- **算术强度**（AI）：$\text{AI} = F / B$（FLOP/Byte）
- **运行时间下界**：$T \ge \max(F / \pi_{\text{peak}}, B / \beta_{\text{peak}})$
  其中 $\pi_{\text{peak}}$ 是 GPU 峰值 TFLOPS，$\beta_{\text{peak}}$ 是
  HBM 峰值带宽。

**Roofline 图**：横轴 AI（FLOP/Byte），纵轴可达 TFLOPS；曲线 =
  $\min(\pi_{\text{peak}}, \beta_{\text{peak}} \cdot \text{AI})$。

- AI 低的 kernel（memory-bound）落在斜线段，受带宽限制
- AI 高的 kernel（compute-bound）落在水平段，受算力限制
- 交叉点 = 机器平衡点：$\text{AI}_{\text{machine}} = \pi_{\text{peak}} / \beta_{\text{peak}}$

### 3.2 RTX 4060 Laptop（本机 baseline）

```
Peak FP16 TFLOPS (with sparsity off): ~7.5 TFLOPS  (FP32 ~ 7.5, FP16 ~15 boost)
HBM/GDDR bandwidth: 272 GB/s (GDDR6, 8 Gbps × 16-bit × 8 channels)
Machine balance AI: ~7.5e12 / 272e9 ≈ 28 FLOP/Byte
```

含义：本机 AI ≈ 28 时从 memory-bound 切换到 compute-bound。**对
attention kernel**：

- decode 阶段（seq_len=1）每 token 读全部 KV → AI 极低 → memory-bound
- prefill 阶段（seq_len=2048+）FLOPs 占优 → compute-bound

### 3.3 实际测量指标

`profile_attention.py` 输出 `gflops` 列。对 decode 用例：
- batch_size=8, max_seq_len=1024 时 GFLOPS 数应 **远低于** 机器峰值
  （因为大部分时间是 KV cache 读取）
- batch_size=1, max_seq_len=4096 时同样

`profile_cache_write.py` 输出 `gb_per_s` 列。有效 cache write：
- 应接近 GDDR6 峰值带宽的 60-80%（kernel launch + 校验逻辑会扣掉一部分）
- 显著低于此值 → 内存访问模式不连续 / bank conflict

---

## 4. Occupancy 调优

### 4.1 什么是 Occupancy

Occupancy = 实际驻留 warp 数 / SM 理论最大 warp 数。

- **A100 SM**：最大 64 warps（2048 threads）/ SM
- **RTX 4060 SM**（sm_89）：最大 48 warps（1536 threads）/ SM

**低 occupancy 的影响**：
- 内存延迟隐藏能力下降（一个 warp stall 时无 warp 可切换）
- 访存密集型 kernel 受影响最大

### 4.2 三个约束

占用率受 **三** 个 SM 资源限制：

| 资源 | 约束 |
|---|---|
| Register file | 每个 thread 最多 255 registers（255 × 1536 = ~390KB per SM） |
| Shared memory | SM 通常 100KB（A100）/ 100KB（4090）/ 64KB（4060） |
| Max threads per block | 1024 |

Triton 编译时自动分配 register / shared mem，但有几个可调：

```python
@triton.jit
def kernel(...):
    # num_warps: 控制每个 program 的 warp 数（默认 4）
    # num_stages: 控制 pipeline 深度（影响 shared mem）
    pass

# 调用时：
kernel[grid](..., num_warps=8, num_stages=3)
```

### 4.3 Triton kernel 调优模式

```
1. ncu profile 看 achieved occupancy + SOL
2. 如果 occupancy < 50% → num_warps 调到 8 或 16 重测
3. 如果 SOL memory-bound 满载但 SOL compute 低 → 数据访问模式可优化
4. 如果 SOL compute 满载但 occupancy 高 → 已优化到极限
```

---

## 5. 注意力 kernel 的 roofline 放置

`triton_unified_attention` 一次调用处理一个 (batch, layer, head) 块。
在 vLLM v1 中三种调度：

| 调度 | 进入条件 | AI 估计 | Roofline 区 |
|---|---|---|---|
| Decode-only | max_seqlen_q == 1, 无 chunked prefill | 极低（KV read 多）| memory-bound |
| Prefill-only | 全是 prompt，无 decode | 高（QK^T 算力大）| compute-bound |
| Mixed (chunked prefill) | decode + chunked prefill 同 batch | 中 | 边界 |

PD 分离后，**Prefill 和 Decode 在不同 GPU 实例**跑 → 不存在 mixed
调度；这是 PD 分离的性能收益来源之一。

详见 `docs/notes/...`（后续 C232 笔记集）。

---

## 6. 实战工作流

```
新增 / 优化 kernel
    │
    ├─→ step 1: profile_<name>.py 跑一遍，产 baseline CSV
    │
    ├─→ step 2: 看 gflops / gb_per_s vs 机器 peak
    │          - memory-bound → 检查 coalesced access / block_size
    │          - compute-bound → 检查 register pressure / 算术优化
    │
    ├─→ step 3: ncu profile 看具体瓶颈指标
    │          - SOL 段：标到 peak 的 80%+ 可接受
    │          - achieved occupancy < 60% → 调 num_warps
    │
    ├─→ step 4: 优化 → 再 profile → 记录 delta
    │
    └─→ step 5: 写进 commit message / snapshot
```

---

## 7. 限制与未做

- 本笔记暂未涵盖：
  - 多 GPU NCCL 通信 profile（见 C201 配套 `nccl-architecture.md`）
  - CUDA Graph capture 开销（vLLM V1 关键优化）
  - vLLM scheduler 调度延迟（CPU 端 profiling）
- 这些放在后续 commit 的笔记里。

---

## 8. 参考

- NVIDIA Nsight Compute Documentation: <https://docs.nvidia.com/nsight-compute/>
- NVIDIA Nsight Systems Documentation: <https://docs.nvidia.com/nsight-systems/>
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual
  Performance Model for Multicore Architectures* (2009).
- vLLM `benchmarks/attention_benchmarks/` — 上游 attention benchmark 模板
- vLLM `benchmarks/cutlass_benchmarks/` — CUTLASS scaled MM 基准
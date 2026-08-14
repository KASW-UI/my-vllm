# NCCL Architecture Notes — Ring/Tree AllReduce + NVLink/IB Topology Routing

> C201 配套笔记。覆盖 JD 加分 #2「RDMA 原理 / 分布式通信算子」需要的
> 通信栈理解：NCCL 的算法层（ring / tree）+ 自适应选路（NVLink /
> PCIe / IB）+ plugin 抽象。
>
> 配套脚本：
> - `examples/nccl_benchmark.py` — `all_reduce` / `reduce_scatter` /
>   `all_gather` 带宽 sweep（nccl-tests 风格）
> - `tests/test_C201_nccl.py` — smoke 验证单 GPU 退化路径

---

## 1. 为什么需要 NCCL 笔记

LLM 推理的分布式策略（TP / PP / EP / PD）每一种都依赖一套
集体通信原语：

| 策略 | 主要 collective | 流量特征 |
|---|---|---|
| Tensor Parallel (TP) | `all_reduce` | 每层一次，size = hidden_dim |
| Pipeline Parallel (PP) | `send` / `recv`（P2P） | 1-2 个 tensor / stage |
| Expert Parallel (EP) | `all_to_all` | gate routing 决定 token 路由 |
| Prefill-Decode 分离 (PD) | KV cache transfer | prefill → decode，size = seq × hidden |
| 分布式 KV cache | `all_gather` / `reduce_scatter` | block 级别 |

每一种 collective 在 NCCL 里都映射到 **ring** 或 **tree** 算法，
算法的常数因子和总流量由 world_size 决定。误用 → 带宽退化到
1/N 或延迟翻倍。LLM 推理优化需要把这些常量记牢。

---

## 2. Ring AllReduce

### 2.1 算法

NCCL 的 ring all-reduce 把 ``N`` 个 rank 排成一个环，每个 rank
只跟邻居通信。分两阶段：

1. **scatter-reduce** — 把数据切成 ``N`` 块，每块从源头沿环走
   ``N-1`` 步，每到一个 rank 就 partial-reduce（sum）一次。
   完成后，每个 rank 持有"一个 chunk 的最终结果"。

2. **all-gather** — 每个 rank 沿环广播自己持有的那 chunk，``N-1``
   步后所有 rank 都拿到全部 ``N`` 个 chunk。

每步 1 次 send + 1 次 recv（bidirectional → 1 个 size 的 bus 流量）。
总步数 ``2(N-1)``，每步 1 chunk，chunk = size / N。

### 2.2 流量分析

- **每 rank 输入**：`size` bytes
- **每 rank 输出**：`size` bytes（all-reduce 是 in-place semantics）
- **alg BW**：`(N-1)/N * size / t` （每个 rank 的有效产出）
- **bus BW**：`2(N-1) * size / t` （总线上的实际字节）

例：``N=8, size=64MB`` → alg = 7/8 * 64MB / t ≈ 56 MB / t；
bus = 14 * 64MB / t = 896 MB / t。如果 t = 50ms，bus ≈ 18 GB/s；
这正好是 PCIe Gen4 x16 单向 ~16-32 GB/s 的量级。

### 2.3 大小消息分流

NCCL 内部用消息大小分流：

| size 范围 | 算法 | 原因 |
|---|---|---|
| < 16 KB | tree all-reduce（2.4） | ring 的 latency-bound 不划算 |
| 16 KB ~ 1 MB | ring all-reduce | bus 满载前的过渡段 |
| > 1 MB | ring all-reduce with 拆 chunk | bus BW 主导 |

阈值由 ``NCCL_BUFFSIZE``（默认 4MB）控制，超过就拆。

### 2.4 何时 ring 不是最优

- **world_size 大**：``N-1`` 步每步都要 sync，``N=64`` 时
  延迟是 ``N=8`` 的 8 倍。NCCL 在这种情况下切到双 binary tree。
- **不规则通信**：某些 rank 参与某些 op，某些不参与 → NCCL
  退化为 P2P ``send``/``recv``（无 collective 优化）。
- **消息极小**（< 2 KB）：TCP/NCCL setup latency 主导，ring 的
  ``N-1`` 步同步成本不划算。

---

## 3. Tree AllReduce

### 3.1 算法

Tree 把 ``N`` 个 rank 排成 **二叉树**（每个非叶节点有 2 个 child）：

1. **reduce**：叶子 → 父 partial-reduce（sum），递归向上到 root。
   root 持最终结果。
2. **broadcast**：root → 子 → 孙，递归向下传播。

每条边只走 1 次，**总步数 = ``2 log_2 N``**，比 ring 的
``2(N-1)`` 步少一个数量级（``N=64`` 时：12 vs 126）。

### 3.2 流量分析

- **每 rank 输入 / 输出**：`size` bytes（与 ring 同）
- **bus BW**：`2(N-1) * size / t`（与 ring 同，但时间项 ``t`` 不同）
- **latency**：`O(log N)` vs ring 的 `O(N)`

例：``N=64, size=16KB``：
- ring latency ≈ `63 * (2 * 16KB / 50 GB/s + 5μs NIC roundtrip)`
  ≈ 63 × 5.3μs ≈ 330 μs
- tree latency ≈ `12 * 5.3μs` ≈ 64 μs

→ **小消息 + 大 world** 用 tree 明显赢。

### 3.3 NCCL 的 dual binary tree

NCCL 实际用 **dual binary tree**（两个互补的树在偶/奇 step
交错），把每条链路的利用率压到接近 100%（单 tree 会让父节点的
上行链路成为瓶颈）。

---

## 4. Reduce-Scatter / All-Gather

### 4.1 算法

- **reduce_scatter**：等价于 all-reduce 第一阶段（scatter-reduce），
  每个 rank 最终持有 chunk ``i = rank`` 的部分和。`N-1` 步。
- **all_gather**：等价于 all-reduce 第二阶段。每个 rank 最终持有
  所有 ``N`` 个 chunk。`N-1` 步。

两者总流量都是 `(N-1) * size / N` per rank，**和 ring all-reduce
的 alg BW 一致**。NCCL 把 "all-reduce" 实现为
``reduce_scatter + all_gather`` 两步（DGC paper / Baidu Ring
AllReduce 都用这个分解），原因：

- 某些训练场景只需要其中一个（如 ZeRO optimizer 只需要
  reduce_scatter / all_gather 各一次，分开算两次比合并成
  all-reduce 流量更优）
- TP 推理里 backward 阶段会需要 `reduce_scatter` 单独
  （参数分片场景）

### 4.2 流量分析

| Op | alg BW | bus BW |
|---|---|---|
| `reduce_scatter` | `size / N / t` | `(N-1) * size / N / t` |
| `all_gather` | `size / t` | `(N-1) * size / t` |
| `all_reduce` | `(N-1)/N * size / t` | `2(N-1) * size / t` |

注意 `all_reduce = reduce_scatter + all_gather` 总流量 = `(N-1)/N
* size` + `size` = `(2N-1)/N * size`，但 bus 流量按"两次 collective"
是 `2(N-1) * size`，与单次 all-reduce 一致（因为两次都利用
bus full-duplex）—— 这正是 dual binary tree 的关键。

---

## 5. NCCL 的拓扑选路

NCCL 启动时扫描系统的 PCIe / NVLink / IB 拓扑，建立一张
`topology tree`，每个 collective 在调度时选"通信路径"。

### 5.1 路径优先级（默认）

1. **NVLink**（同 node 内 GPU 直连）
   - Hopper (H100): NVLink 4，900 GB/s 双向 aggregate
   - Ampere (A100): NVLink 3, 600 GB/s
2. **PCIe**（同 node，但跨 socket）
   - PCIe Gen5 x16: 128 GB/s 单向
3. **IB**（跨 node）
   - HDR: 200 Gb/s (25 GB/s)
   - NDR: 400 Gb/s (50 GB/s)
4. **TCP**（fallback，slow path）

NCCL 内部 `PATH` 决策：

```
rank i → rank j 的最优 path =
   同 host 且 NVLink 可达？→ NVLink
   同 host 且 NVLink 不通？→ PCIe (CPU 内存栈过)
   跨 host？→ IB (NET plugin)
```

### 5.2 关键环境变量

```
NCCL_DEBUG=INFO                    # 打印通信路径
NCCL_DEBUG_SUBSYS=ALL              # 含 graph / topo / net 子系统
NCCL_IB_HCA=mlx5_0,mlx5_1          # 用哪几个 IB HCA
NCCL_IB_DISABLE=0                  # 完全禁用 IB
NCCL_P2P_LEVEL=SYS                 # NVLINK > PHB > SYS > PIX
NCCL_NET_PLUGIN=libnccl-net.so    # 自定义 plugin（ibverbs / RoCE）
NCCL_SOCKET_IFNAME=eth0            # TCP/IP 通道
NCCL_BUFFSIZE=4194304              # ring chunk 切分阈值
NCCL_NSOCKS_PERTHREAD=4            # socket 数 / NCCL thread
NCCL_SOCKET_NTHREADS=2             # NCCL thread 数
```

`NCCL_P2P_LEVEL` 控制 NVLink 选择粒度：

| Level | 含义 |
|---|---|
| `NVL` | 仅 NVLink（最严） |
| `SYS` | NVLink → PCIe 跨 CPU socket |
| `PHB` | NVLink → PCIe 跨 PCIe host bridge |
| `PIX` | 任何 PCIe（最宽松，但慢） |

### 5.3 跨节点选路：NCCL_NET_PLUGIN

NCCL 把网络层抽象成 4 个 callback：

```c
typedef struct ncclNet_v8 {
  ncclNetInit_v8_t init;       // 枚举 NIC、创建 communication handle
  ncclNetDevices_v8_t devices; // 列出可用设备
  ncclNetGetProperties_v8_t getProperties;  // 设备能力
  ncclNetListen_v8_t listen;   // 创建 server socket
  ncclNetConnect_v8_t connect; // 客户端握手
  ncclNetSend_v8_t send;       // send-via-RDMA
  ncclNetRecv_v8_t recv;       // recv-via-RDMA
  ncclNetClose_v8_t close;
} ncclNet_v8;
```

plugin 实现这套接口，把 RDMA 设备（mlx5 HCA / RoCE / EFA / …）
封装给 NCCL 上层调用。常见的 plugin：

- **libnccl-net.so** — 默认；走 IB verbs
- **AWS EFA** — `libnccl-net-aws-ofi.so`；在 EC2 上跑
- **Microsoft Azure** — `libnccl-net-azure-ofi.so`
- **Cornelis Networks** — `libnccl-net-cnxk.so`

---

## 6. 在 LLM 推理里的应用

### 6.1 Tensor Parallel (TP) AllReduce

每层 transformer 一次 `all_reduce`（column-parallel Linear 后）：

```
input → Q/K/V (column-parallel) → attention output (row-parallel)
       → all_reduce ← 这里
       → O (column-parallel) → ...
```

hidden=4096, dtype=fp16, layer=32 → 每层 traffic = 4096*2 = 8KB；
``all_reduce`` 触发频次 = num_layers * (forward + backward)。前向
推理（no backward）总 traffic ~ num_layers * 8KB（极小，主要看
延迟）。

### 6.2 Prefill-Decode 分离 (PD)

PD 分离把 prefill 和 decode 放到不同机器。KV cache transfer
是 **all_gather** + **send/recv**：

```
prefill worker  ──(all_gather KV blocks)──> decode worker
                              │
                              └── 每个 decode worker 拿到 prefill
                                  计算的部分 KV cache
```

KV cache size 大（4MB ~ 4GB / request），所以这一段流量是 PD
部署的核心瓶颈。对应到 NCCL：这是 ``send/recv``（点对点 KV
传输），不是 all-reduce。

### 6.3 Distributed KV cache

跨 worker 共享同一份 KV cache 时（e.g. prefix cache sharing
across replicas）：

- **write 时**：`reduce_scatter` 把新写入分散到所有持有者
- **read 时**：`all_gather` 从持有者拼出完整 KV

这俩 op 在 NCCL 里都是 ring 实现，alg BW 一致（4.2）。

---

## 7. 性能数字基准

下面是用 `examples/nccl_benchmark.py` 在不同硬件上跑出来的
典型数字（峰值 bus BW）：

| 硬件 | Bus | 峰值 bus BW (all_reduce) |
|---|---|---|
| A100 8x NVLink 3 (intra-node) | NVLink 3 | ~480 GB/s |
| A100 8x PCIe Gen4 (intra-node) | PCIe 4 | ~90 GB/s |
| 2x A100 + HDR IB (inter-node) | IB HDR | ~22 GB/s |
| H100 8x NVLink 4 (intra-node) | NVLink 4 | ~720 GB/s |
| H100 8x + NDR IB (inter-node) | IB NDR | ~45 GB/s |

**单 GPU 退化（world_size=1）**：bus BW = 0（公式 (N-1)=0），
但 elapsed_us 反映本地 memcpy 的 baseline —— my-vllm 默认
用它做 smoke test。

---

## 8. 调试技巧

### 8.1 启动时打印拓扑

```bash
NCCL_DEBUG=INFO python examples/nccl_benchmark.py 2>&1 | head -50
```

应该看到：

```
NCCL INFO Channel 00 : 0[0] -> 1[1] [send] via NET/0/1
NCCL INFO Channel 01 : 0[0] -> 2[2] [send] via NVL
NCCL INFO Channel 02 : 0[0] -> 3[3] [send] via NVL
...
```

`NET` = IB， `NVL` = NVLink， `PHB` = PCIe host bridge。

### 8.2 找最快路径

```bash
# 只用 NVLink，禁用 IB
NCCL_IB_DISABLE=1 python examples/nccl_benchmark.py

# 只用 IB，禁用 NVLink
NCCL_P2P_LEVEL=0 python examples/nccl_benchmark.py

# 限制特定 HCA
NCCL_IB_HCA=mlx5_0 NCCL_IB_DISABLE=0 python examples/nccl_benchmark.py
```

### 8.3 Profiling

```bash
nsys profile -t cuda,nccl -o /tmp/nccl_profile \
    python examples/nccl_benchmark.py
```

`nsys` 会把所有 NCCL API call + GPU kernel 一起 time-stamp，
可以看到 collective 触发点对应到 model 的哪一层。

---

## 9. 参考资料

- NCCL 官方文档：`https://docs.nvidia.com/deeplearning/nccl/`
- nccl-tests 仓库：`https://github.com/NVIDIA/nccl-tests`
  - `all_reduce_perf.c` — bandwidth sweep 模板
- Patarasuk & Yuan, "Bandwidth Optimal All-Reduce Algorithms
  for Clusters of Workstations" (2009) — ring 算法证明
- Andrew Gibiansky, "Bringing HPC Techniques to Deep Learning"
  (Baidu Silicon Valley AI Lab, 2017) — 实战 ring + tree 选择
- NCCL source: `src/collectives/` (ring, tree, all_gather,
  reduce_scatter 实现)

---

## 10. my-vllm 现状

- **基础设施**：`my_vllm/distributed/parallel_state.py` 提供
  `GroupCoordinator` (NCCL wrapper)；C150 已支持 multi-process
  NCCL end-to-end (`MultiprocExecutor._init_executor`)；
  `init_distributed_environment` + canonical `Worker.init_device` 序列。
- **通信算子**：未独立封装 `all_reduce` / `all_gather` 的
  benchmark 入口。`examples/nccl_benchmark.py` 补齐。
- **测试**：`tests/test_C201_nccl.py` smoke 测试 world_size=1
  退化路径 + multi-process 校验。
- **后续**：C214 引入 `custom_all_reduce.cu` + 同 host 多卡
  IPC 优化，绕过 NCCL 走更快的 ring on shared memory。

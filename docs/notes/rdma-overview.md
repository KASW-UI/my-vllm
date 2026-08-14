# RDMA Overview Notes — Verbs / RoCE / iWARP / NCCL_NET_PLUGIN

> C201 配套笔记。覆盖 JD 加分 #2「RDMA 原理」需要的传输层理解：
> RDMA 语义、Verbs / RoCE / iWARP 三种实现、NCCL 如何接入 RDMA。
>
> 配套脚本：
> - `examples/nccl_benchmark.py` — NCCL collective 带宽基准
>   （背后是 NCCL_NET_PLUGIN → ibverbs → RDMA HCA）

---

## 1. RDMA 是什么

RDMA (Remote Direct Memory Access) 让一台机器的进程直接读写
另一台机器的用户态内存，**不经过对端 CPU**：

```
传统 TCP/IP send：
  src_app → src_kernel (memcpy + TCP stack) → NIC →
  dst_NIC → dst_kernel (TCP stack + memcpy) → dst_app
                                    ^^^^^^^^^^^^^^^^
                                    CPU 拷贝 4 次

RDMA send：
  src_app → src_NIC (DMA from app buffer) →
  dst_NIC (DMA to dst app buffer) → dst_app
                ^^^^^^^^^^^^^^^^^^^
                CPU 0 次拷贝（除了 setup）
```

优势：

- **latency**：user → user ~ 1-2μs（TCP ~ 10-30μs）
- **bandwidth**：kernel bypass → 不占 host CPU 周期；CPU
  可以同时做别的
- **zero-copy**：app buffer → wire → app buffer，中间不复制

代价：

- 需要 RDMA HCA (Host Channel Adapter) 卡（mlx5 / EFA / …）
- 需要配置：内存注册、QP (Queue Pair) 建链、ring buffer
- TCP 套接字接口不复用；要学一套新 API（libibverbs / rdmacm）

---

## 2. RDMA 三种语义

### 2.1 Send / Receive (channel semantics)

```
src:    send_wr → SQ → HCA → wire → dst HCA → RQ → recv CQE → dst app
```

- **双向 channel**：src 必须先 post 一个 send，dst 必须先
  post 一个 recv（receiver-driven）。
- 不需要 dst CPU 预选 buffer —— dst 在 recv 时才决定哪块内存
  接收。
- **缺点**：要求双方同步，复杂。
- **场景**：连接型 message passing（OpenMPI, UCX）。

### 2.2 RDMA Read / Write (memory semantics)

```
write: src HCA 直接 DMA 写入 dst 内存（不需要 dst 参与）
read:  src HCA 直接 DMA 从 dst 内存读（不需要 dst 参与）
```

- **单向**：src 发起即可，dst 完全不参与。
- 需要 dst 提前把目标 buffer **register**（pin 在物理内存）。
- **场景**：大块数据传输、PUT/GET（one-sided RPC）。NCCL
  内部默认走 write（如果有 RDMA write 支持）。

### 2.3 RDMA Atomic

```
HCA 直接执行 dst 内存的 atomic op（fetch-and-add / compare-and-swap）
```

- **场景**：分布式锁、计数器。
- NCCL 不直接用，但深度学习 framework 偶尔会用（e.g. 分布式
  embedding 更新）。

---

## 3. 三种 RDMA 实现：Verbs / RoCE / iWARP

### 3.1 原始 IB Verbs (libibverbs)

```c
#include <infiniband/verbs.h>

struct ibv_context *ctx = ibv_open_device(device);
struct ibv_pd *pd = ibv_alloc_pd(ctx);
struct ibv_mr *mr = ibv_reg_mr(pd, buf, size, IBV_ACCESS_LOCAL_WRITE
                                            | IBV_ACCESS_REMOTE_WRITE);
struct ibv_cq *cq = ibv_create_cq(ctx, qdepth, ...);
struct ibv_qp *qp = ibv_create_qp(pd, &attr);  // 关键：Queue Pair

// 建链（RC — Reliable Connection）
struct ibv_qp_attr attr = {
  .qp_state = IBV_QPS_INIT,
  .pkey_index = 0,
  .port_num = port,
  .qp_access_flags = IBV_ACCESS_REMOTE_WRITE,
};
ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | ...);

// RTR (Ready to Receive)
attr.qp_state = IBV_QPS_RTR;
attr.ah_attr = {...remote QP info...};
ibv_modify_qp(qp, &attr, ...);

// RTS (Ready to Send)
attr.qp_state = IBV_QPS_RTS;
ibv_modify_qp(qp, &attr, ...);

// 提交 send WR
struct ibv_send_wr wr = {
  .wr_id = 0,
  .sg_list = &sge,  // scatter-gather entry
  .num_sge = 1,
  .opcode = IBV_WR_RDMA_WRITE,
  .send_flags = IBV_SEND_SIGNALED,
  .wr.rdma = {.remote_addr = dst_addr, .rkey = dst_rkey},
};
struct ibv_send_wr *bad_wr;
ibv_post_send(qp, &wr, &bad_wr);

// poll CQ 完成
struct ibv_wc wc;
while (ibv_poll_cq(cq, 1, &wc) == 0) {}
```

IB Verbs 提供最完整的 RDMA 语义，**几乎所有高性能网络栈
（UCX, libfabric, NCCL plugin）最终都走这一层**。

### 3.2 RoCE (RDMA over Converged Ethernet)

把 IB 协议封装到普通以太网上：

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ IB Verbs │ →  │ RoCE v2  │ →  │ Ethernet │
│   API    │    │ UDP/IP   │    │  PHY     │
└──────────┘    └──────────┘    └──────────┘
```

- **RoCE v1**：纯 L2 以太网，无 IP（同一子网）。
- **RoCE v2**：UDP/IP，可路由（跨子网）。

优势：HCA 卡可以用普通以太网交换机（不需要 InfiniBand switch），
部署便宜。

代价：需要 lossless Ethernet（启用 PFC + ECN），否则 RoCE
碰到丢包 → 重传 → 性能塌方。

NCCL plugin：`NCCL_IB_HCA=mlx5_0,mlx5_1` 启用 HCA 时，如果是
RoCE 模式，NCCL 自动走 RoCE v2 协议。

### 3.3 iWARP (Internet Wide Area RDMA Protocol)

把 RDMA 封装到 TCP 上：

```
IB Verbs API → iWARP → TCP → IP → Ethernet
```

- 在标准 TCP/IP 上跑，无需 lossless network。
- 但 TCP overhead 大（每包 100+ 字节 header），性能不如 RoCE。
- Mellanox ConnectX 系列都支持 iWARP，但实际部署少。

---

## 4. Queue Pair (QP) 与传输模式

### 4.1 RC (Reliable Connection)

每对 (src, dst) 一个 QP，**1:1 dedicated connection**。

- **优点**：reliable ordering，有最大 BW。
- **缺点**：QP 数量 = `O(N^2)` for `N` ranks（每个 rank 都跟
  所有人连），connect 建立慢。

适用：两个固定 endpoint 间的高速传输（KV cache transfer）。

### 4.2 UC (Unreliable Connection)

类似 RC 但不保证 delivery（如果丢包不回 NAK）。

适用：实时流（视频 / 语音），NCCL 不用。

### 4.3 UD (Unreliable Datagram)

1 个 QP 可以 send 给任意 destination，类似 UDP。

- **优点**：QP 数量 = `O(N)`。
- **缺点**：每个 message 大小受限（典型 MTU ~ 4KB）。
- **场景**：控制平面、discovery（小 message）。

### 4.4 XRC / SRD

XRC (Extended Reliable Connected) 让多个进程共享一个目标
QP，减少 context switch；SRD (Shared Receive Queue) 类似。

NCCL 一般用 RC（小集群）或 XRC（中等集群）。大规模时用
SRQ (Shared Receive Queue) 减少内存占用。

---

## 5. NCCL 如何使用 RDMA

### 5.1 调用栈

```
my_vllm.distributed.parallel_state (Python)
       ↓
torch.distributed.all_reduce(...)   # PyTorch C++ binding
       ↓
c10d::ProcessGroupNCCL::allReduce   # c10d backend
       ↓
ncclAllReduce(...)                  # NCCL C API
       ↓
ncclNet send/recv callbacks         # NCCL plugin 接口
       ↓
ibv_post_send / ibv_post_recv       # libibverbs
       ↓
mlx5 HCA driver → PCIe → wire       # HCA hardware
```

### 5.2 NCCL 选择传输层

启动时 NCCL 选 plugin：

```
NCCL_NET_PLUGIN 指定 → loadlibrary(libnccl-net.so)
                       ↓
                       调用 ncclNetInit()
                       ↓
                       枚举 NIC：检查 mlx5_0, mlx5_1, ...
                       ↓
                       选择最优路径：NVLink > PCIe > IB > TCP
```

### 5.3 Plugin 接口

NCCL 把网络层抽象成 5 个 callback（精简版）：

```c
// nccl_net.h
ncclResult_t ncclNetInit(ncclDebugLogger_t log);
ncclResult_t ncclNetDevices(int *ndev, ncclNetProperties_v8_t **props);
ncclResult_t ncclNetListen(int dev, void *handle, void **listenComm);
ncclResult_t ncclNetConnect(int dev, void *handle, void **sendComm);
ncclResult_t ncclNetSend(void *sendComm, void *data, int size,
                         int tag, void *mhandle, void **request);
ncclResult_t ncclNetRecv(void *recvComm, int n, void **buffers,
                         int *sizes, int *tags, void **mhandle,
                         void **request);
ncclResult_t ncclNetClose(void *comm);
ncclResult_t ncclNetCloseListen(void *listenComm);
```

每个 plugin 实现这套接口。AWS EFA plugin、Azure OFI plugin、
Cornelis plugin 都是这样接入。

### 5.4 Send/Recv 在 NCCL 中的角色

NCCL 的 collective（all_reduce / all_gather 等）最终拆成
点对点的 send/recv 调用：

```
ring all_reduce 第 k 步:
   src_rank.postSend(buf[k], size/N, dst=next_rank)
   dst_rank.postRecv(buf[k], size/N, src=prev_rank)
```

这些 send/recv 走 plugin → ibverbs → HCA。如果是 PCIe/NVLink
路径，plugin 直接用 NCCL built-in P2P transport（不走 ibverbs）。

---

## 6. RDMA 在 LLM 推理里的具体场景

### 6.1 PD 分离的 KV cache transfer

```
prefill worker: 
   buf = KV cache (size = seq × hidden × num_layers)
   postSend(buf, size, dst=decode_worker)
   ← ibverbs → mlx5 → IB HDR/NDR → mlx5 → ibverbs →
decode worker:
   postRecv(buf, size, src=prefill_worker)
   ← 直接 DMA 到 app memory
```

延迟：~5-15μs setup + wire time。1GB KV cache 在 NDR IB
（50 GB/s）上 ~20ms wire。

### 6.2 Tensor Parallel 的 all_reduce

```
TP all_reduce over NVLink (intra-node):
   每个 chunk = hidden_dim * 2 / num_TP_ranks
   通过 NVLink 走 → NCCL 内部用 NVL transport，不走 ibverbs
```

不需要 NIC，但需要 GPU ↔ GPU 直连。NVLink 4 (H100) 提供
900 GB/s aggregate。

### 6.3 EP 的 all_to_all

每个 rank 把自己的 token 按 gate routing 发到对应 expert 所在 rank：

```
all_to_all:
   rank 0 → expert 0,1,2,3
   rank 1 → expert 4,5,6,7
   ...
```

NCCL 通过 ring all_to_all 实现（拆成多轮 P2P send/recv）。
RDMA HCA 上的延迟 ~ 5-10μs。

### 6.4 跨 host PD 部署

最常见的 KV cache transfer 路径：prefill host → decode host，
中间走 IB HDR/NDR。NCCL 的 NCCL_NET plugin 把这段 traffic
走 ibverbs → HCA → IB switch → HCA → ibverbs → dst app。

延迟 profile：

```
component               | latency
------------------------|----
app → HCA (DMA setup)   | 1-2 μs
HCA → IB switch         | 0.5-2 μs (per hop)
IB switch → dst HCA     | 0.5-2 μs
dst HCA → app (DMA)     | 1-2 μs
------------------------|----
total wire latency      | ~5-10 μs (intra-DC)
```

加上 TCP/NCCL setup ~20-50μs。

---

## 7. 在 my-vllm 里的现状

- **C150 multi-process NCCL**：`MultiprocExecutor._init_executor`
  export `MASTER_ADDR` / `MASTER_PORT` / `WORLD_SIZE` env vars，
  然后 `Worker.init_device` 走 NCCL。
- **C201（本 commit）**：`examples/nccl_benchmark.py` 提供带宽基准。
- **C214（后续）**：引入 `custom_all_reduce.cu`，同 host 多卡走
  shared-memory ring（绕过 NCCL），比 IB path 快 ~10x。

---

## 8. 调试 RDMA 问题

### 8.1 常用工具

```bash
# 看 HCA 设备
ibv_devinfo                        # 列出所有 HCA
ibv_devinfo -d mlx5_0              # 看特定 HCA 详情

# 看连接状态
iblinkinfo                         # IB fabric 拓扑
ibstatus                           # 端口状态

# 测试 RDMA 带宽
ib_write_bw                        # RDMA write 带宽基准
ib_send_bw                         # send/recv 带宽基准
```

### 8.2 NCCL 通信问题诊断

```bash
# 打印 NCCL 决策
NCCL_DEBUG=INFO python script.py 2>&1 | grep -E "(NET|Channel|Path)"

# 典型输出：
# NCCL INFO Channel 00 : 0[0] -> 1[1] [send] via NET/0/1
#       (走 NIC 0 的 port 1 = IB HCA 0)
# NCCL INFO Channel 00 : 0[0] -> 2[2] [send] via NVL
#       (走 NVLink)
```

### 8.3 性能异常时排查清单

1. **丢包率高** → IB fabric 问题，查 PFC / ECN
2. **延迟突然翻倍** → 多租户争抢 NIC（QOS）
3. **带宽上不去** → MTU 不对齐（RoCE 默认 1500B，改 9000B）
4. **跨 NUMA 慢** → PCIe root complex 拥塞

---

## 9. 参考资料

- RFC 5040 — RDMA Protocol Verbs Specification
- libibverbs man pages: `ibv_post_send(3)`, `ibv_poll_cq(3)`
- RDMAmojo 博客（性能分析实战）
- NCCL source: `src/transport/`, `src/net/` （plugin 加载）
- Mellanox RDMA Aware Networks Programming User Manual

---

## 10. 一句话总结

RDMA = 把 "copy from app → kernel → wire → kernel → app" 变成
"app → wire → app"。NCCL 通过 NET plugin 抽象 ibverbs / RoCE
/ iWARP，让上层 Python 代码只看到 `all_reduce` 这种高级原语，
但底层走 RDMA HCA 拿到 ~1μs latency 和 ~50-200 GB/s 带宽。
LLM 推理的 PD 分离、TP、EP 都依赖这条管线。

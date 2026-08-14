# `docs/notes/` — my-vllm Engineering Notes Index

> **Purpose**: a single searchable index over the deep-dive notes
> produced by Phase 1 + Phase 2 + Phase 3 of the engineering-alignment
> plan.  Each note is anchored to a specific commit and a specific
> JD keyword; this README gives the top-down view.
>
> Total: 8 notes, ~3500 lines of notes + ~300 lines of indexing
> (this file).  Each note is self-contained but cross-linked here
> so you don't have to grep across `docs/snapshots/Cxxx.md` to find
> the right deep-dive.

---

## 1. Topical map

```
docs/notes/
│
├── README.md                         ← you are here
│
├── GPU architecture & performance
│   ├── gpu-arch-primer.md            (C232) SM / warp / memory hierarchy
│   └── gpu-profiling.md              (C200) Nsight Compute / Roofline / Occupancy
│
├── Communication
│   ├── nccl-architecture.md          (C201) Ring/Tree AllReduce + NVLink/IB routing
│   └── rdma-overview.md              (C201) Verbs / RoCE / iWARP / NCCL_NET_PLUGIN
│
├── Quantization & numerics
│   ├── quantization-landscape.md     (C232) FP8 / INT4 / AWQ / Marlin survey
│   └── cutlass-scaled-mm.md          (C220) CUTLASS scaled MM contract + ecosystem
│
└── Runtime stack
    └── vllm-runtime-overview.md      (C232) vLLM V1 stack top-down walkthrough
```

---

## 2. The 8 notes — one-paragraph summaries

### gpu-arch-primer.md *(C232, NEW)*

**SM / warp / memory-hierarchy primer for the LLM-kernel author.**
Covers SM anatomy (4 warp schedulers, 16-32 lanes each), the
register → shared → L1 → L2 → HBM memory pyramid, tensor-core
throughput matrix (BF16 / FP8 / INT8 / FP4 per SM), the CUDA
execution model (host launches kernels → SMs execute warps →
SIMT lockstep), memory coalescing and shared-mem bank conflicts,
and the occupancy vs ILP tradeoff.  This is the background you
need to read `gpu-profiling.md` or `cutlass-scaled-mm.md` without
flipping back to NVIDIA docs every other paragraph.

### gpu-profiling.md *(C200)*

**Nsight Compute + Roofline + occupancy.**  The methodology for
measuring kernel performance: Nsight Compute setup for our
Triton kernels (`triton_unified_attention`,
`triton_reshape_and_cache_flash`), the Roofline performance
model (peak FLOPS / peak bandwidth → kernel classification
as memory-bound / compute-bound / latency-bound), occupancy
analysis (the warps-per-SM cap and what limits it: registers,
shared memory, block size).  The decision tree for "is my kernel
underperforming" — start at the Roofline, then check occupancy,
then check the SASS for stalls.

### nccl-architecture.md *(C201)*

**NCCL algorithms + adaptive routing.**  How NCCL picks between
ring and tree allreduce / allgather / reducescatter, the
constants behind each (ring: 2(n-1)/n traffic; tree: 2*log(n)),
NVLink / PCIe / IB topology detection, the `NCCL_NET_PLUGIN`
hook for RDMA bypass, and `NCCL_IB_HCA` / `NCCL_SOCKET_IFNAME`
selection knobs.  The bandwidth model and what limits
multi-node scaling.  Companion: `examples/nccl_benchmark.py`
(384 lines: all_reduce / reduce_scatter / all_gather sweep).

### rdma-overview.md *(C201)*

**RDMA transport layer.**  The three RDMA implementations (InfiniBand
Verbs, RoCE / iWARP), why LLM serving needs RDMA (TP/PP/PD
traffic crosses NIC, not NVLink), how NCCL binds to RDMA via
the plugin interface, and the kernel-bypass setup.  Companion:
the NCCL plugin call chain from the application down to the HCA.

### quantization-landscape.md *(C232, NEW)*

**The quantization method zoo.**  Symmetric vs asymmetric,
per-tensor / per-channel / per-token scales, FP8 (E4M3 vs
E5M2 — when each), INT8 (per-channel vs per-token), INT4
(Marlin / AWQ / GPTQ — when each is the right call), activation
quantization (online dynamic vs calibrated static), and the
decision tree for picking a quantization recipe per workload.
The 2-4× throughput story (BF16 → FP8 → INT8 → INT4) with
the accuracy cliff at each step.  Companion:
`cutlass-scaled-mm.md` for the GEMM-side details.

### cutlass-scaled-mm.md *(C220)*

**CUTLASS scaled GEMM contract.**  The vLLM `cutlass_scaled_mm`
ABI (output, A, B, scale_a, scale_b, bias, out_dtype), the
three quantization layouts (per-tensor / per-token /
per-channel), PyTorch's native equivalents (`torch._scaled_mm`
for FP8, `torch._int_mm` for INT8), the CUTLASS-3.x kernel
matrix per SM (sm_80 / sm_89 / sm_90 / sm_100 / sm_120), the
Marlin / Machete / DeepGEMM ecosystem, and the methodology
for picking a quantization recipe per SM target.  Companion:
`examples/quant_gemm_benchmark.py` (356 lines: BF16 vs FP8 vs
INT8 sweep over M × (K, N)).

### vllm-runtime-overview.md *(C232, NEW)*

**vLLM V1 stack top-down walkthrough.**  From HTTP request to
GPU token: EngineCore → Scheduler → InputBatch →
GPUModelRunner → forward → Sampler → OutputProcessor →
Detokenizer.  Each layer annotated with the concrete
`my_vllm/` file and the Cxxx commit that implemented it.
The architectural boundaries (EngineCore ↔ Executor ↔ Worker
↔ Runner ↔ Attention), the performance hotspots
(unified_attention / reshape_and_cache / rotary / moe_gemm),
and the dataflow contract for each layer.

---

## 3. JD keyword → note mapping

| JD keyword                                | Notes                                                                                                  |
|-------------------------------------------|--------------------------------------------------------------------------------------------------------|
| **硬性 #1: GPU/CUDA 底层**                | [gpu-arch-primer.md](./gpu-arch-primer.md), [vllm-runtime-overview.md](./vllm-runtime-overview.md)  |
| **硬性 #4: GPU 性能分析**                | [gpu-profiling.md](./gpu-profiling.md), [gpu-arch-primer.md](./gpu-arch-primer.md)                  |
| **硬性 #5: 分布式系统**                  | [nccl-architecture.md](./nccl-architecture.md), [rdma-overview.md](./rdma-overview.md)              |
| **加分 #1: 性能/成本/稳定性**             | [gpu-profiling.md](./gpu-profiling.md), [cutlass-scaled-mm.md](./cutlass-scaled-mm.md)              |
| **加分 #2: RDMA / 通信**                  | [nccl-architecture.md](./nccl-architecture.md), [rdma-overview.md](./rdma-overview.md)              |
| **加分 #3: 推理优化 (量化)**              | [quantization-landscape.md](./quantization-landscape.md), [cutlass-scaled-mm.md](./cutlass-scaled-mm.md) |
| **加分 #4: PD 分离**                      | [vllm-runtime-overview.md](./vllm-runtime-overview.md) § KV Connector                                |
| **加分 #5: 分布式 KV cache**              | [vllm-runtime-overview.md](./vllm-runtime-overview.md) § Paged KV                                       |
| **加分 #6: 弹性计算**                      | [vllm-runtime-overview.md](./vllm-runtime-overview.md) § Sleep / Wake Up + VMM                        |

---

## 4. Commit → note mapping (reverse)

| Commit | Title                                  | Note(s)                                                    |
|--------|----------------------------------------|------------------------------------------------------------|
| C200   | GPU profiling + roofline                | [gpu-profiling.md](./gpu-profiling.md)                    |
| C201   | NCCL + RDMA benchmarks + notes          | [nccl-architecture.md](./nccl-architecture.md), [rdma-overview.md](./rdma-overview.md) |
| C210   | cache_kernels multi-op                 | — (in `docs/snapshots/C210.md`)                            |
| C211   | triton_unified_attention PD-aware       | — (in `docs/snapshots/C211.md`)                            |
| C212   | PD disaggregated e2e                   | — (in `docs/snapshots/C212.md`)                            |
| C213   | triton_reshape_and_cache_flash         | — (in `docs/snapshots/C213.md`)                            |
| C214   | custom IPC all-reduce                  | — (in `docs/snapshots/C214.md`)                            |
| C220   | CUTLASS scaled MM + quant benchmark    | [cutlass-scaled-mm.md](./cutlass-scaled-mm.md)            |
| C221   | Triton fused MoE                       | — (in `docs/snapshots/C221.md`)                            |
| C223   | ngram spec decoder                     | — (in `docs/snapshots/C223.md`)                            |
| C224   | rotary embedding fused (Triton)        | — (in `docs/snapshots/C224.md`)                            |
| C230   | elastic EP rebalance + state machine   | [vllm-runtime-overview.md](./vllm-runtime-overview.md) § Elastic EP |
| C231   | cumem_allocator (real C extension)     | [vllm-runtime-overview.md](./vllm-runtime-overview.md) § Sleep/Wake Up + VMM |
| C232   | **this commit** — notes index          | All 8 notes                                                |

---

## 5. Reading order

If you're new to the codebase, read in this order:

1. **[gpu-arch-primer.md](./gpu-arch-primer.md)** — the hardware
   you're targeting.  Skip if you already know SM / warp / memory
   hierarchy.
2. **[vllm-runtime-overview.md](./vllm-runtime-overview.md)** — the
   software stack top-down.  This is the entry point for
   understanding *where* the Cxxx commits fit.
3. **[nccl-architecture.md](./nccl-architecture.md)** — if you're
   working on distributed paths (TP / PP / EP / PD).
4. **[gpu-profiling.md](./gpu-profiling.md)** — when you need to
   measure kernel performance.
5. **[cutlass-scaled-mm.md](./cutlass-scaled-mm.md)** +
   **[quantization-landscape.md](./quantization-landscape.md)** —
   when you're working on the GEMM / quantization path.

If you're debugging a specific area, jump straight to the relevant
note (topical map above).

---

## 6. Source-of-truth ordering

When in doubt, the source-of-truth is the vLLM repo at
`/home/kasw/文档/GitHub/GPU-Project/vllm/vllm/`.  These notes
**summarize and re-explain** vLLM's choices; they don't replace
reading the upstream.  Cross-references to the vLLM source files
appear in each note.

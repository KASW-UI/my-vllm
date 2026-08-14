# GPU Architecture Primer — SM / Warp / Memory Hierarchy

> **C232, NEW**.  Background for reading
> [`gpu-profiling.md`](./gpu-profiling.md) +
> [`cutlass-scaled-mm.md`](./cutlass-scaled-mm.md) +
> [`vllm-runtime-overview.md`](./vllm-runtime-overview.md)
> without flipping to NVIDIA docs every other paragraph.
>
> Audience: LLM kernel author working on Triton / CUDA / CUTLASS
> on Ada / Hopper / Blackwell.  Assumes you know Python + NumPy;
> doesn't assume prior GPU programming.

---

## 1. Why this matters for LLM serving

LLM kernels run on **tensor cores**, not scalar FPUs.  Tensor
cores do matrix-multiplication-accumulate (MMA) at fixed
shapes (m16n8k16 for FP16, m16n8k32 for FP8, etc.).  The
hardware is a deeply-pipelined SIMT machine — every cycle, an
SM (Streaming Multiprocessor) issues warps to functional units
that operate on shared register files and shared memory.

The kernel you write compiles to PTX → SASS, then runs on the
SMs.  To write a fast kernel you need to understand:

* how warps are scheduled within an SM (occupancy + ILP)
* where data lives in the memory hierarchy (register /
  shared / L1 / L2 / HBM)
* what the tensor cores can do (precision × shape × SM target)
* what makes a kernel *underperform* (coalescing misses, bank
  conflicts, register pressure, occupancy cliffs)

This doc walks through each.

---

## 2. The Streaming Multiprocessor (SM)

The SM is the GPU's compute unit.  RTX 4060 (sm_89 / Ada) has
**24 SMs**; H100 (sm_90 / Hopper) has **132 SMs**; H200 / B100
(sm_100 / Blackwell) has ~144+ SMs.  Each SM runs independently,
sharing only HBM (high-bandwidth memory) and L2 across the chip.

Inside one SM (Ada reference; Hopper / Blackwell differ in
detail but the model is the same):

```
SM
├── 4 warp schedulers (each issues 1 warp / cycle)
├── 16-32 lanes per warp (= SIMT width)
├── ~128 FP32 cores (= 4 schedulers × 32 lanes)
├── 64-128 tensor cores per SM (depending on arch)
├── Register file: 65536 × 32-bit registers (256 KB)
├── Shared memory: 100-228 KB (configurable carveout with L1)
├── L1 cache: 128 KB (shared with shared memory)
├── Load/Store units: 16-32 LSU per SM (memory ops per cycle)
└── Special Function Units (SFU): sin / cos / exp / rsqrt
```

**The cardinal rule**: a kernel is fast when it keeps all 4
warp schedulers busy *every cycle* with warps that have all
their operands ready in registers.  A stalled warp (waiting on
memory, waiting on a dependency, waiting on a barrier) wastes
its scheduler slot.

---

## 3. Warps and SIMT

A warp is 32 threads that execute in lockstep (SIMT — Single
Instruction Multiple Thread).  At each cycle, the scheduler
picks one ready warp and issues its next instruction; the 32
threads execute the same op on different data (registers).

**Implications**:

* **Branches**: a divergent branch serializes both paths.  If
  thread 0 takes `if` and thread 1 takes `else`, the SM
  executes both paths sequentially with each thread masked
  off the wrong path.  Pattern: "warp divergence" costs you
  throughput.

* **Memory access**: a coalesced load reads a 128-byte sector
  with one transaction.  A scattered load reads N sectors and
  wastes throughput.  Pattern: align accesses to 32-byte /
  128-byte boundaries, keep adjacent threads reading adjacent
  addresses.

* **Tensor cores**: when all 32 lanes of a warp can be cast
  onto a single MMA instruction, the SM issues it.  Patterns
  that don't fit (e.g. head_dim not divisible by 16) require
  padding or split-K.

---

## 4. Memory hierarchy

The memory pyramid on Ada (similar on Hopper):

```
Register file          ~256 KB / SM    ~32 TB/s   per-thread
Shared memory          ~100 KB / SM    ~30 TB/s   per-warp (programmable cache)
L1 / Texture cache    ~128 KB / SM    ~20 TB/s   per-warp (read-only)
L2 cache                32 MB          ~5 TB/s    across all SMs
HBM (HBM2e / GDDR6X)   24 GB (Ada)    ~1 TB/s    across the whole GPU
```

RTX 4060 (gaming) is GDDR6X with 272 GB/s bandwidth.  H100 SXM5
(80 GB HBM3) hits 3.35 TB/s.  B100 (Blackwell) is HBM3e at
~8 TB/s.

**Key insight**: HBM is the bottleneck for most LLM serving
kernels.  A "compute-bound" kernel still spends most of its
time waiting for the next weight tile from HBM.  The Roofline
model captures this (see [`gpu-profiling.md`](./gpu-profiling.md)):

```
FLOPS / sec ≤ min(peak_compute, peak_bandwidth × arithmetic_intensity)
```

Where `arithmetic_intensity = FLOPS / bytes_loaded`.  Below the
ridge point, you're memory-bound (add more bandwidth or fewer
bytes).  Above it, you're compute-bound (add more FLOPS or
fewer ops).

For LLM GEMMs:
- BF16: ~165 TFLOPS (Ada) — but actual kernel hits ~150 TFLOPS,
  bandwidth-bound below batch 32.
- FP8: ~330 TFLOPS (Ada) — bandwidth-bound below batch 64.
- INT8: ~330 TFLOPS (Ada) — same.
- INT4 (Marlin): ~660 TFLOPS — bandwidth-bound below batch 128.

---

## 5. Tensor cores

Tensor cores do MMA in fixed shapes.  The shape is a 3D tile
(M, N, K) that varies by precision and architecture:

| Arch    | Precision | Tile (M×N×K) | Per-SM throughput | Per-cycle FP32 ops |
|---------|-----------|--------------|-------------------|---------------------|
| Ada     | BF16     | 16×8×16      | ~1024 GFLOPS      | 2048                |
| Ada     | FP8      | 16×8×32      | ~2048 GFLOPS      | 4096                |
| Ada     | INT8     | 16×8×32      | ~2048 GFLOPS      | 4096                |
| Hopper  | BF16     | 16×8×16      | ~1970 GFLOPS      | 4096 (2× Ada)       |
| Hopper  | FP8      | 16×8×32      | ~3940 GFLOPS      | 8192 (2× Ada)       |
| Hopper  | FP4      | 16×8×64      | ~5910 GFLOPS      | 16384               |
| Hopper  | FP8 (WGMMA) | 64×256×32 | ~16 TFLOPS / SM    | 32768              |
| Blackwell | FP8   | 64×256×32    | ~32 TFLOPS / SM    | 65536                |

(Hopper WGMMA and Blackwell's wide tiles are the result of
specialized MMA paths that move the operand descriptors through
TMA — Tensor Memory Accelerator — rather than through SM
registers.  Out of scope for a primer.)

**For Triton kernels**: Triton's `tl.dot(a, b)` lowers to PTX
`mma.sync` instructions matching the architecture's tile shape.
The compiler picks the right shape — you don't usually need
to specify it.

**For CUDA CUTLASS**: you pick the tile shape explicitly.
CUTLASS ships ~50+ shapes per architecture; the autotuner
picks the best one for your specific input.

---

## 6. Occupancy vs ILP

Two competing strategies for hiding memory latency:

**High occupancy**: pack as many warps as possible onto an SM.
When one warp stalls on memory, the scheduler switches to
another.  Latency is hidden by warp count.

Ada limits: 64 warps / SM.  Each warp uses up to 64 registers
(voluntary) and up to ~100 KB shared mem.  If your kernel uses
256 registers per thread (= 8192 per warp), you're capped at
8 warps per SM = 12.5% occupancy.

**Low occupancy + high ILP (Instruction-Level Parallelism)**: keep
few warps per SM but give each warp enough independent work
that when one memory op stalls, another can issue.  Latency is
hidden by per-warp work.

Modern LLM kernels often run at *low* occupancy (16-32 warps / SM
out of 64) because the tensor cores have so much throughput
that you don't need warp parallelism — you just need each warp
to issue a long stream of independent MMA instructions.

---

## 7. Memory coalescing

A coalesced load: 32 adjacent threads in a warp load 32
adjacent 4-byte values → one 128-byte transaction → 1 sector
read from L2.

A scattered load: 32 threads in a warp load 32 random 4-byte
values → up to 32 separate transactions → 32× slower.

The hardware's coalescing rule: if all 32 threads in a warp
hit the same 128-byte sector, you get 1 transaction.  If they
hit 2 sectors, you get 2 transactions.  N sectors → N
transactions (worst case).

**For Triton kernels**: use `tl.arange(0, BLOCK_SIZE)` and
`tl.load(ptr + offsets)` where `offsets` is contiguous.  Triton
will coalesce automatically.  If you need non-contiguous
access (e.g. gathering from a block table), the cost is
inherent — but you can mask the wasted lanes with `tl.where`.

**For CUDA kernels**: explicit `__shfl_*` + per-thread load + SM
coalescing.  Easier to get wrong; CUTLASS abstracts this.

---

## 8. Shared memory bank conflicts

Shared memory has 32 banks (4 bytes each, on most archs).
If 32 threads in a warp each access a different bank, no conflict.
If two threads access the same bank at the same address, 1
transaction.  If they hit the same bank at *different* addresses,
serialized.

Pattern: stride-32 access (e.g. accessing the same column of a
matrix by all threads) is a conflict.  Padding the leading
dimension to avoid it (e.g. `[M, K+1]` instead of `[M, K]`) is
the standard fix.

---

## 9. Roofline model in 1 paragraph

The Roofline model says: a kernel's attainable FLOPS is bounded
by `min(peak_compute, peak_bandwidth × arithmetic_intensity)`.
Below the ridge point (low arithmetic intensity), the kernel is
bandwidth-bound and you need to reduce bytes loaded (better
caching, smaller dtype, fusion).  Above it, the kernel is
compute-bound and you need to reduce ops (bigger tiles, better
tile shapes, lower-precision MMA).  See
[`gpu-profiling.md`](./gpu-profiling.md) § 2 for the full
treatment.

---

## 10. CUDA execution model in 1 paragraph

The host launches kernels onto a stream.  The kernel's grid
runs on the SMs: each thread block (CTA) is assigned to one
SM.  Threads within a CTA share shared memory + can synchronize
via `__syncthreads()`.  CTAs run independently and can
communicate only via global memory (HBM) or atomics on L2.

`__syncthreads()` is the per-CTA barrier.  It serializes all
warps in the CTA — heavy.  Most kernels use it once per tile
boundary, not per element.

`cudaDeviceSynchronize()` blocks the host.  Even heavier.  Use
only at stream boundaries (one per step in the inference
loop).

---

## 11. What to remember

When you're staring at a slow kernel:

1. **Measure first** — Nsight Compute (see `gpu-profiling.md`)
   tells you which of these is the bottleneck.
2. **Memory first** — most LLM kernels are bandwidth-bound,
   not compute-bound.  Check bytes loaded vs bytes actually
   needed.
3. **Coalescing** — if your load isn't 1 transaction per warp,
   that's the first thing to fix.
4. **Bank conflicts** — if your shared-mem accesses are
   serialized, pad the leading dimension.
5. **Occupancy** — if you're below 25% occupancy with low
   arithmetic intensity, increase warps per SM.
6. **Tensor core utilization** — if MMA throughput is below
   70% of peak, you're losing compute; check tile shapes
   and K-loop unrolling.
7. **Synchronization** — `__syncthreads()` is expensive; one
   per tile is usually enough.

---

## 12. References

- NVIDIA, "CUDA C++ Programming Guide" (current edition).
  The SM chapter and the "Memory Hierarchy" appendix are
  the canonical reference.
- NVIDIA, "Whitepaper: NVIDIA Hopper Architecture in Depth"
  (2023).  WGMMA, TMA, async copies, distributed shared memory.
- NVIDIA, "Whitepaper: NVIDIA Blackwell GPU Architecture"
  (2024).  New MMA shapes, FP4 tensor cores, transformer
  engine.
- Williams, Waterman & Patterson, "Roofline: An Insightful
  Visual Performance Model for Multicore Architectures" (CACM
  2009).  The roofline model paper.
- Cutler et al., "CUTLASS: A GPU-Accelerated Library for
  Matrix Multiplication and Beyond" (NVIDIA tech report 2022).
  The CUTLASS-3.x design with tile-shape autotuning.
- The CUDA Occupancy Calculator spreadsheet (NVIDIA) — gives
  the exact warps-per-SM cap for a given kernel's register /
  shared-mem usage on each arch.

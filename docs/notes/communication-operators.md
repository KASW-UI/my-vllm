# Distributed Communication Operators — Notes

> C248 — `phase3-distributed-runtime-plan.md` § Phase 4 (Distributed
> Runtime 强化).  Closes the A-level "Distributed comm benchmark +
> 笔记" gap.

## Scope

This note covers the **5 core distributed communication collectives**
that every LLM serving system must deal with, and how my-vllm benchmarks
them across the backends it has actually implemented (C201, C214, C240,
C241, C242, C244).  After reading this note you should understand:

* **What the 5 collectives are** — `all_reduce`, `all_gather`,
  `reduce_scatter`, `all_to_all`, `broadcast` — and what each moves
  per rank across the wire.
* **How the algorithms decompose** — ring, tree, halving-doubling,
  one-shot vs two-shot, gather-then-broadcast — and why the
  algorithmic bandwidth formulas (`alg_bw`, `bus_bw`) look the way
  they do.
* **Which bus each backend rides** — NVLink-Switch (NVLS),
  NVLink, PCIe, IB / RoCE — and what bandwidth you can expect
  from each.
* **The backend hierarchy in my-vllm** — for each collective, which
  backends are available, and how the dispatcher picks one.
* **The decision tree** — given a message size and world size, which
  backend should win for `all_reduce`?

## What the 5 collectives are

Throughout, let

* `n` = world size (number of ranks in the collective)
* `s` = per-rank message size (in bytes)
* `t` = wall-clock time per call (in seconds)

### all_reduce

* **Semantics**: every rank's input is summed (or other op) and the
  result is written to every rank's output.  Bandwidth-optimal ring:
  `2·(n-1)·s` total bytes across the wire (each chunk is sent once +
  received once).
* **Algorithmic bandwidth**: `alg = 2·(n-1)/n · s / t` (the
  reduction is `n`-way per chunk, but each rank contributes `1/n` of
  the reduction).
* **Bus bandwidth**: `bus = 2·(n-1)·s / t`.
* **Where it shows up in LLM serving**: every `RowParallelLinear`
  backward, every TP `reduce` step, every `RMSNorm` after a
  ColumnParallel linear, every gradient synchronization in training,
  every vLLM `tensor_model_parallel_all_reduce` call.

### all_gather

* **Semantics**: every rank's input chunk is concatenated and the
  full sequence is written to every rank's output.  Total bytes
  across the wire: `(n-1)·s` per rank.
* **Algorithmic bandwidth**: `alg = s / t`.
* **Bus bandwidth**: `bus = (n-1)·s / t`.
* **Where it shows up**: every `RowParallelLinear` forward (gather
  partial outputs), every `VocabParallelEmbedding` forward (gather
  partial logits), every `AgRsAll2AllManager.dispatch` (small-eP
  paths).

### reduce_scatter

* **Semantics**: each rank contributes a chunk of the input, and the
  `n`-way sum of all chunks is written to the corresponding rank's
  output.  Total bytes: `(n-1)·s` per rank.
* **Algorithmic bandwidth**: `alg = s / t`.
* **Bus bandwidth**: `bus = (n-1)·s / t`.
* **Where it shows up**: every `RowParallelLinear` input gradient,
  every `ColumnParallelLinear` gradient, every `AgRsAll2AllManager.combine`,
  every `FusedMoE` weight-gradient split.

### all_to_all

* **Semantics**: each rank's input is split into `n` chunks (one per
  rank), and rank `i` receives the chunks from all ranks.  Total
  bytes: `(n-1)·s` per rank (uniform chunk size).
* **Algorithmic bandwidth**: `alg = s / t`.
* **Bus bandwidth**: `bus = (n-1)·s / t`.
* **Where it shows up**: every MoE dispatch / combine on distributed
  experts, every sequence-parallel rerouting, every ragged KV
  redistribution for elastic EP rebalancing.

### broadcast

* **Semantics**: one rank's tensor is sent to all other ranks.  Total
  bytes: `s` per receiver (no sum).
* **Algorithmic bandwidth**: `alg = s / t`.
* **Bus bandwidth**: `bus = s / t` (the source emits `s` once).
* **Where it shows up**: every `WeightTransferEngine` send/recv
  (C243), every NCCL `init_` call that needs a synchronizing
  rendezvous, every TP setup broadcast (the linear weight is
  partitioned on rank 0 and broadcast to the others), every
  ShmMessageQueue (C242) sender flush.

## Algorithm taxonomy

The **ring** algorithm is the canonical NCCL all-reduce:

1. Split the input into `n` chunks of `s/n` bytes each.
2. Each rank sends and receives a chunk with one neighbor per
   ring step; `n-1` steps total.
3. After the first `n-1` steps the value is **reduced** but only
   committed to one rank's copy.
4. After the next `n-1` steps the reduced value is **broadcast** to
   all ranks.

Total steps: `2·(n-1)`.  Total bytes per rank: `2·(n-1)·s/n`.  Sum
across all ranks: `2·(n-1)·s` (consistent with the `bus_bw`
formula).

The **tree** algorithm is bandwidth-optimal for very small `n`
(`bus = 2·s` for binary tree) but high-latency.  NCCL uses tree
for small messages and ring for large ones (heuristic in the
`nccl_tuner` plugin).

The **halving-doubling** algorithm is a variant of ring used by
older MPI libraries; equivalent bandwidth but different
fragmentation pattern.

The **one-shot** vs **two-shot** distinction matters for **all_reduce**
when NVLS is available:

* **Two-shot**: ring-style; first half (n-1 steps) reduces, second
  half broadcasts.  Uses `2·(n-1)·s` bytes.
* **One-shot** (NVLS-multimem): all ranks read a multicast window
  and write the sum in a single round-trip using NVLS hardware.
  Uses `s` bytes per rank (or `n·s` total).

For **all_to_all**, the naive algorithm is **gather-then-broadcast**
(`O(P²)` network traffic), but the bandwidth-optimal token-routing
algorithm swaps tokens via ring or randomized oblivious routing.
Production EP libraries (DeepEP, FlashInfer, NIXL EP) implement
the latter.

For **all_gather**, NCCL implements a **doubling** algorithm:
in step `k`, rank `i` sends its chunk to rank `i XOR 2^k`.  After
`log₂(n)` steps everyone has the full sequence.  Lower latency
than ring for small `n`, similar bandwidth.

For **reduce_scatter**, NCCL implements a **ring** variant: each
rank sends its chunk to one neighbor, the chunk is reduced along
the way, and the result is truncated to the local output.  Same
total bytes as `all_gather`.

For **broadcast**, NCCL uses a `tree` or `chain` topology.  Single
receiver `s` bytes, total `s` bytes transmitted.

## Backend hierarchy

my-vllm has implemented the following backends (one per cell of
the op × backend matrix that's actually useful):

| Backend | Op(s) | Hardware | Mechanism | Code |
|---|---|---|---|---|
| `torch.dist` | all 5 | NCCL / Gloo | PyTorch native | `torch.distributed.*` |
| `custom_ar` | all_reduce | CUDA IPC, single-node, n ≤ 8 | Patariasuk & Yuan 2009 ring | `c214` `csrc/custom_all_reduce.cu` |
| `symm_mem` | all_reduce | NVLS (Hopper+, sm_90+) | 1-shot multicast | `c241` `symm_mem.py` |
| `ag_rs` | all_to_all | NCCL/Gloo | AllGather + ReduceScatter | `c240` `all2all.py` |
| `shm` | broadcast | Same-host SHM | Lock-free ring buffer | `c242` `shm_broadcast.py` |
| `nccl` (NCCL) | all 5 | NCCL | `nccl*.cu` kernels | vendored via PyTorch |
| (planned) FlashInfer mnnvl | all_reduce | NVLink-Switch | multicast | `c244` shell |
| (planned) QuickReduce | all_reduce | ROCm MI300 | quant codec | `c244` shell |

The first 5 rows are **fully exercisable** on my-vllm's dev box
(RTX 4060 / sm_89 plus multi-process gloo fallback for `n > 1`).
The last 2 are **class shells** with computed heuristics but no
runtime — they would need the vendor library to actually run.

## Hardware-bandwidth hierarchy

In a typical multi-GPU host (H100 + NVLink + IB):

| Bus | Direction | Effective bandwidth | Latency |
|---|---|---|---|
| NVLink-Switch (NVLS) | GPU↔GPU (multicast) | ~900 GB/s aggregate | ~1 µs |
| NVLink | GPU↔GPU (unicast) | ~450-900 GB/s/link | ~1 µs |
| PCIe Gen5 | CPU↔GPU | ~64 GB/s | ~2 µs |
| IB HDR | GPU↔GPU (cross-node) | ~25 GB/s | ~5 µs |
| IB NDR | GPU↔GPU (cross-node) | ~50 GB/s | ~3 µs |
| RoCE | GPU↔GPU (cross-node) | ~25 GB/s (~10-25% loss to CPU) | ~10-50 µs |
| Gloo CPU | GPU↔GPU fallback | ~1 GB/s | ~50 µs |

For a `2 MB` all-reduce on 8 ranks via NVLink:

* `bus = 2·(8-1)·2 MB = 28 MB` per rank.
* At 450 GB/s wire, `t ≈ 28 MB / 450 GB/s ≈ 62 µs`.
* `alg_bw = 2·7/8 · 2 MB / 62 µs ≈ 56 GB/s`.

Compare to a single-process GPU memcpy at ~1 TB/s — collectives
trail by ~20×.

## Backend dispatch (my-vllm)

The CudaCommunicator's `all_reduce` (in `cuda_communicator.py:125`)
implements the dispatch chain:

```
                ┌─ world_size == 1 → identity
                │
all_reduce ─────┤
                ├─ symm_mem attached + eligible → symm_mem.all_reduce
                │
                ├─ group_coordinator bound → GroupCoordinator.all_reduce
                │
                └─ torch.distributed.all_reduce (PG)
```

For `all_gather` and `reduce_scatter` the dispatch is the
GroupCoordinator (which calls the underlying NCCL/Gloo).  For
`all_to_all`, the dispatch routes to `AgRsAll2AllManager` (which
decomposes into `all_gather` + `reduce_scatter`).

## Decision tree for all_reduce

The single biggest optimization for LLM serving is choosing the
right backend for `all_reduce` (because it's called every decoder
step).  The decision tree:

```
                   ┌─ NVLS available (Hopper+) ─┬─ input ≤ SYMM_MEM_MAX → multimem (1-shot)
all_reduce ───────┤                             └─ input > SYMM_MEM_MAX → two-shot (still NVLS)
                   │
                   ├─ single-node + n ≤ 8 + input ≤ CUSTOM_AR_MAX → custom_ar (CUDA IPC ring)
                   │
                   └─ everything else → NCCL (PyTorch)
```

The actual heuristics my-vllm ships:

* `should_nccl_symm_mem_ag_rs(input)` — decides whether the
  Next-Generation NCCL NVLS-AG+RS path is eligible.
* `should_nccl_symm_mem_allreduce(input)` — same for the
  all-reduce path.
* `CustomAllreduce.should_custom_ar(input)` — same for the
  CUDA IPC path.

The thresholds are in `my_vllm/distributed/device_communicators/all_reduce_utils.py`:

```python
SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0":  {4: 64 * MiB, 6: 32 * MiB, 8: 16 * MiB},
    "10.0": {6: 64 * MiB, 8: 32 * MiB},
    "10.3": {6: 64 * MiB, 8: 32 * MiB},  # GB200
}
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    2: 2 * MiB, 4: 2 * MiB, 6: 1 * MiB, 8: 1 * MiB,
}
```

The intuition:

* **Custom AR** is fastest for small messages (≪ 1 MB) because the
  CUDA IPC + ring takes 1 round-trip but doesn't pay the full
  NCCL setup cost.
* **SymmMem (NVLS)** is fastest for medium messages (1-64 MB) on
  Hopper+: the 1-shot multicast avoids the 2-round-trip ring cost.
* **NCCL** wins for large messages (> 64 MB) because its bandwidth
  scales with the bus speed, while SymmMem is bottlenecked by the
  NVLink-Switch multicast bandwidth.

## Implementation in my-vllm

The unified benchmark lives at `examples/distributed_comm_benchmark.py`
and exposes a single CSV schema:

```
backend,op,size_bytes,num_elements,dtype,world_size,elapsed_us,alg_bw_GBps,bus_bw_GBps,note
```

Two modes:

* **Single-rank** (`python examples/distributed_comm_benchmark.py`)
  — runs on a single GPU/CPU with `world_size=1`.  Each backend is
  benchmarked in identity mode.  Useful for overhead comparison
  and "is the harness wired right?" verification on a single dev box.
* **Multi-process gloo** (`python examples/distributed_comm_benchmark.py --gloo --world_size 2`)
  — spawns `n` workers via `torch.multiprocessing.spawn` with the
  gloo backend.  Round-trip correctness for each op (broadcast /
  all_gather / all_reduce).  Bandwidth on real NCCL requires a
  multi-GPU box.

The benchmark is **single-source-of-truth** for op × backend
dispatch: adding a new backend requires adding one entry to the
`Backend` registry in `_build_available_backends`.

## Tests (`tests/test_C248_distributed_comm.py`)

The test suite (25 tests) covers:

1. Module import + CSV header schema.
2. `make_sizes` geometric sweep.
3. `Backend` dataclass construction.
4. Bandwidth formulas (per NCCL convention).
5. `_bench_single_rank` emits one CSV row per (backend, size,
   dtype).
6. Skipped backends emit `elapsed_us=-1` rows.
7. Backend registry contains all 5 ops (torch.dist) + custom_ar +
   symm_mem + ag_rs + shm.
8. SymmMem is disabled on RTX 4060 (sm_89 doesn't support NVLS).
9. CustomAllreduce is identity for `world_size == 1`.
10. AgRsAll2AllManager `dispatch` + `combine` is identity on
    `world_size == 1`.
11. Multi-process gloo round-trip for broadcast / all_gather /
    all_reduce (via `torch.multiprocessing.spawn`).
12. CLI smoke test (subprocess invocation + CSV parse).

## Examples

```bash
# Single-rank (identity baseline; no NCCL needed)
python examples/distributed_comm_benchmark.py

# Multi-process gloo round-trip (4 workers)
python examples/distributed_comm_benchmark.py --gloo --world_size 4

# Limited size range for quick smoke test
python examples/distributed_comm_benchmark.py --min_bytes 1024 --max_bytes 4096

# Different dtype (symm_mem only supports bfloat16)
python examples/distributed_comm_benchmark.py --dtype bfloat16
```

## Risks / limitations

1. **Single-rank latency is dominated by Python overhead**.  The
   single-rank benchmark is meant for **correctness** and
   **relative overhead comparison**, not for measuring actual
   bandwidth.  Real bandwidth needs a multi-GPU box with NCCL.
2. **Custom AR requires CUDA** — the C214 extension is only built
   with `setup.py build_ext --inplace`.  On a CPU-only box the
   custom AR backend is silently identity.
3. **SymmMem requires NVLS hardware** (Hopper+, sm_90+).  On
   pre-Hopper GPUs (e.g. RTX 4060 / sm_89) the communicator is
   disabled and the benchmark emits `-1` rows.
4. **SHM broadcast is single-host** — the C242 `MessageQueue` is
   for same-host broadcast.  Cross-node broadcast must use the
   NCCL `broadcast` path.
5. **Naive all_to_all is O(P²)** — the C240 `AgRsAll2AllManager`
   is the naive AllGather + ReduceScatter backend.  True
   bandwidth-optimal all-to-all would need DeepEP / FlashInfer /
   NIXL EP (out of scope for the JD-aligned subset).
6. **No CUDA graph capture** — vLLM's `benchmark_device_communicators.py`
   captures CUDA graphs to remove launch overhead.  my-vllm's
   benchmark measures **eager-mode** latencies; CUDA graph capture
   would reduce the small-message numbers by ~30%.

## Future work (per `phase3-distributed-runtime-plan.md`)

* **C250 (NIXL KV Connector)** — cross-node KV transfer uses
  distinct bandwidth / latency dynamics from the all-reduce
  family.  The C248 harness will be extended with a `kv_transfer`
  column to benchmark prefill → decode KV migration.
* **C251 (KV event bus)** — cross-engine event broadcast (prefill
  done, decode slot free, etc.).  Builds on the SHM broadcast
  primitive from C242.
* **C262 (CUDA VMM 闭环)** — sleep / wake up with CPU offload.
  The communication operators in this note are all in-memory;
  C262 adds the disk-tier path that complements the GPU RAM
  tier.
* **C264 (Spec decode runner 闭环)** — the verify / accept / bonus
  path uses all_reduce (for token-tree verification) and
  all_gather (for distributed draft state).  The C248 benchmark
  will be extended to measure the spec-decode-specific colletive
  patterns.
* **C265 (MoE AllToAll 集成)** — full MoE forward with DeepEP
  integration.  The C248 benchmark will gain a `deep_ep` backend
  once the kernels are vendored.

## References

* NCCL all-reduce algorithm: Patarasuk & Yuan, "Bandwidth Optimal
  All-reduce Algorithms for Clusters of Workstations" (2009).
* NCCL NVLS docs:
  <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#ncclcommwindowregister>.
* PyTorch symmetric memory:
  <https://pytorch.org/docs/stable/distributed.symm_mem.html>.
* my-vllm `docs/notes/nccl-architecture.md` — ring vs tree, NCCL
  adaptive routing.
* my-vllm `docs/notes/rdma-overview.md` — IB / RoCE / iWARP.
* my-vllm `docs/notes/all2all-overview.md` — DeepEP / NIXL EP /
  FlashInfer backend survey.
* my-vllm `docs/notes/symm-mem-overview.md` — NVLS hardware
  primer + PyTorch runtime + vLLM heuristic.
* my-vllm `docs/notes/shm-broadcast-overview.md` — SHM ring buffer
  + MessageQueue.
* vLLM `benchmarks/kernels/benchmark_device_communicators.py` —
  CustomAllreduce + PyNccl + SymmMem + FlashInfer benchmark.

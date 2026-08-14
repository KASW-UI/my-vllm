# All-to-All Communication — Notes

> C240 — `phase3-distributed-runtime-plan.md` § Phase 3 (Distributed
> Runtime 强化).  Closes the S-level "All-to-All 通信算子" gap.

## Scope

This note covers the **All-to-All** collective primitive that
underlies Expert-Parallel (EP) and Data-Parallel (DP) load
balancing in vLLM (and my-vllm).  After reading this note you
should understand:

* **What All-to-All is** and why MoE / DP serving needs it.
* **The two algorithm families**: AG+RS (naive, bandwidth-wasteful)
  and bandwidth-optimal token-routed all-to-all (DeepEP, FlashInfer
  NVLink, NIXL EP, Mori).
* **How vLLM organises the abstraction layer**
  (`All2AllManagerBase` → concrete backends → `PrepareAndFinalize`
  ABCs → `FusedMoE` modular kernel).
* **Where my-vllm lands**: the naive AG+RS path is fully
  implemented; the other backends are class shells awaiting
  external-library vendoring; the Prepare/Finalize ABCs and the
  no-EP / naive-EP prepare-finalize classes are implemented.
* **Why we don't vendor DeepEP / NIXL EP** as part of my-vllm:
  they're 50K+ lines of C++/CUDA and the algorithmic core
  (AG+RS) is enough to demonstrate the design.

## Why All-to-All matters

### MoE Expert Parallel

In a Mixture-of-Experts layer with `E` experts, each token is
routed to `top_k` (typically 2-8) experts.  With Expert-Parallel
(`ep_size > 1`), each rank owns a **disjoint slice** of experts
(e.g. 8 experts per rank when `E=64, ep_size=8`).  For the
forward pass:

1. **Dispatch**: each rank needs the **tokens routed to its
   experts**.  Rank `r` initially holds `M_r` local tokens; after
   the all-to-all, every rank has `sum(M_r')` tokens, **but each
   token only needs to be processed by the rank that owns its
   target expert**.

2. **Local expert compute**: each rank runs its local experts on
   its slice of the dispatched tokens.

3. **Combine**: each token needs to gather its `top_k` expert
   outputs (which may live on different ranks); the combine step
   routes the per-rank expert outputs back to the rank that owns
   the original token and reduces them.

Without EP, every rank runs every expert → `ep_size`× wasted
compute.  With EP, each rank runs only `E / ep_size` experts →
`ep_size`× throughput.

### DP load balancing

Data-Parallel serving (multiple replicas of the same model)
imbalances load: some replicas are busy on long sequences while
others are idle.  **DP load balancing** (a.k.a. "continuous
batching with prefix cache sharing across replicas") routes
tokens from a busy replica to an idle one.  This requires
**All-to-All** of the activation buffers between the replicas
between the schedule step and the model forward.

In vLLM V1, the `AgRsAll2AllManager` is the default backend
for DP load balancing (the `AgRs` in `AgRsAll2AllManager` =
"AllGather + ReduceScatter").  DeepEP / FlashInfer are bandwidth-
optimal for the EP path; for the DP path, AG+RS is typically
fast enough.

## Algorithm: AllGather + ReduceScatter

The "naive" all-to-all has two phases:

### Phase 1: Dispatch via AllGather

Each rank `r` holds `M_r` local tokens in `A_r` of shape
`(M_r, K)`.  All ranks call:

```
all_gather(A_r, dim=0, sizes=[M_0, M_1, ..., M_{P-1}])
```

After the AG, every rank has the world-concatenated tensor of
shape `(sum(M_r), K)`.  Each rank then runs the experts whose
IDs it owns, filtering tokens by `expert_map` and `topk_ids`.

**Bandwidth**: `O(P × M)` per rank — each rank receives `P` chunks
even though it only needs `1/P` of them.  This is the
"bandwidth-wasteful" property.

### Phase 2: Combine via ReduceScatter

After the experts run, each rank has `(sum(M_r), top_k, K)`
partial outputs (one per token × top-k expert, before reduction).
All ranks call:

```
reduce_scatter(B_r, dim=0, sizes=[M_0, M_1, ..., M_{P-1}], op=SUM)
```

After the RS, rank `r` has `(M_r, K)` — the per-rank-summed
output for its original tokens.

**Bandwidth**: `O(P × M)` per rank (same as dispatch).

### Why this is correct

The AG gives every rank the world-wide token set.  After experts
run, each rank has world-wide partial outputs.  The RS sums
across all ranks and slices.  Crucially, **the per-rank
contribution at slice `r` is exactly the contribution from
experts on rank `r`** — so the sum is the right thing to
compute.

### Why this is bandwidth-wasteful

A true all-to-all would route only the tokens that need to go to
each rank.  For `P` ranks with uniform `M` tokens each, the true
bandwidth is `O(M)` per rank (each rank sends `M/P` tokens and
receives `M/P`).  AG+RS sends `M(P-1)/P` tokens and receives
`M(P-1)/P`, which is `O(PM)` — `P×` more.

For `ep_size = 8` on a 1024-token batch with `K=4096` FP16, the
"wasted" bandwidth is `8 × 1024 × 4096 × 2 = 64 MB` per token
slice, vs the ideal `~8 MB`.  In practice, AG+RS dominates
latency at `ep_size >= 4` and is the reason DeepEP / NIXL EP /
FlashInfer exist.

## Bandwidth-optimal all-to-all (DeepEP, FlashInfer, NIXL EP)

These libraries route tokens at the **expert granularity**: each
rank sends only the tokens destined for each expert.  The
algorithm is fundamentally different:

* **DeepEP High-Throughput**: NVLink intra-node + RDMA inter-node
  with custom CUDA kernels for the token-permutation + RDMA
  transfer.  Optimized for large `M` (e.g. prefill).
* **DeepEP Low-Latency**: SM-count optimized, RDMA-only; designed
  for small `M` (decode).
* **DeepEP V2 ElasticBuffer**: NCCL GIN + IBGDA; supports elastic
  rank add/remove.
* **NIXL EP**: elastic EP via NIXL (cross-node RDMA transport);
  designed for elastic EP rebalance (C247 future work).
* **FlashInfer NVLink (two-sided / one-sided)**: `MnnvlMoe` /
  `MoeAlltoAll` over MNNVL NVLink one-/two-sided collectives
  (Hopper+ NVLink Switch).
* **Mori**: ROCm-specific (out of scope for my-vllm).

These are external libraries; my-vllm ships class shells that
raise `NotImplementedError` for the dispatch / combine methods.

## vLLM's abstraction layer

vLLM organises the All-to-All surface into four layers:

```
FusedMoE (nn.Module)
  ↓ forward → apply (calls quant_method.apply)
FusedMoEMethodBase (FP8 / INT8 / etc. quant dispatch)
  ↓ apply (calls moe_kernel.apply)
FusedMoEModularKernel
  ├── PrepareAndFinalize (e.g. MoEPrepareAndFinalizeNaiveDPEPModular)
  │     prepare() → quantize + AllGather dispatch
  │     finalize() → top-K reduce + ReduceScatter combine
  └── FusedMoEExpertsModular
        apply() → run experts kernel
```

The `All2AllManagerBase` ABC is the leaf of the abstraction:

```
All2AllManagerBase (abstract)
  ├── AgRsAll2AllManager (naive, fully impl.)
  ├── DeepEPAll2AllManagerBase
  │     ├── DeepEPHTAll2AllManager
  │     ├── DeepEPLLAll2AllManager
  │     └── DeepEPV2All2AllManager
  ├── NixlEPAll2AllManager
  ├── FlashInferNVLinkOneSidedAll2AllManager
  ├── FlashInferNVLinkTwoSidedAll2AllManager
  └── MoriAll2AllManager
```

The `DeviceCommunicatorBase` (per-device façade) holds a
`all2all_manager` instance and delegates
`dispatch`/`dispatch_router_logits`/`combine` to it.

The `GroupCoordinator` (process-group façade) exposes
`all_gatherv` / `reduce_scatterv` that the AG+RS manager uses
for the actual collectives.  In my-vllm, the
`GroupCoordinator.all_gatherv` is a gather-then-broadcast pattern
(when variable sizes are non-uniform); the uniform path uses
`torch.distributed.all_gather_into_tensor` /
`torch.distributed.reduce_scatter_tensor` which map to NCCL on
GPU.

## my-vllm C240 implementation

### Files

```
my_vllm/distributed/
  __init__.py                     # + get_ep_group, init_expert_parallel re-exports
  parallel_state.py               # + GroupCoordinator.all_gatherv / reduce_scatterv /
                                  #   dispatch / dispatch_router_logits / combine façade
                                  # + _EP group + init_expert_parallel
  device_communicators/
    __init__.py                   # package + re-exports
    base_device_communicator.py   # All2AllManagerBase + DeviceCommunicatorBase + Cache
    all2all.py                    # AgRsAll2AllManager (full) + DeepEP/NIXL/Mori shells
    cuda_communicator.py          # CudaCommunicator (delegates to GroupCoordinator)

my_vllm/model_executor/layers/fused_moe/
  modular_kernel.py               # FusedMoEPrepareAndFinalize ABC + Modular/Monolithic
  all2all_utils.py                # maybe_make_prepare_finalize() factory
  prepare_finalize/
    __init__.py                   # re-exports
    naive_dp_ep.py                # NaiveDPEP Modular + Monolithic + factory
    no_dp_ep.py                   # NoDPEP Modular + Monolithic + factory
  config.py                       # + use_*_kernels properties on FusedMoEParallelConfig

my_vllm/config/parallel.py        # + All2AllBackend literal + ParallelConfig.all2all_backend
```

### Public API

```python
from my_vllm.distributed import (
    get_ep_group, init_expert_parallel,
)

from my_vllm.distributed.device_communicators import (
    All2AllManagerBase, AgRsAll2AllManager, CudaCommunicator,
    DeviceCommunicatorBase, Cache,
)

from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
    MoEPrepareAndFinalizeNaiveDPEPMonolithic,
    MoEPrepareAndFinalizeNoDPEPModular,
    MoEPrepareAndFinalizeNoDPEPMonolithic,
)

from my_vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
```

### Algorithm correctness

The `AgRsAll2AllManager` is fully implemented:

* **dispatch_router_logits(hidden, router_logits, is_sequence_parallel, extra_tensors)**:
  AllGather `(hidden, router_logits[, extras])` across the EP /
  DP / PCP group.  Each rank receives the world-wide
  concatenation.
* **dispatch(hidden, topk_weights, topk_ids, is_sequence_parallel, extra_tensors)**:
  AllGather `(hidden, topk_weights, topk_ids[, extras])` across
  the comm group.
* **combine(hidden, is_sequence_parallel)**: ReduceScatter
  `(hidden)` across the comm group; rank `r` receives its slice
  summed across all ranks' contributions.
* **Identity when world_size=1**: dispatch returns inputs
  unchanged; combine returns input unchanged.

### GroupCoordinator facade

The `GroupCoordinator.all_gatherv` is a thin layer over
`torch.distributed`:

* **Uniform sizes** (or `sizes is None`): falls through to
  `torch.distributed.all_gather_into_tensor` (NCCL `all_gather`).
* **Non-uniform sizes**: gather-then-broadcast via rank 0
  (correct, not bandwidth-optimal — true
  `ncclAllGatherv` requires PyNCCL).

The `GroupCoordinator.reduce_scatterv` mirrors this:

* **Uniform sizes**: `torch.distributed.reduce_scatter_tensor`
  (NCCL `ncclReduceScatter`).
* **Non-uniform sizes**: gather-then-sum-then-scatter via rank 0
  (correct, not bandwidth-optimal).

### FusedMoEParallelConfig selectors

The `FusedMoEParallelConfig` (dataclass) gains the vLLM-style
selectors:

* `use_all2all_kernels: bool` — `True` if EP or DP is enabled.
* `use_ag_rs_all2all_kernels: bool` — `True` for the default
  AG+RS backend.
* `use_deepep_kernels: bool` — `True` for `deepep_*` backends.
* `use_nixl_ep_kernels: bool` — `True` for `nixl_ep`.
* `use_flashinfer_kernels: bool` — `True` for
  `flashinfer*` backends.
* `use_mori_kernels: bool` — `True` for `mori` (ROCm).

These drive the `maybe_make_prepare_finalize()` factory
dispatch.

### PrepareAndFinalize ABC

The `FusedMoEPrepareAndFinalize` ABC has two subclasses:

* `FusedMoEPrepareAndFinalizeModular`: kernels that take
  `(topk_weights, topk_ids)` and apply them.
* `FusedMoEPrepareAndFinalizeMonolithic`: kernels that take
  `router_logits` and compute top-k internally.

Each subclass has `prepare()` + `finalize()`:

```
prepare: a1 → quantize → dispatch via All2All manager
finalize: fused_expert_output → top-K reduce → combine via All2All manager
```

### Concrete PrepareAndFinalize implementations

* `MoEPrepareAndFinalizeNoDPEPModular` + `Monolithic`:
  identity (no comm).  Used for dense OPT (default
  `dp_size == ep_size == 1`).
* `MoEPrepareAndFinalizeNaiveDPEPModular` + `Monolithic`:
  AG+RS via `get_ep_group().dispatch()` /
  `get_ep_group().combine()`.

### Factory dispatch

`maybe_make_prepare_finalize(moe)` selects the PrepareAndFinalize
based on `moe.moe_parallel_config`:

```python
if not use_all2all_kernels: return NoDPEP
if use_ag_rs_all2all_kernels: return NaiveDPEP
# DeepEP / NIXL EP / FlashInfer / Mori fall back to NoDPEP
# (not vendored; log warning).
```

## Test coverage

`tests/test_C240_all2all.py` (61 test cases):

* `TestAll2AllManagerBase` — ABC contract (`NotImplementedError`).
* `TestAgRsAll2AllManagerIdentity` — single-rank dispatch / combine
  are identity.
* `TestAgRsAll2AllManagerAlgorithm` — algorithm correctness with
  a mock `MockCommGroup` simulating `world_size=4`.
* `TestAgRsAll2AllManagerVariableSizes` — uniform AG path.
* `TestCudaCommunicatorIdentity` — `CudaCommunicator` identity +
  dispatch / combine shim with no `all2all_manager`.
* `TestFusedMoEParallelConfigProperties` — `use_*_kernels` mapping.
* `TestMaybeMakePrepareFinalize` — factory dispatch.
* `TestNoDPEPRoundTrip` — end-to-end single-rank round-trip.
* `TestNaiveDPEPRoundTripIdentity` — when no EP group is set.
* `TestParallelStateEPGroup` — `init_expert_parallel` lifecycle.
* `TestGroupCoordinatorAllGatherRSingle` — single-rank
  short-circuit.
* `TestAll2AllBackendEnum` — `All2AllBackend` literal set.
* `TestBackendShellsRaise` — DeepEP / NIXL / FlashInfer / Mori
  shells raise `NotImplementedError`.
* `TestModularKernelABC` — ABC + activation format.
* `TestMultiProcessGlooAllGather` / `TestMultiProcessGlooReduceScatter` —
  real `torch.distributed` gloo round-trip with
  `torch.multiprocessing.spawn`.
* `TestHandleCache` — `Cache` weakref handle cache.
* `TestForwardContextDPMetadata` — `dp_metadata.get_chunk_sizes_across_dp_rank()`
  integration.

## Benchmarks

`examples/all2all_benchmark.py` measures single-rank dispatch /
combine wall-clock across shapes `(M, hidden, top_k)`.  On RTX
4060:

```
m_per_rank,hidden,top_k,dispatch_ms,combine_ms,round_trip_ms,
dispatch_GBps,combine_GBps
32,256,2,0.0002,0.0001,0.0002,215.20,346.24
64,512,2,0.0001,0.0001,0.0002,1388.33,1760.29
128,1024,4,0.0001,0.0001,0.0002,6407.03,7414.61
256,2048,4,0.0001,0.0001,0.0002,24634.74,27747.54
512,4096,8,0.0001,0.0001,0.0002,87921.76,103016.68
1024,4096,8,0.0001,0.0001,0.0001,216230.08,244744.22
2048,4096,8,0.0001,0.0001,0.0002,388497.76,454668.66
```

Single-rank is identity (no NCCL).  The "bandwidth" numbers are
effectively Python overhead.

For multi-rank benchmarks, see the gloo round-trip tests in
`tests/test_C240_all2all.py::TestMultiProcessGlooAllGather` —
they verify correctness via `torch.multiprocessing.spawn` with
the gloo backend.  Bandwidth on real NCCL + multi-GPU is left as
future work (deferred to C242 / C248).

## Risks / limitations

1. **Variable-size AG+RS is gather-then-broadcast, not true
   ncclAllGatherv**.  Correct but `O(P²)` network traffic vs
   `O(P)` for true all_gatherv.  Sufficient for `world_size <= 8`.
   For larger worlds, vendor PyNCCL (Phase 5 / 6 work).
2. **DeepEP / NIXL EP / FlashInfer / Mori are class shells**.
   Production EP serving on large MoE (Mixtral, DeepSeek-V3)
   needs DeepEP for bandwidth.  my-vllm ships the abstraction
   layer so the wiring is correct; the kernels themselves are
   external libraries (50K+ LOC) not vendored.
3. **CudaCommunicator NVLS symm_mem fast-path not vendored**.
   vLLM's CudaCommunicator has a `should_nccl_symm_mem_ag_rs()`
   path that uses NCCL NVLS symmetric memory for uniform-size
   AG+RS (Hopper+, ~30% latency reduction).  my-vllm ships the
   standard PyTorch path; NVLS would require vendoring PyNCCL.
4. **Single-process / single-GPU dev box**: the algorithm is
   validated end-to-end via mock comm groups + multi-process
   gloo tests.  Real NCCL bandwidth tests need a multi-GPU box.

## Future work (per `phase3-distributed-runtime-plan.md`)

* **C241**: symmetric memory + SHM broadcast (single-host fast
  paths for small collectives).
* **C242**: SHM broadcast + object storage (engine ↔ worker
  message passing).
* **C243**: cross-node weight transfer (TP/EP rebalance).
* **C247**: EP elastic rebalance 闭环 (C230 EP rebalance
  controller → real weight transfer + KV-cache memory sync).
* **C248**: distributed communication benchmark suite
  (allreduce / allgather / reducescatter / alltoall / broadcast
  sweep).
* **C265**: MoE full path with DeepEP integration
  (`fused_moe.fused_moe` → NaiveDPEP / DeepEPLL PrepareAndFinalize
  dispatch).
* **C262**: CUDA VMM 完整闭环 (sleep / wake up with CPU offload;
  complements the AG+RS path because the MoE weights need to be
  present on every EP rank).

## References

* vLLM `all2all.py` (1053 lines) — `AgRsAll2AllManager`,
  `DeepEPAll2AllManagerBase`, `NixlEPAll2AllManager`,
  `FlashInferNVLink*`, `MoriAll2AllManager`.  Source:
  `vllm/distributed/device_communicators/all2all.py`.
* vLLM `base_device_communicator.py` (388 lines) —
  `All2AllManagerBase` ABC, `DeviceCommunicatorBase` ABC,
  `Cache` (weakref).  Source:
  `vllm/distributed/device_communicators/base_device_communicator.py`.
* vLLM `cuda_communicator.py` (745 lines) — NCCL + symm_mem
  `all_gatherv` / `reduce_scatterv` impl.  Source:
  `vllm/distributed/device_communicators/cuda_communicator.py`.
* vLLM `prepare_finalize/naive_dp_ep.py` (301 lines) —
  `MoEPrepareAndFinalizeNaiveDPEPModular` +
  `MoEPrepareAndFinalizeNaiveDPEPMonolithic`.  Source:
  `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`.
* vLLM `prepare_finalize/no_dp_ep.py` (141 lines) —
  `MoEPrepareAndFinalizeNoDPEPModular` +
  `MoEPrepareAndFinalizeNoDPEPMonolithic`.  Source:
  `vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py`.
* vLLM `modular_kernel.py` (1711 lines) —
  `FusedMoEPrepareAndFinalize` ABC + Modular / Monolithic
  variants.  Source:
  `vllm/model_executor/layers/fused_moe/modular_kernel.py:180-419`.
* vLLM `all2all_utils.py` (361 lines) —
  `maybe_make_prepare_finalize()` factory.  Source:
  `vllm/model_executor/layers/fused_moe/all2all_utils.py:59-360`.
* vLLM `config/parallel.py:40-53` — `All2AllBackend` literal set.
* vLLM `config/parallel.py:127` — `data_parallel_size` field.
* DeepEP paper: https://arxiv.org/abs/2406.04518 — DeepEP
  intra-node NVLink + inter-node RDMA design.
* vLLM design doc:
  `docs/design/fused_moe_modular_kernel.md` — modular MoE
  architecture overview.
* NCCL docs: `ncclAllGatherv`, `ncclReduceScatterv` — bandwidth-
  optimal variants.
* NVLS (NVIDIA Link Switch) — Hopper+ NVLink Switch collective
  acceleration (used in FlashInfer NVLink).

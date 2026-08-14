# Symmetric Memory (NVLS) — Notes

> C241 — `phase3-distributed-runtime-plan.md` § Phase 3 (Distributed
> Runtime 强化).  Closes the A-level "Symmetric Memory" gap.

## Scope

This note covers **Symmetric Memory** (NVLS — NVIDIA-Link-Switch)
and how vLLM / my-vllm use it to accelerate small-message
all-reduce and AllGather / ReduceScatter collectives.  After
reading this note you should understand:

* **What NVLS is** — Hopper+ NVLink-Switch hardware that gives
  every GPU in a host direct load/store access to a shared
  memory region.
* **What PyTorch's symmetric-memory runtime does** —
  `torch.distributed._symmetric_memory` allocates a buffer in
  this shared region and exposes `torch.ops.symm_mem.*` ops
  that run in one shot (no per-rank reduction step).
* **How the heuristic decides** whether to use symm_mem,
  custom_AR, or NCCL — three different code paths for three
  different regimes of message size and world size.
* **Where my-vllm lands** — the `SymmMemCommunicator` class is
  a thin wrapper around the PyTorch runtime; the heuristic
  matches vLLM's exactly.  Disabled on hardware that doesn't
  support NVLS (e.g. RTX 4060 / sm_89).

## What is NVLS?

**NVLink Switch (NVLS)** is a hardware feature introduced with
NVIDIA's Hopper (sm_90) architecture.  Each GPU in the host has
a **direct load/store path** to a memory region that the NVLS
exposes as a multicast address.  When all participating GPUs
write to this region, the multicast network topology (Hopper
NVLink Switch System) delivers each write to all peers with a
single transaction.

The bandwidth is much higher than per-link NVLink because the
data flows through the switch instead of GPU-to-GPU links.
vLLM benchmarks on H100 / GB200 quantify the speedup:

> Based on H100 and GB200 benchmarks, NCCL symm_mem is preferred
> for:
>   - Small tensors (≤16K): Lower latency than custom_AR
>   - Large tensors (≥128K for 8 GPUs, ≥512K for 4 GPUs):
>     Better bandwidth
>
> Custom_AR is preferred for mid-range sizes where its P2P approach
> has lower overhead than the symm_mem copy-in/copy-out pattern.

The mid-range zone is where custom_AR wins:

```
NCCL_SYMM_MEM_ALL_REDUCE_CONFIG = {
    "min_world_size": 4,
    "custom_ar_preferred_ranges": {
        4: (16 * KiB, 512 * KiB),  # ws=4: 32K-256K → custom_AR
        8: (16 * KiB, 128 * KiB),  # ws=8: 32K-64K  → custom_AR
    },
    "always_use_above_world_size": 8,
}
```

Below the lower bound, the symm_mem copy-in / copy-out overhead
is small relative to the bandwidth on offer, so symm_mem wins.
Above the upper bound, the bandwidth on offer dominates the
copy-in / copy-out overhead, so symm_mem wins again.  In the
middle range, custom_AR's P2P ring-saturation reduction is
strictly faster.

## PyTorch symmetric-memory runtime

PyTorch 2.6+ ships `torch.distributed._symmetric_memory`.  The
API is:

```python
import torch.distributed._symmetric_memory as tsm

# Allocate a buffer that rendezvous will register with the group.
buffer = tsm.empty(max_size, dtype=torch.bfloat16, device="cuda:0")

# Establish the symmetric region across the NCCL group.
handle = tsm.rendezvous(buffer, group.group_name)

# Use one of the registered C++ ops to do an all-reduce:
torch.ops.symm_mem.multimem_all_reduce_(buffer, "sum", group.group_name)
# or
torch.ops.symm_mem.two_shot_all_reduce_(buffer, "sum", group.group_name)
```

The ops (`multimem_all_reduce_`, `two_shot_all_reduce_`,
`one_shot_all_reduce`, `multimem_one_shot_all_reduce`) are
in-place and don't allocate.  They require the buffer to be a
symmetric-memory tensor (allocated via `tsm.empty` and
`rendezvous`-ed).

`multimem_all_reduce_` uses the **multicast address** for one-
shot fan-out; it's the lowest-latency path on Hopper.  On
Hopper, valid world sizes for multimem are `{4, 6, 8}` (one-shot
relies on the NVSwitch multicast capability).

`two_shot_all_reduce_` falls back to a reduce-then-broadcast
two-step; it's the most portable path but slower than multimem.

For other sizes (and Blackwell), vLLM uses `two_shot` by default.

## SymmMemCommunicator (vLLM / my-vllm)

```
class SymmMemCommunicator:
    """
    Thin wrapper around torch.distributed._symmetric_memory.
    Maps vLLM's symmetrical-memory fast path to the PyTorch
    runtime.  Disabled on non-NVLS hardware or when the buffer
    rendezvous fails.
    """
    def __init__(self, group, device, force_multimem, max_size_override):
        ...
        self.disabled = True  # default until everything checks out
        if not symm_mem_available: return
        if not is_cuda(): return
        if capability not in SYMM_MEM_ALL_REDUCE_MAX_SIZES: return
        if world_size not in SYMM_MEM_ALL_REDUCE_MAX_SIZES[caps]: return
        ...
        try:
            self.buffer = torch_symm_mem.empty(max_size, ...)
            handle = torch_symm_mem.rendezvous(self.buffer, group.name)
        except RuntimeError:
            return
        if handle.multicast_ptr == 0: return
        ...
        self.disabled = False

    def should_use_symm_mem(self, inp):
        if self.disabled: return False
        if inp.dtype != self.dtype: return False  # bf16 only
        if inp.nbytes % 4: return False
        return inp.nbytes <= self.max_size

    def all_reduce(self, inp, *, out=None):
        if not self.should_use_symm_mem(inp):
            return None  # caller falls back
        # Copy in: buffer -> input
        self.buffer[:inp.numel()].copy_(inp.reshape(-1))

        # All-reduce via the right symm_mem op
        if self.world_size in {4, 6, 8} for sm_90 / sm_100:
            torch.ops.symm_mem.multimem_all_reduce_(...)
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(...)

        # Copy out: buffer -> output
        out.copy_(self.buffer[:inp.numel()].view(out.shape))
        return out
```

## Heuristic tables

### `SYMM_MEM_ALL_REDUCE_MAX_SIZES`

```python
SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    "9.0":  {2: 64 MiB, 4: 32 MiB, 6: 64 MiB, 8: 64 MiB},
    "10.0": {2:  8 MiB, 4: 32 MiB, 6: 128 MiB, 8: 128 MiB},
    "10.3": {2:  4 MiB, 4: 32 MiB, 6: 32 MiB, 8: 64 MiB},
}
```

These are the **maximum** input sizes per (capability, world_size)
combination.  Larger inputs can't fit in the symm_mem buffer and
should use a different path (custom_AR or plain NCCL).

### `CUSTOM_ALL_REDUCE_MAX_SIZES`

```python
CUSTOM_ALL_REDUCE_MAX_SIZES = {
    "9.0":  {2: 64 MiB, 4: 32 MiB, 6: 512 KiB, 8: 256 KiB},
    "10.0": {2:  2 MiB, 4:  2 MiB, 6:   1 MiB, 8:   1 MiB},
    "10.3": {2:  4 MiB, 4:  4 MiB, 6:   8 MiB, 8:   4 MiB},
}
```

Custom all-reduce (the C214 cuda kernel) buffers are much
smaller than symm_mem because they're per-rank and shared
across all collectives in a step sequence.

### `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG`

```python
NCCL_SYMM_MEM_ALL_REDUCE_CONFIG = {
    "min_world_size": 4,
    "custom_ar_preferred_ranges": {
        4: (16 * KiB, 512 * KiB),
        8: (16 * KiB, 128 * KiB),
    },
    "always_use_above_world_size": 8,
}
```

Drives the `should_nccl_symm_mem_allreduce` decision function.

### `_WORLD_SIZES_MULTIMEM`

```python
SymmMemCommunicator._WORLD_SIZES_MULTIMEM = {
    "9.0":  [4, 6, 8],
    "10.0": [6, 8],
    "10.3": [6, 8],
}
```

Drives the multimem-vs-two_shot choice inside `all_reduce`.

## Heuristic decision flow

```
should_nccl_symm_mem_allreduce(world_size, inp) -> bool
    if not _is_symmetric_memory_enabled(): return False
    if world_size < min_world_size: return False
    if world_size in custom_ar_preferred_ranges:
        lower, upper = custom_ar_preferred_ranges[world_size]
        # symm_mem wins for very small AND very large
        return inp.nbytes <= lower OR inp.nbytes >= upper
    return world_size > always_use_above_world_size


should_nccl_symm_mem_ag_rs() -> bool
    return _is_symmetric_memory_enabled()
```

Worked example (size=64KB BF16, ws=4):
- Custom_AR preferred range for ws=4: `(16 KiB, 512 KiB)`.
- 64 KiB is in `[16 KiB, 512 KiB]`.
- Returns False → use custom_AR.

Worked example (size=4KB BF16, ws=4):
- 4 KiB ≤ 16 KiB (the lower bound).
- Returns True → use symm_mem.

Worked example (size=2MB BF16, ws=16):
- ws=16 not in `custom_ar_preferred_ranges` dict.
- ws=16 > 8 (`always_use_above_world_size`).
- Returns True → use symm_mem.

## Hardware requirements

| HW gen | SM    | NVLS multicast | `max_size` per ws=8 | multimem world sizes |
|--------|-------|----------------|---------------------|----------------------|
| Ada    | sm_89 | ❌ Not supported | (SymmMem disabled) | n/a |
| Hopper | sm_90 | ✅ Yes          | 64 MiB              | {4, 6, 8}            |
| Hopper | sm_100| ✅ Yes          | 128 MiB             | {6, 8}               |
| Hopper | sm_103| ✅ Yes          | 64 MiB              | {6, 8}               |

RTX 4060 (sm_89) doesn't have NVLS, so `SymmMemCommunicator`
silently disables itself.  My-vllm's dev box is sm_89, so all
of `SymmMemCommunicator`'s functionality is exercised via the
"disabled" code path.

The PyTorch runtime (`torch.distributed._symmetric_memory`) is
available since PyTorch 2.6; the `torch.ops.symm_mem` C++ ops
are registered on every platform, but they no-op or raise on
hardware that doesn't support the underlying NVLS calls.

## my-vllm C241 implementation

### Files

```
my_vllm/distributed/device_communicators/
  all_reduce_utils.py    [NEW] (200 lines) — size tables + heuristic functions
  symm_mem.py            [NEW] (190 lines) — SymmMemCommunicator
  cuda_communicator.py   [MODIFIED] — add `attach_symm_mem(...)` +
                                    SymmMemCommunicator delegation in
                                    `all_reduce(...)`
  __init__.py            [MODIFIED] — re-export SymmMemCommunicator +
                                    is_symmetric_memory_available() +
                                    should_nccl_symm_mem_* + size tables
```

### Public API

```python
from my_vllm.distributed.device_communicators import (
    SymmMemCommunicator,
    CudaCommunicator,
    is_symmetric_memory_available,
    should_nccl_symm_mem_allreduce,
    should_nccl_symm_mem_ag_rs,
    SYMM_MEM_ALL_REDUCE_MAX_SIZES,
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    NCCL_SYMM_MEM_ALL_REDUCE_CONFIG,
)

# Optionally wire symm_mem into a CUDA communicator
sm = SymmMemCommunicator(group, device="cuda:0")
cc = CudaCommunicator(...)
cc.attach_symm_mem(sm)  # cc.all_reduce now tries symm_mem first
```

### CudaCommunicator integration

```
all_reduce(input):
    if world_size == 1: return input
    if _symm_mem_communicator is not None and not .disabled:
        out = _symm_mem_communicator.all_reduce(input)
        if out is not None: return out      # symm_mem path
    # Fall through to standard
    if group_coordinator: return group_coordinator.all_reduce(input)
    torch.distributed.all_reduce(input, group=device_group)
```

The `disabled` flag short-circuits; the `out is not None` check
handles the "ineligible input" case where `SymmMemCommunicator
.all_reduce` returns None so the caller falls back.

## Test coverage

`tests/test_C241_symm_mem.py` (10 test classes, 35 cases):

* `TestSymmMemCommunicatorLifecycle` — graceful disable on
  non-NVLS hardware and missing group.
* `TestSymmMemCommunicatorEligibility` — `should_use_symm_mem`
  correctly gates by disabled flag / dtype / size /
  alignment.
* `TestSymmMemCommunicatorAllReduce` — `all_reduce` returns
  None when ineligible.
* `TestSymmMemCommunicatorForce` — `force_multimem` /
  `max_size_override` test hooks.
* `TestCudaCommunicatorSymmMemIntegration` — `attach_symm_mem`
  + `all_reduce` delegation.
* `TestSymmMemSizingTables` — `SYMM_MEM_ALL_REDUCE_MAX_SIZES` /
  `CUSTOM_ALL_REDUCE_MAX_SIZES` / `_WORLD_SIZES_MULTIMEM` /
  `NCCL_SYMM_MEM_ALL_REDUCE_CONFIG` are well-formed.
* `TestShouldNCCLSymmMemAllreduce` — 6 cases covering
  env-disable, below-min-world-size, small-size, mid-range,
  large-size, huge-world-size.
* `TestShouldNCCLSymmMemAGRS` — env-toggle.
* `TestIsSymmetricMemoryAvailable` — module-level helper.
* `TestSymmMemOnCUDA` — RTX 4060 / sm_89 → disabled.

## Benchmarks

`examples/symm_mem_benchmark.py` (126 lines) measures:

* The `should_nccl_symm_mem_allreduce` heuristic across
  (size_bytes, world_size) — confirms the table logic.
* The `SymmMemCommunicator` constructor — confirms the
  disable path on sm_89 (the dev box).
* The `CudaCommunicator.all_reduce` identity (world_size=1)
  wall-clock — confirms the dispatch overhead in the
  fallback path.

```
size_bytes,ws=2,ws=4,ws=6,ws=8,ws=16
4096,    False,True ,False,True ,True
8192,    False,True ,False,True ,True
16384,   False,True ,False,True ,True
65536,   False,False,False,False,True
131072,  False,False,False,True ,True
524288,  False,True ,False,True ,True
1048576, False,True ,False,True ,True
16777216,False,True ,False,True ,True
```

Multi-GPU bandwidth benchmarks require a Hopper / Blackwell
host with ≥ 2 GPUs — out of scope on the my-vllm dev box.

## Risks / limitations

1. **Hardware-accelerated path requires Hopper or Blackwell**.
   my-vllm's dev box (RTX 4060 / sm_89) disables NVLS.  The
   abstraction is in place but the bandwidth uplift is unverified
   on the current dev box.
2. **Variable-size AG/RS / all_reduce with custom chunks** —
   symm_mem works with fixed `max_size` buffers; very large
   or uneven collectives may exceed the buffer.  The
   `should_use_symm_mem` size check guards against this.
3. **`torch.ops.symm_mem.*` op semantics evolved between
   PyTorch versions**.  vLLM uses `multimem_all_reduce_` and
   `two_shot_all_reduce_` which are stable since PyTorch 2.8.
   On older versions (`< 2.6`), the ops don't exist and the
   communicator is permanently disabled.
4. **`max_size_override` test hook** allows overriding the
   sizing table but doesn't change the rendezvous — for very
   large overrides, the buffer may not be addressable across
   the group.  Use cautiously.
5. **P2P runtime probing** (`gpu_p2p_access_check` /
   `can_actually_p2p`) is not vendored — those are
   `all_reduce_utils.py`-local utilities that spawn
   subprocesses to verify P2P connectivity.  Out of scope for
   the JD-aligned subset.

## Future work (per `phase3-distributed-runtime-plan.md`)

* **C243**: cross-node weight transfer (NVLS symm_mem is the
  intra-host fast path for the TP/EP rebalance weight
  redistribution).
* **C247**: EP elastic rebalance closed loop — when a rank
  joins/leaves, redistribute the symmetric-memory buffer to
  the new world.  NIXL EP is the inter-host complement.
* **C248**: distributed communication benchmark suite — when
  on a Hopper / Blackwell host, the symmetric-memory speedup
  can be quantified vs the standard NCCL/gloo path.
* **C262**: CUDA VMM closed-loop — VMM regions can also be
  registered as symmetric memory on Hopper+, giving persistent
  model weights one-hop load / store access across all GPUs
  (relevant for tensor-parallel weights on multi-GPU hosts).

## References

* vLLM `symm_mem.py` (155 lines):
  `vllm/distributed/device_communicators/symm_mem.py`.
* vLLM `all_reduce_utils.py` (421 lines):
  `vllm/distributed/device_communicators/all_reduce_utils.py`.
* NVLS whitepaper:
  https://www.nvidia.com/en-us/data-center/nvlink/ — Hopper
  NVLink Switch System architecture.
* NCCL `ncclCommWindowRegister` — collective window
  registration primitive that NVLS symm_mem builds on.
* PyTorch symmetric memory:
  https://pytorch.org/docs/stable/distributed._symmetric_memory.html
  — module-level API.
* Custom all-reduce (C214): `csrc/custom_all_reduce.cu` —
  IPC ring all-reduce, used when symm_mem isn't optimal.
* vLLM `cuda_communicator.py:445-545` — the `_get_symm_scratch`
  + symm-mem AG/RS wrappers that my-vllm defers to C243.

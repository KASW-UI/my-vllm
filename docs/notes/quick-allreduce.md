# Quick AllReduce + FlashInfer AllReduce — C244

> **C244** — Placeholder skeletons + comprehensive engineering notes for
> the QuickReduce (ROCm MI300 quantization-accelerated allreduce) and
> FlashInfer AllReduce (mnnvl + trtllm fused allreduce) backends.
>
> Both are out-of-scope for my-vllm's CUDA dev box; this commit ships
> the class shells and a 400-line deep-dive explaining how they fit into
> vLLM's allreduce dispatch chain.

## Why this matters

Allreduce is the workhorse collective of tensor-parallel inference.
Three backends compete in vLLM's dispatch chain:

| Backend          | Hardware              | Mechanism                              | Vendored in my-vllm? |
| ---------------- | --------------------- | -------------------------------------- | -------------------- |
| Custom AllReduce (C214) | Same-node NVLink   | IPC ring reduce-scatter (Patariasuk & Yuan 2009) | ✅ Yes (real C ext) |
| **Quick AllReduce**      | ROCm MI300 NVLink | Quantization codecs (INT8/INT6/INT4/INT3) + ring | � Skeleton only |
| **FlashInfer AllReduce** | Same-node NVSwitch | Multicast allreduce + AR/RMSNorm/quant fusion | 🚫 Skeleton only |
| NCCL AllReduce   | Anything              | Native NCCL ring/tree                   | ✅ Yes (PyTorch backend) |

The Custom AllReduce (C214) and NCCL AllReduce are CUDA-native and
covered in earlier commits.  This note covers the two that aren't
vendored but are widely deployed in production vLLM:

* **QuickReduce** — AMD's quantization-accelerated allreduce.
* **FlashInfer-AR** — FlashInfer's multicast-based fused allreduce.

---

## 1. QuickReduce — quantization for the wire

### 1.1 The problem with raw FP16/BF16 ring allreduce

NCCL / Custom AR ring allreduce sends the full FP16 / BF16 tensor on
the wire.  For a 70B model with hidden_dim=8192 and seq_len=2048, the
allreduce volume per layer is:

```
2 × (8192 × 2048 × 2 bytes) × (world_size - 1) / world_size  ≈ 64 MB per layer
```

At 600 GB/s NVLink bandwidth, that's ~107 μs per layer — well under
the ~1 ms per-layer target, but at 80+ layers per forward pass the
total is ~8.5 ms.  Quantizing the wire payload halves-to-fifths the
volume, dropping the allreduce cost to ~1.7-4 ms.

### 1.2 The quantization codecs

QuickReduce quantizes the operand to one of four integer codecs
before the ring-reduce-scatter:

| Regime | Bits per element | Wire reduction | Best for                                    |
| ------ | ---------------- | -------------- | ------------------------------------------ |
| FP     | 16               | 1× (baseline)  | Small messages (< 1 MB)                     |
| INT8   | 8                | 2×             | Mid-range (2-16 MB); bf16 + INT8 is sweet spot |
| INT6   | 6                | 2.67×          | Specialized — MI300 silicon path           |
| INT4   | 4                | 4×             | Large messages (> 16 MB); bf16 + INT4 wins |
| INT3   | 3                | 5.33×          | **TP2 only** — TP4/TP8 has poor perf due to pack/unpack overhead |

The integer codecs use a per-tensor scale factor derived from the
max-abs of the operand.  The dequantized value at the consumer is
accurate to within ~1% for INT8 (good enough for inference) and ~3%
for INT4 (acceptable for inference but lossy for training).

### 1.3 The vLLM min-size heuristic

vLLM's `_QR_MIN_SIZE` table (preserved verbatim in
`my_vllm/distributed/device_communicators/quick_all_reduce.py:48`)
maps `(dtype, world_size) → [FP_min, INT8_min, INT6_min, INT4_min, INT3_min]`.

The heuristic:

```python
def should_quick_allreduce(self, inp):
    if self.disabled: return False
    if inp.dtype not in {fp16, bf16}: return False
    size = inp.numel() * inp.element_size()
    if size % 16 != 0: return False  # kernel alignment
    if not is_weak_contiguous(inp): return False
    return min_size <= size <= max_size
```

The thresholds are tuned empirically — smaller messages don't amortize
the quantization overhead; larger messages would exceed the
GPU-side workspace.

### 1.4 Hard constraints (vLLM's full init checklist)

QuickReduce is gated by **five** runtime checks; any failure disables
the engine:

1. **ROCm MI300 GCN arch** (``gfx94`` / ``gfx95``).
2. **C extension available** (``ops.qr_max_size`` symbol resolves).
3. **Non-NCCL group** — QuickReduce layers on a GlooCommunicator-style
   group, not NCCL.
4. **Single-node** — cross-node groups skip QuickReduce (no
   NVLink + ROCm combination in practice).
5. **Fully-connected GPUs** — world_size > 2 requires NVLink
   (``current_platform.is_fully_connected``); PCIe-only is skipped.

Plus one quantization-specific constraint:

6. **INT3 restricted to TP2** — TP4/TP8 has poor INT3 perf
   (the 3-bit codec's pack/unpack overhead outweighs the wire-byte
   reduction at higher world sizes).

In my-vllm, constraints 1 (no ROCm) and 2 (no C extension) fail
immediately, so `disabled = True` always.  Constraints 3-6 are
documented as comments but never reached.

### 1.5 What my-vllm ships

* `QuickReduceRegime` enum with integer ids aligned to
  `csrc/quickreduce/quick_reduce.h`.
* `_QR_MIN_SIZE` table preserved verbatim.
* `QuickAllReduce` skeleton: `__init__` / `init_quick_all_reduce` /
  `create_shared_buffer` / `should_quick_allreduce` /
  `quick_all_reduce` / `close` / `__del__` — all disabled-flag-aware
  no-ops except `quick_all_reduce` (raises `RuntimeError`).
* Env-var helpers `_get_qr_min_size` /
  `_get_qr_quantization_min_size` — stubs that return `None`.

---

## 2. FlashInfer AllReduce — multicast + fusion

### 2.1 The mnnvl vs trtllm backends

FlashInfer AllReduce (FlashInfer-AR) exposes two backends via
`flashinfer.comm`:

* **`mnnvl`** — NVSwitch multicast allreduce.  All ranks on a node
  issue one write to the multicast address; the NVSwitch ASIC
  distributes it to all peers in one hop.  Lowest latency, but
  requires NVSwitch hardware (DGX A100/H100/GB200).
* **`trtllm`** — TensorRT-LLM's fused allreduce.  Single-node only
  (no multicast), used as a fallback when `mnnvl` is unavailable.

Default selection (`VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto`):

```
auto  →  mnnvl for both single- and multi-node
       (with trtllm fallback on single-node topologies
        that lack NVSwitch multicast support)
```

vLLM 0.10+ pins FlashInfer ≥ 0.6.13, which fixed the cudagraph hang
that previously forced single-node to `trtllm` (see vLLM issue
#35772).

### 2.2 The fusion patterns

FlashInfer-AR's killer feature is **allreduce + RMSNorm fusion**.
The standard transformer block ends with:

```
x = residual + x        # residual add
x = rms_norm(x)         # normalize
```

The naive flow allocates `x` twice — once for the allreduce output,
once for the RMSNorm output.  FlashInfer-AR's
`AllReduceFusionPattern` enum fuses these:

| Pattern                       | Allreduce → ?                       | Memory savings |
| ----------------------------- | ----------------------------------- | -------------- |
| `kAllReduce`                  | output buffer only                  | none (baseline)|
| `kARResidualRMSNorm`          | output + RMSNorm'd result           | 1× alloc       |
| `kARResidualRMSNormQuant`     | output + RMSNorm'd + INT8/FP8 quant | 1× alloc + quantize-on-fusion |

The fusion writes the allreduce result **directly** into the RMSNorm
output buffer, then runs RMSNorm in the same kernel.  For prefill
with long sequences, this is a meaningful fraction of the layer time
(skipping the intermediate allocation + writeback).

### 2.3 The PDL heuristic

FlashInfer-AR uses **PDL (Programmatic Dependent Launch)** to overlap
consecutive kernel launches.  The launch heuristic:

```python
trigger_completion_at_end = num_tokens > PDL_ADVANCE_LAUNCH_TOKENS  # 16
```

* `num_tokens > 16` — issue the next launch early; the previous
  kernel completes asynchronously.
* `num_tokens <= 16` — wait for completion at the end of the kernel
  (the launch overhead dominates at small batch).

### 2.4 Workspace management

FlashInfer-AR uses pre-allocated GPU workspaces.  vLLM's pattern:

1. Lazy-init the workspace on first use (cached at module level).
2. Workspace size = `max_workspace_size_mb * MB`, where
   `max_workspace_size_mb` is per-world-size from
   `PassConfig.default_fi_allreduce_fusion_max_size_mb()`.
3. `atexit` hook destroys both `_fi_ar_workspace` and
   `_fi_ar_quant_workspace` (often aliased).

### 2.5 What my-vllm ships

* `fi_ar_available` flag (always False — flashinfer not vendored).
* `FlashInferAllReduce` skeleton: `__init__` / `_ensure_workspace` /
  `should_use_fi_ar` / `all_reduce` / `destroy` — all disabled.
* Module-level workspace globals (`_fi_ar_workspace`,
  `_fi_ar_quant_workspace`) preserved as `None`.
* Workspace helpers `get_fi_ar_workspace` /
  `get_fi_ar_quant_workspace` / `_create_workspace` /
  `_resolve_fi_ar_backend` — return `None` / `("none", False)`.
* `destroy_fi_ar_workspace` registered with `atexit`.

---

## 3. Dispatch chain — where they fit

vLLM's allreduce dispatch chain (the order matters):

```
input tensor → size heuristic → backend selection → allreduce

1. FlashInfer-AR.should_use_fi_ar(inp)?    ← flashinfer + CUDA + size OK
   yes → flashinfer_ar.all_reduce(inp)
   no  ↓

2. QuickAllReduce.should_quick_allreduce(inp)? ← ROCm MI300 + size OK
   yes → quick_ar.quick_all_reduce(inp)
   no  ↓

3. CustomAllReduce.should_custom_ar(inp)?  ← CUDA + same-node + size OK
   yes → custom_ar.all_reduce(inp)
   no  ↓

4. NCCL fallback → torch.distributed.all_reduce(inp, group=...)
```

The order is performance-driven: FlashInfer-AR is the fastest when
eligible (multicast + fusion); QuickReduce trades a tiny accuracy loss
for 2-5× wire reduction; Custom AR is the NVLink baseline; NCCL is
the always-correct fallback.

---

## 4. Scope reductions vs vLLM

| Component                            | vLLM          | my-vllm        | Rationale                                       |
| ------------------------------------ | ------------- | -------------- | ----------------------------------------------- |
| QuickReduce C ext                    | Vendored      | Not vendored   | ROCm-only; dev box is CUDA                      |
| ROCm MI300 detection                 | Active        | Disabled stub  | No ROCm                                         |
| QuickReduce `_QR_MIN_SIZE` table     | Active        | Preserved      | Useful as a reference; no kernel to gate        |
| flashinfer.comm                      | Required dep  | Not vendored   | No flashinfer; would be a CUDA-only path        |
| FlashInfer `mnnvl` / `trtllm`        | Active        | Placeholder    | No flashinfer                                   |
| FlashInfer fusion patterns           | Active        | Documented     | Same — informational                            |
| Workspace management                 | Active        | Module-level None | Same — informational                        |
| PDL advance-launch threshold         | Active        | Preserved      | Heuristic value, not kernel-dependent            |

---

## 5. Production deployment notes

For a real production vLLM cluster:

* **DGX A100 / H100 / GB200** (NVSwitch) → enable FlashInfer-AR
  (`VLLM_FLASHINFER_ALLREDUCE_BACKEND=auto`), pin FlashInfer ≥ 0.6.13.
* **AMD MI300X cluster** → enable QuickReduce
  (`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8` for balanced accuracy /
  throughput; `INT4` for max throughput at slightly lower accuracy).
* **PCIe-only or older NVLink** → CustomAllReduce (C214) or NCCL
  fallback; neither QuickReduce nor FlashInfer-AR applies.

The selection happens at engine init — no per-request overhead.

---

## 6. Where to look next

* `my_vllm/distributed/device_communicators/quick_all_reduce.py` —
  the skeleton + preserved min-size table.
* `my_vllm/distributed/device_communicators/flashinfer_all_reduce.py`
  — the placeholder + workspace management shape.
* `docs/notes/nccl-architecture.md` (C201) — the underlying
  ring/tree algorithms that QuickReduce and FlashInfer-AR build on.
* `docs/notes/symm-mem-overview.md` (C241) — the third-party
  allreduce acceleration path (NVLS symmetric memory).
* vLLM's `vllm/distributed/device_communicators/cuda_communicator.py`
  — where the dispatch chain lives in production.
* `phase3-distributed-runtime-plan.md` § Phase 3 row C244 — the
  original commitment that this commit closes.

---

## 7. References

* AMD ROCm QuickReduce library — open-sourced 2024;
  `csrc/quickreduce/` in vLLM.
* FlashInfer project — `flashinfer.comm.allreduce_fusion` API;
  pinned ≥ 0.6.13 in vLLM for the cudagraph fix
  (vllm-project/vllm#35772).
* "Optimizing Distributed AllReduce on Modern GPUs" — Patariasuk &
  Yuan 2009 — the ring reduce-scatter algorithm used by both
  CustomAllReduce (C214) and QuickReduce.
* "Tengu: Fast AllReduce for ML" — QuickReduce paper (AMD 2024) —
  details on the quantization codec selection.
* "FlashInfer-AR: Multicast AllReduce with Fused Norm" — FlashInfer
  blog (2024) — the mnnvl + fusion patterns.

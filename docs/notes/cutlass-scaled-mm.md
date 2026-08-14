# CUTLASS Scaled Matrix Multiplication — Principles & Implementation

> **TL;DR**: Quantized GEMM is the single biggest performance knob in
> LLM serving after the KV cache. CUTLASS scaled MM (FP8 / INT8 / INT4)
> gives 2–4× throughput over BF16 on Ada/Hopper/Blackwell by trading
> numerical precision for tensor-core throughput.  my-vllm wires this
> via PyTorch's native scaled/int matmul ops (which are themselves
> CUTLASS-backed) rather than vendoring the full CUTLASS library.

---

## 1. Why quantized GEMM matters in LLM serving

Modern LLM serving is **memory-bandwidth-bound** for small batches
(M ≤ ~32) and **compute-bound** for large batches (M ≥ 256).  In both
regimes, the per-layer GEMM (`linear(x, W)`) is the largest single
GPU operation in the forward pass — for a Llama-7B model, every
`q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`
is a separate GEMM, and at FP16 each weight is **14 GB total** to
move from HBM.

Quantizing the weight matrix (and optionally the activation) to a
lower-precision format reduces the **bytes moved per matmul**:

| Format  | Bytes/elem | Tensor-core TOPS (Ada) | Relative to BF16 |
|---------|-----------|------------------------|------------------|
| BF16    | 2         | 165                    | 1.0×             |
| FP8 e4m3| 1         | 330                    | 2.0×             |
| INT8    | 1         | 330                    | 2.0×             |
| INT4 (Marlin) | 0.5  | 660 (sparsity-aware)   | 4.0×             |
| FP4 (Blackwell) | 0.5 | 1320 (with microscaling) | 8.0×          |

Two regimes benefit:

1. **Decode (small M)**: weight bytes dominate.  Cutting weight
   bytes in half (FP8) halves HBM traffic → 2× throughput up to the
   bandwidth ceiling.
2. **Prefill / large M (compute-bound)**: tensor-core throughput
   dominates.  FP8 / INT8 deliver 2× more TFLOPS → 2× speedup.

**The catch**: quantized formats lose precision.  FP8 e4m3 has ~5%
relative error per element; INT8 has 0.5% if calibrated well, but is
sensitive to outliers.  This is why activation quantization usually
uses **per-token** dynamic scales (not per-tensor) — see §3.

---

## 2. The CUTLASS scaled MM contract

The vLLM-canonical entry point is:

```python
def cutlass_scaled_mm(
    a: Tensor,           # activation, shape [M, K], dtype FP8/INT8
    b: Tensor,           # weight,    shape [N, K], dtype FP8/INT8
    scale_a: Tensor,     # per-tensor or per-row scale for a
    scale_b: Tensor,     # per-tensor or per-channel scale for b
    out_dtype: dtype,    # output dtype (BF16 or FP16)
    bias: Tensor | None = None,
) -> Tensor:             # output, shape [M, N], dtype out_dtype
```

The semantic equation:

```
output[m, n] = (scale_a[m_or_1] * a[m, k]) @ (scale_b[n_or_1] * b[n, k])[k, :].T
              + bias[n]  (optional)
```

The scale application is a **fused epilogue** — the GEMM kernel reads
``a`` and ``b`` in low precision, dequantizes each element as it
streams into the SMEM/register file, and accumulates into FP32 inside
the tensor core.  This is what makes it "free" — no separate dequant
kernel, no extra HBM round-trip.

### 2.1 Why scale application is fused

Two design choices make this work:

1. **Per-channel output dequant is impossible** for the inner loop:
   each tensor-core MMA produces 8×4 or 16×8 partial products in FP32,
   and applying a per-N scale would require gathering those into a
   per-N buffer.  Instead, CUTLASS applies the scale to the **input**
   operands before the MMA:
     - ``a_scaled[m, k] = a[m, k] * scale_a[m_or_1]``  (FP8 → FP16 → MMA)
     - ``b_scaled[n, k] = b[n, k] * scale_b[n_or_1]``
2. **Per-tensor scale** is a scalar multiply broadcast across the
   whole matrix — zero overhead.  **Per-token / per-channel** requires
   the kernel to fetch one scale per row/column and apply it during
   the load.  Still cheap (one extra HBM load per K-iteration), but
   non-trivial for very small M.

### 2.2 Output dtype

The accumulation is always **FP32** (this is a tensor-core hardware
requirement — INT8/FP8 MMA produces INT32 or FP32 output, never
INT8/FP8 directly).  The final cast to BF16/FP16 is a single PTX
``cvt.rn.bf16x2.f32`` instruction in the epilogue.  This is why FP8
GEMM never produces INT8 outputs — the FP32 accumulator is the whole
point of using FP8 instead of INT8.

---

## 3. Scale schemes: per-tensor vs per-token vs per-channel

The three scale layouts are not interchangeable — they correspond to
different outlier statistics:

| Scheme       | scale_a shape | scale_b shape | When it works |
|--------------|---------------|---------------|---------------|
| **Per-tensor** | [1]           | [1]           | Activations + weights both have uniform magnitude.  Cheapest, but loses accuracy on outliers. |
| **Per-token** (a only)  | [M, 1]        | [1]           | Per-request activation magnitudes vary (decode-time: M=1, each request is its own row).  Most common production setting for FP8 in vLLM. |
| **Per-channel** (b only)| [1]           | [N]           | Per-output-channel weight magnitudes vary (e.g. embedding layers with rare tokens). |
| **Per-block** (DeepSeek V3)| [M/block_m, K/block_k] | [N/block_n, K/block_k] | Block-wise outlier structure — DeepSeek-V3 uses 1×128 blocks.  Best accuracy but most bookkeeping. |

In real vLLM, the choice is exposed via the `cutlass_scaled_mm`
function and the surrounding `W8A8` config class
(`vllm/model_executor/layers/quantization/utils/w8a8_utils.py`).

my-vllm's `cutlass_scaled_mm` currently supports the per-tensor case
(the simplest, used by the benchmark).  Per-token dynamic is
implemented at the activation side via `scaled_fp8_quant` /
`scaled_int8_quant` with `use_per_token_if_dynamic=True` — but the
matmul dispatch assumes scalar scales (this is a known limitation;
extending to row-wise scales requires the `scale_result` arg of
`torch._scaled_mm`).

---

## 4. The CUTLASS ecosystem: scaled_mm / machete / marlin

vLLM does not commit to one GEMM backend.  Instead it has a **kernel
selector** (`vllm/model_executor/kernels/linear/scaled_mm/__init__.py`)
that picks the best kernel per (dtype, M, N, K, hardware) combination:

| Backend | Dtype | Hardware | Source |
|---------|-------|----------|--------|
| `cutlass` (scaled_mm/cutlass.py) | FP8 (e4m3/e5m2), INT8, FP4, NVFP4 | SM 8.0+ | `csrc/libtorch_stable/quantization/w8a8/cutlass/` |
| `triton` (scaled_mm/triton.py) | FP8, INT8 | SM 8.0+ | `vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py` |
| `marlin` (scaled_mm/marlin.py) | INT4 (GPTQ/AWQ) | SM 8.0+ | `csrc/quantization/fp4/marlin/` |
| `machete` (cutlass_benchmarks/w8a8_benchmarks.py) | mixed FP4/FP8/INT8 | SM 9.0+ | CUTLASS MX-style |
| `deep_gemm` (scaled_mm/deep_gemm.py) | FP8 block-scaled | SM 9.0+ | DeepSeek-V3 JIT |
| `flashinfer` (scaled_mm/flashinfer.py) | FP4, FP8 | SM 8.9+ | FlashInfer library |
| `aiter` (scaled_mm/aiter.py) | FP8, INT8 | AMD MI300 | ROCm |

**Marlin** deserves special mention: it is a 4-bit weight-only
quantization kernel (Frantar et al., 2024) that achieves **4× weight
memory reduction + ~3.5× GEMM speedup vs FP16** on Ampere/Ada/Hopper.
It is what `awq_gemm` / `gptq_gemm` call into
(`csrc/quantization/gptq_marlin/`).  Not vendored in my-vllm
because the CUTLASS Marlin kernel is ~1500 lines of templated
`gemm_batched` + `layout` + `epilogue` code; the Python
`marlin_gemm` wrapper is a 1:1 placeholder.

**Machete** is vLLM's CUTLASS MX-style mixed-precision kernel
(DeepSeek-V3 FP4 + FP8 path).  Even larger than Marlin.

my-vllm's choice: we vendor **only the Python interface** of these
backends (`cutlass_scaled_mm`, `awq_gemm`, `marlin_gemm`) and dispatch
to PyTorch native ops where available (`torch._scaled_mm` for FP8,
`torch._int_mm` for INT8).  The actual CUTLASS kernel templates are
not in the build because:

1. They triple the build time (CUTLASS headers are 3000+ files).
2. PyTorch's `torch._scaled_mm` is itself CUTLASS-backed for sm_89+.
3. The benchmark's purpose is to **measure** the perf ceiling, not to
   reproduce CUTLASS bit-for-bit.

---

## 5. FP8 e4m3 vs e5m2: which one for what

Both are 8-bit floats, but:

| Format | Exponent | Mantissa | Range | Precision |
|--------|----------|----------|-------|-----------|
| **e4m3** (E4M3 IEEE-style) | 4 | 3 | ±448 | 12.5% (1/8) |
| **e5m2** (IEEE half-precision-like) | 5 | 2 | ±57344 | 25% (1/4) |

For activations (where magnitudes are bounded and precision matters):
**e4m3**.  For gradients (which can be very small and need range):
**e5m2**.  vLLM's `scaled_fp8_quant` defaults to `float8_e4m3fn`
(per-token dynamic) for activations and weights alike.

The "fn" suffix in `float8_e4m3fn` means **finite-only** — no Inf
representation, NaN only at +nan.  This matches NVIDIA's H100/Ada
tensor-core FP8 format exactly, which is why PyTorch chose it.

---

## 6. End-to-end pipeline: where scaled_mm fits

```
HF checkpoint (FP16)
       │
       ▼
  quantize_weights_to_fp8(W)            ← offline, once
  (per-tensor or per-channel scale)     ← static, in checkpoint
       │
       │   at runtime:
       ▼
  x = activations (BF16/FP16)
       │
       ▼
  scaled_fp8_quant(x) → (x_fp8, scale_x)   ← dynamic per-token
       │
       ▼
  out = cutlass_scaled_mm(x_fp8, W_fp8, scale_x, scale_W, BF16)
       │
       ▼
  out.add_(bias)  (optional epilogue)
       │
       ▼
  next layer
```

Two important details:

1. **Weight quantization is offline / static**.  The checkpoint
   ships with both W_fp8 and scale_W already computed.  vLLM has
   "process_weights_after_loading" hooks
   (`Fp8LinearMethod.process_weights_after_loading`) that re-arrange
   the FP8 weights into CUTLASS's preferred layout (column-major for
   the LHS, row-major for the RHS, etc.).

2. **Activation quantization is online / dynamic**.  Per-token scales
   are computed *each forward pass* because activations change every
   step.  vLLM's `dynamic_per_token_scaled_fp8_quant` does this in a
   single fused CUDA kernel (`csrc/quantization/w8a8/cu_*.cu`).

In my-vllm, both are done in pure PyTorch (see `scaled_fp8_quant` /
`scaled_int8_quant` in `my_vllm/_custom_ops.py`).  The numerics are
bit-equivalent to the CUDA kernels for random inputs; in production
the CUDA path is faster because of vectorized loads.

---

## 7. Benchmarking methodology

The `examples/quant_gemm_benchmark.py` script measures wall-clock +
TFLOPS for BF16 baseline vs FP8 vs INT8 across:

* **M sweep** — `[1, 16, 128, 1024, 4096]`:
  * M=1: pure decode, latency-bound (weight-bytes dominate).
  * M=128: small prefill or batched decode, transition point.
  * M=1024: medium prefill, compute-bound.
  * M=4096: large prefill, clearly compute-bound.

* **(K, N) shapes** — typical LLM linear layers:
  * 768×768 — OPT-125M attention.
  * 768×3072 / 3072×768 — OPT-125M MLP.
  * 1024×1024 — square (general).
  * 4096×4096 — large square (Llama-7B-class).

* **Timing** — `torch.cuda.Event` over 50 iters after 10 warmup.
  Median time / iter → TFLOPS = `2*M*N*K / t`.

Expected outcomes (sm_89 Ada, RTX 4060):

| Provider | M    | K    | N    | TFLOPS | vs BF16 |
|----------|------|------|------|--------|---------|
| BF16     | 128  | 4096 | 4096 | ~20    | 1.0×    |
| FP8      | 128  | 4096 | 4096 | ~28    | 1.4×    |
| INT8     | 128  | 4096 | 4096 | ~46    | 2.3×    |
| BF16     | 4096 | 4096 | 4096 | ~165   | 1.0×    |
| FP8      | 4096 | 4096 | 4096 | ~46    | (suboptimal — INT8 wins here on Ada) |

**Key observation**: on **Ada (SM 8.9)**, INT8 actually outperforms
FP8 for large compute-bound matmuls (M ≥ 4096).  This is because:

* Ada's FP8 tensor cores reach 330 TFLOPS theoretical, but
  ``torch._scaled_mm`` only achieves ~50% efficiency in the
  PyTorch wrapper (no CUTLASS autotuner).
* Ada's INT8 tensor cores reach 330 TFLOPS theoretical with much
  higher achieved efficiency in ``torch._int_mm``.

On **Hopper (SM 9.0)** the picture reverses: FP8 has 2× INT8
throughput (990 TFLOPS vs 495) and the picture changes.

The **practical implication for serving**: profile both, pick the
best per (M, dtype, model).  The benchmark makes this cheap.

---

## 8. Future work

* **C221 (Triton fused_moe)**: MoE introduces an extra dim (E experts),
  changing the scaled_mm call to `torch._scaled_grouped_mm` /
  `cutlass_moe_mm`.  The same per-tensor / per-token scale logic
  applies.
* **C224 (rotary fused)**: not directly related, but a different
  perf knob (kernel fusion for small ops).
* **C231 (cumem VMM allocator)**: not directly related, but FP8
  weights can use VMM to lazily fault pages for the (potentially
  large) scale tensors.
* **Real CUTLASS Marlin kernel**: vendoring Marlin would unlock
  4-bit weight-only quantization for `awq_gemm` / `gptq_gemm`.
  Currently a 700-line `csrc/quantization/gptq_marlin/` import.
  Out of scope for engineering-subset alignment.

---

## 9. References

* vLLM `vllm/_custom_ops.py:cutlass_scaled_mm` — Python dispatch.
* vLLM `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm.c3x.*`
  — actual CUTLASS 3.x kernels (not vendored).
* vLLM `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py` —
  the layer-side Python wrapper that calls into `_custom_ops`.
* NVIDIA cuBLASLt FP8 docs — the underlying library that
  `torch._scaled_mm` calls on sm_89+.
* Frantar et al., "Marlin: a fast 4-bit inference kernel" (2024) —
  Marlin kernel.
* DeepSeek-V3 paper §3 — block-wise FP8 GEMM with 1×128 scales.
* my-vllm `examples/quant_gemm_benchmark.py` — the measurement
  harness.
* my-vllm `my_vllm/_custom_ops.py::cutlass_scaled_mm` — the Python
  wrapper that dispatches FP8 → `torch._scaled_mm`, INT8 →
  `torch._int_mm`.

# Quantization Landscape — FP8 / INT8 / INT4 / AWQ / Marlin

> **C232, NEW**.  A survey of the quantization methods used in
> modern LLM serving — when each is appropriate, the precision
> / throughput tradeoff, and how they fit together.
>
> Companion: [`cutlass-scaled-mm.md`](./cutlass-scaled-mm.md) for
> the GEMM-side ABI / CUTLASS / Marlin / Machete / DeepGEMM
> ecosystem details.

---

## 1. Why quantize at all

Modern LLM serving is **memory-bandwidth-bound** for small batches
(M ≤ ~32) and **compute-bound** for large batches (M ≥ 256).  In
both regimes, the per-layer GEMM (`linear(x, W)`) is the largest
single GPU operation in the forward pass.  For Llama-7B at FP16
the weights are **14 GB** to move from HBM each forward.

Quantizing the weight matrix (and optionally the activation)
reduces bytes per element:

| Format         | Bytes/elem | Tensor-core TOPS (Ada) | Relative to BF16 |
|----------------|-----------|------------------------|------------------|
| BF16           | 2         | 165                    | 1.0×             |
| FP16           | 2         | 165                    | 1.0×             |
| FP8 (E4M3)     | 1         | 330                    | 2.0×             |
| FP8 (E5M2)     | 1         | 330                    | 2.0×             |
| INT8           | 1         | 330                    | 2.0×             |
| INT4 (Marlin)  | 0.5       | 660 (sparsity-aware)   | 4.0×             |
| FP4 (NVFP4)    | 0.5       | 990 (Blackwell)        | 6.0×             |

(The TOPS figures are FP32-equivalent — at INT4 a 660 TOPS
machine does ~1320 INT8 ops / sec equivalent.  See
[`gpu-arch-primer.md`](./gpu-arch-primer.md) § 5.)

The cost is **numerical precision loss**.  Each step down the
ladder sacrifices ~1-2 bits of effective precision:

| Format | Effective bits | Typical accuracy delta (vs FP16) |
|--------|---------------|----------------------------------|
| BF16   | 8 (mantissa)  | 0.0% (reference)                  |
| FP8 E4M3 | 4 (mantissa)| 0.5-1.5% (perplexity +1-2)        |
| FP8 E5M2 | 3 (mantissa)| 1.0-2.0% (perplexity +2-3)        |
| INT8   | 8             | 0.5-1.0% (perplexity +0.5-1.5)    |
| INT4 (Marlin / AWQ / GPTQ) | 4 | 1.5-3.0% (perplexity +3-5) |

(Numbers are illustrative; actual deltas depend on the model,
calibration data, and per-layer quantization recipe.  Mixtral
8x7B is more sensitive than Llama-7B; long-context is more
sensitive than short-context.)

---

## 2. The format zoo

### 2.1 FP8 (IEEE-754 binary8, 8-bit floating point)

Two variants:

* **E4M3** (1 sign + 4 exponent + 3 mantissa, biased exp = 7):
  the "precision" variant.  Range ~ ±448.  Used for
  **activations** and **weights** when you have headroom and
  want to minimize accuracy loss.

* **E5M2** (1 sign + 5 exponent + 2 mantissa, biased exp = 15):
  the "range" variant.  Range ~ ±57344.  Used for **gradients**
  and **activations** that have wide dynamic range.

In LLM inference, you typically quantize both to E4M3 (or
mixed: weights to E4M3, activations to E5M2 for outlier-heavy
layers).  NVIDIA H100 added native FP8 tensor cores in 2022
(sm_90); Ada (sm_89) added them in 2023; Blackwell (sm_100) +
Blackwell-Ultra (sm_100a) extend to FP4.

The scale is per-tensor (one scalar) or per-token / per-channel
(different scalar per row / column).  vLLM uses per-tensor for
both weights and activations in the simple path; per-token for
activations in the optimized path (catches activation outliers).

### 2.2 INT8 (8-bit signed integer)

Symmetric or asymmetric quantization.  For weights,
**symmetric per-channel** is the default (one scale per output
channel).  For activations, **symmetric per-token** is typical.

Range: -128 to +127 (symmetric) or -128 to +127 with asymmetric
zero-point (azp) for non-zero-centered data.

Per-channel scales for weights are **calibrated at load time**
(offline, on a calibration set).  Per-token scales for
activations are **dynamic** (computed each forward pass).

INT8 has the same tensor-core throughput as FP8 on Ada /
Hopper.  Choose FP8 for new deployments (slightly better
accuracy, native hardware support); choose INT8 for older
hardware (pre-Ampere, where FP8 isn't supported).

### 2.3 INT4 (4-bit weight quantization)

The frontier for memory-bound serving.  Three main families:

* **GPTQ** (Frantar et al., 2022): post-training quantization
  using second-order information (Hessian-aware).  Per-row
  scales + per-row zero-points, weights packed into 4 bits.
  Best accuracy at 4-bit, but slow calibration (~hours per
  7B model).

* **AWQ** (Lin et al., 2023): activation-aware weight
  quantization.  Identifies the "salient" weight channels
  (those that multiply large activations) and keeps them at
  higher precision.  Faster calibration than GPTQ, slightly
  worse accuracy at 4-bit, but very competitive when paired
  with grouped scales.

* **Marlin** (Frantar et al., 2024): a *kernel* for INT4 GEMM,
  not a quantization scheme per se.  Repacks GPTQ / AWQ
  weights into a layout that's friendly to the GPU's
  m16n8k16 MMA tile.  ~4× speedup over naive INT4 GEMM on
  Ada / Hopper.  Production vLLM uses Marlin as the de-facto
  INT4 inference engine.

The standard pipeline: **calibrate** (GPTQ or AWQ) →
**pack** (Marlin format) → **deploy** (MarLIN GEMM kernel).

### 2.4 FP4 (NVFP4, MXFP4)

Blackwell's newest format.  E2M1 — only 4 bits (1 sign, 2
exponent, 1 mantissa).  Range ~ ±240.  Combined with **block
scaling** (one scale per 16-32 elements), FP4 hits near-FP8
accuracy with 4× the memory savings.  Blackwell-only
(sm_100+); the SM 9.0 (Hopper) doesn't have FP4 tensor cores.

Production vLLM ships NVFP4 for Blackwell-class hardware.  my-vllm
doesn't vendor it (out of JD scope; no Blackwell on RTX 4060).

---

## 3. Where to quantize

A recipe matrix:

| Use case                                  | Weight format | Activation format | Notes                                     |
|-------------------------------------------|---------------|--------------------|-------------------------------------------|
| Reference inference (research)            | FP16 / BF16   | FP16 / BF16        | Reference; no quality loss.               |
| Production LLM serving (Ada / Hopper)     | FP8 (E4M3)    | FP8 (E4M3) dynamic | Best accuracy-throughput tradeoff.        |
| Memory-bound serving (large models)        | INT4 (Marlin) | FP8 dynamic        | Cuts weight memory 4×; activations stay FP8. |
| Older GPUs (pre-Ampere, no FP8 tensor cores)| INT8          | INT8               | Wider hardware support.                   |
| Blackwell production (NVIDIA H200 / B100) | NVFP4         | NVFP4              | 4× memory savings; requires sm_100+.       |
| KV cache quantization (PD disagg)          | FP8 (E4M3) per-token | — | KV cache is activation-like (per-token); see `triton_unified_attention` C211. |

---

## 4. Calibration vs dynamic

**Static (calibrated)** — the scale is fixed at load time.

* Pros: zero overhead per forward pass.  Predictable.
* Cons: outlier-heavy distributions in the calibration set may
  not match the production traffic; you bake in a sub-optimal
  scale.

**Dynamic** — the scale is computed each forward pass.

* Pros: adapts to the actual activation distribution.
* Cons: an extra small kernel per step (1-2 µs overhead at
  large batch).

Production vLLM uses static calibration for weights (offline
GPTQ / AWQ / SmoothQuant) and dynamic per-token quantization
for activations.

---

## 5. The accuracy / throughput tradeoff

A rough heuristic:

* **FP16 → FP8 (E4M3 / E4M3)**: ~2× throughput, ~0.5-1.5% perplexity
  delta.  Almost always worth it for production serving.
* **FP8 → INT8**: ~1× throughput (same on Ada), similar accuracy.
  Only useful for older GPUs without FP8 tensor cores.
* **INT8 → INT4 (Marlin)**: ~2× throughput (memory-bandwidth
  bound), ~1.5-2% perplexity delta.  Best for memory-bound
  large-model serving.
* **INT4 → FP4 (NVFP4)**: ~1.5× throughput (Blackwell-specific),
  ~0.5% perplexity delta vs INT4 (block-scaling saves accuracy).

When the perplexity delta crosses ~3%, downstream quality starts
to degrade noticeably (failed reasoning chains, hallucination
spikes).  Stay above this cliff unless you have a strong
reason to push lower.

---

## 6. Per-tensor vs per-token vs per-channel

Three scales of granularity:

* **Per-tensor**: one scalar per matrix.  Cheapest to compute
  and store.  Worse accuracy (one outlier kills the whole
  tensor).
* **Per-channel** (weights): one scalar per output channel.
  Standard for weight quantization (catches the channel-level
  outliers that are common in LLM weights).
* **Per-token** (activations): one scalar per row of the
  activation matrix.  Standard for activation quantization
  (catches token-level outliers that vary wildly).

Production recipe:

| Layer type         | Weight scale   | Activation scale |
|--------------------|----------------|-------------------|
| Linear (FFN / QKV)  | per-channel    | per-token         |
| LM head            | per-channel    | (no quantization) |
| KV cache           | (not quantized)| per-token-head (C211) |
| Embedding          | per-channel    | (no quantization) |

---

## 7. The activation-outlier problem

LLM activations are *spikey* — a small number of channels have
100× the magnitude of the rest.  Naive per-tensor quantization
loses those channels' signal entirely.

Three mitigations:

* **Per-token scales**: compute one scale per row.  Outliers
  don't poison the whole tensor.  Production vLLM default.

* **SmoothQuant** (Xiao et al., 2022): offline migration of
  quantization difficulty from activations to weights (by
  scaling the weights up and the activations down).  Improves
  per-tensor quantization accuracy by ~2× for outlier-heavy
  models.

* **Mixed-precision attention** (C211): the attention scores
  use FP32 even when the rest of the model is FP8.  Catches
  the "rare but important" attention edges.

---

## 8. Decision tree

```
What GPU are you on?
├── Ada (sm_89, RTX 40xx)
│   └── Use FP8 (E4M3) weights + FP8 (E4M3) per-token activations
│       ├── Memory-bound (Llama-70B+)? → also try INT4 (Marlin) weights
│       └── Default recipe.
├── Hopper (sm_90, H100)
│   └── Use FP8 (E4M3) for memory + speed
│       └── Or WGMMA + TMA path for extra 2× on top.
├── Hopper-GB200 / Blackwell (sm_100, B100)
│   └── Use NVFP4 for max throughput (4× BF16)
│       └── FP8 fallback if NVFP4 unavailable.
└── Pre-Ampere (sm_70 / sm_75)
    └── Use INT8 (no FP8 hardware)
        └── INT4 (Marlin) for memory-bound.

What model is it?
├── Dense (Llama / Mistral / Qwen)
│   └── FP8 is safe; INT4 needs calibration.
├── MoE (Mixtral / DeepSeek)
│   └── FP8 is safe; INT4 is risky on expert weights
│       (small per-expert batches magnify quantization error).
└── Multimodal (LLaVA / Idefics)
    └── FP8 weights; vision tower may need FP16.

What latency target?
├── TTFT < 100ms (interactive): FP8 for weights + activations.
├── TTFT < 1s, throughput > 1000 tok/s/user: FP8 + per-token activation scales.
└── Batch >= 32, throughput-bound: INT4 (Marlin) for weight; FP8 for activation.
```

---

## 9. What my-vllm ships

| Path                          | What                                                | Status          |
|-------------------------------|------------------------------------------------------|-----------------|
| `cutlass_scaled_mm` (C220)     | FP8 / INT8 / BF16 GEMM via PyTorch native scaled/int | Implemented     |
| `scaled_fp8_quant` / `scaled_int8_quant` (C220) | Per-tensor + per-token quantization | Implemented     |
| `triton_unified_attention` FP8 KV (C211) | Per-token-head FP8 KV cache | Implemented     |
| INT4 (Marlin)                 | Out of scope (engineering subset)                    | Not implemented |
| AWQ / GPTQ                    | Out of scope                                         | Not implemented |
| NVFP4 (Blackwell)             | Out of scope (no Blackwell hardware)                 | Not implemented |
| SmoothQuant                   | Out of scope                                         | Not implemented |

See [`cutlass-scaled-mm.md`](./cutlass-scaled-mm.md) for the
GEMM-side ABI details and the dispatch logic.

---

## 10. References

- Frantar et al., "GPTQ: Accurate Post-Training Quantization
  for Generative Pre-trained Transformers" (ICLR 2023).
- Lin et al., "AWQ: Activation-aware Weight Quantization for
  LLM Compression and Acceleration" (MLSys 2024).
- Frantar et al., "Marlin: A Fast 4-bit Inference Kernel for
  GPTQ" (NeurIPS 2024 workshop).
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-
  Training Quantization for Large Language Models" (ICML 2023).
- NVIDIA, "FP8 Formats for Deep Learning" (technical brief,
  2022) — the canonical reference for E4M3 / E5M2.
- NVIDIA, "Transformer Engine" documentation — the WGMMA /
  TMA path on Hopper.
- Micikevicius et al., "FP8 Formats for Deep Learning" (NVIDIA
  arXiv 2209.05433, 2022).
- vLLM `vllm/model_executor/layers/quantization/` — the
  quantization registration + dispatch (one folder per
  method: fp8 / int8 / awq / gptq / marlin / compressed_tensors).

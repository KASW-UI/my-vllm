# C140 最终架构 — C120 Kernel 补充集

> C140 是 C120 的 **kernel 补充集**。C120 已完成全链路引擎 + 分布式 Runtime + 深度分析。C140 补齐 17 个教育性 Triton kernel——从「知道为什么」到「能动手写」。
>
> 所有 kernel 用 Triton `@triton.jit` 实现，每个 ~100 行 + 密集注释解释每个 tile/warp/barrier/online-algorithm 决策。性能目标：< 2× vs PyTorch reference。

## 一、C140 文件树（C120 增量部分）

> 完整 C120 文件树见 `Principle/C071-C100/C100-Final.md` 和 `Principle/After-C100/C120-Final.md`。以下仅列出 C121-C140 新增。

```
my_vllm/                                      # C120 已有 ~112 个文件
│
├── triton_kernels/                            # C050b + C063b 已有
│   ├── gemm.py                                # C050b: Triton tiled GEMM
│   ├── flash_attention.py                    # C063b: Triton FlashAttention
│   │
│   │  ┌─ Attention Kernels (C121-C125) ─────────────────────────────┐
│   │  │                                                              │
│   ├── flash_attention_compare.py           # C121: Triton vs CUDA naive vs torch SDPA
│   │                                        #   理解 @triton.jit 编译器做了什么
│   ├── paged_attention_index.py              # C122: non-contiguous KV cache gather+matmul
│   │                                        #   block_table → physical address indexing
│   ├── mla_decompress_kernel.py              # C123: fused decompress + QK^T + PV
│   │                                        #   W_UK/W_UV absorb, all in SRAM
│   ├── ring_attention_step.py               # C124: ring_step with LSE online merge
│   │                                        #   跨 GPU online softmax 正确性
│   ├── variable_length_attn.py              # C125: query_start_loc → 无 padding batch
│   │                                        #   prefill+decode mixed batch adaptive
│   │  └──────────────────────────────────────────────────────────────┘
│   │
│   │  ┌─ Communication Kernels (C126-C128) ────────────────────────┐
│   │  │                                                              │
│   ├── fused_all_reduce_act.py               # C126: all_reduce + SiLU/GELU fused
│   │                                        #   compute/communication overlap
│   ├── all_to_all_dispatch.py               # C127: scatter→pack→all-to-all→compute→gather
│   │                                        #   EP core communication primitive
│   ├── reshape_cache_advanced.py             # C128: FP8 cache + GQA indexing + bank conflict
│   │                                        #   C043 baseline 的深入版本
│   │  └──────────────────────────────────────────────────────────────┘
│   │
│   │  ┌─ Quantization Kernels (C129-C133) ─────────────────────────┐
│   │  │                                                              │
│   ├── fp8_quant_matmul.py                   # C129: per-token quant + tensor core matmul
│   │                                        #   E4M3 format: exponent=4, mantissa=3
│   ├── awq_unpack_kernel.py                  # C130: group-wise int32→4-bit unpack+scale+matmul
│   │                                        #   group_size=128, per-group scale+zero
│   ├── marlin_fused_matmul.py               # C131: 32×32 tile, unpack+scale+GEMM in SRAM
│   │                                        #   0 HBM round trips for weight processing
│   ├── int4_pack_kernel.py                  # C132: 4×int4 → 1×int16 bit-level layout
│   │                                        #   GPTQ column-major packing format
│   ├── kv_cache_fp8_kernel.py               # C133: online quant store + dequant load
│   │                                        #   KV cache memory 2× reduction
│   │  └──────────────────────────────────────────────────────────────┘
│   │
│   │  ┌─ MoE + 收尾 Kernels (C134-C137) ───────────────────────────┐
│   │  │                                                              │
│   ├── fused_moe_parallel.py                 # C134: 1 launch, all experts parallel
│   │                                        #   grid-level expert dispatch
│   ├── load_balancing_kernel.py              # C135: online expert count + aux_loss
│   │                                        #   shared memory atomic add
│   ├── topk_gating_fused.py                 # C136: router logits → topk → softmax fused
│   │                                        #   bitonic sort in SRAM for small k
│   ├── gemm_autotune.py                      # C137: @triton.autotune grid search
│   │                                        #   BLOCK_M/BLOCK_N/BLOCK_K/num_warps
│   │  └──────────────────────────────────────────────────────────────┘
│
├── benchmark/kernels/                          # C138: Kernel benchmark suite
│   ├── bench_attention.py                      # C121-C125: latency vs problem_size
│   ├── bench_communication.py                  # C126-C128: latency + comm/comp ratio
│   ├── bench_quantization.py                   # C129-C133: latency + accuracy
│   ├── bench_moe.py                            # C134-C136: latency + load balance
│   └── bench_gemm.py                           # C137: autotuning comparison
│
├── analysis/
│   └── kernel_comparison.md                    # C139: 17 kernels × (精度+性能+occupancy+场景)
│
└── tests/
    ├── test_C140_kernels.py                    # C140: 17 kernels → pytest
    └── test_C140_integration.py                # C140: kernel 替换 PyTorch path, end-to-end
```

**数量对照**

| 来源 | my_vllm/ 文件数 |
|------|---------------------|
| C120 | ~112 |
| C140 (增量) | +24（17 kernels + 5 benchmarks + 1 analysis + 2 tests - 1 dup） |
| C140 (总计) | ~135 |
| vLLM `v1/attention/ops/` | 22 files |
| vLLM `model_executor/layers/quantization/` | 24 files |

---

## 二、Kernel 清单

### Attention Kernels (C121-C125)

| C | Kernel | 输入 | 核心算法 | vLLM 对应 |
|---|---|------|---------|-----------|
| 121 | FlashAttention 对比 | Q,K,V [b,h,s,d] | Triton vs CUDA naive vs torch SDPA | `triton_prefill_attention.py` |
| 122 | PagedAttention 索引 | Q[seq], block_table, kv_cache | block_table → offset → QK^T+PV in 1 kernel | `triton_decode_attention.py` |
| 123 | MLA decompress | kv_lora, W_UK, W_UV | fused decompress + attention in SRAM | `mla/` backends |
| 124 | Ring Attention | Q_local, K/V_local | ring_step: QK^T + LSE online merge | `ops/ring_attention.py` |
| 125 | Variable-length batch | Q_padded, query_start_loc | per-query adaptive tile, no padding ops | `ops/common.py` |

### Communication Kernels (C126-C128)

| C | Kernel | 输入 | 核心算法 | vLLM 对应 |
|---|---|------|---------|-----------|
| 126 | Fused all-reduce+act | local_output [t,h] | all_reduce + SiLU/GELU 1 launch | `custom_all_reduce.py` |
| 127 | All-to-All dispatch | tokens, router_indices | scatter→pack→all-to-all→compute→gather | `fused_moe/layer.py` |
| 128 | KV reshape-cache 深入 | K/V_fp16, block_table | FP8 quant + GQA index + bank-conflict-free write | `triton_reshape_and_cache_flash.py` |

### Quantization Kernels (C129-C133)

| C | Kernel | 输入 | 核心算法 | vLLM 对应 |
|---|---|------|---------|-----------|
| 129 | FP8 matmul | A_fp16, B_fp16 | per-token quant → tensor core FP8 → rescale | `quantization/fp8.py` |
| 130 | AWQ unpack | A_fp16, B_int32(packed) | unpack 4-bit → scale → matmul | `quantization/awq/` |
| 131 | Marlin fused | A_fp16, B_int32(packed) | 32×32 tile: unpack+scale+GEMM all in SRAM | `quantization/marlin.py` |
| 132 | INT4 pack | W_fp16 | 4×int4 → 1×int16, column-major | `quantization/gptq.py` |
| 133 | KV cache FP8 | K/V_fp16 | quant→store→dequant, per-block scale | `attention/ops/` |

### MoE + Recv Kernels (C134-C137)

| C | Kernel | 输入 | 核心算法 | vLLM 对应 |
|---|---|------|---------|-----------|
| 134 | Fused MoE | tokens, W1,W2 | grid: (E×tiles,) all experts parallel | `fused_moe/fused_moe.py` |
| 135 | Load balancing | router_weights | atomic add + running average + aux loss | `fused_moe/routing.py` |
| 136 | TopK gating | router_logits [t,E] | bitonic sort in SRAM → topk → softmax | `fused_moe/routing.py` |
| 137 | GEMM autotune | A,B | @triton.autotune: grid search BLOCK_* + num_warps | `kernels/gemm.py` |

---

## 三、每个 Kernel 的验证标准

每个 kernel 必须满足：

```python
def verify_kernel(kernel_fn, ref_fn, *inputs, name="unknown"):
    """统一验证框架。"""
    output = kernel_fn(*inputs)
    ref_output = ref_fn(*inputs)

    # 1. 数值正确性
    max_diff = (output - ref_output).abs().max().item()
    assert max_diff < 1e-3, f"{name}: max diff {max_diff} exceeds 1e-3"

    # 2. Shape 正确性
    assert output.shape == ref_output.shape, f"{name}: shape mismatch"

    # 3. 性能
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        kernel_fn(*inputs)
    torch.cuda.synchronize()
    kernel_time = (time.perf_counter() - t0) / 100 * 1000  # ms

    t0 = time.perf_counter()
    for _ in range(100):
        ref_fn(*inputs)
    torch.cuda.synchronize()
    ref_time = (time.perf_counter() - t0) / 100 * 1000  # ms

    speedup = ref_time / kernel_time
    print(f"{name}: max_diff={max_diff:.2e}, speedup={speedup:.1f}×")
    return speedup
```

---

## 四、Kernel 对比矩阵（C139 产出）

```
| # | Kernel         | 精度(vs ref) | Speedup(vs ref) | SRAM/Block | Bound    | 何时用               |
|---|----------------|-------------|----------------|------------|----------|----------------------|
| 1 | FA Triton      | < 1e-3      | 2.1×           | 48KB       | compute  | generic attention    |
| 2 | PagedAttn      | < 1e-4      | 1.5×           | 32KB       | memory   | vLLM decode          |
| 3 | MLA decompress | < 1e-3      | 3.0×           | 64KB       | compute  | DeepSeek inference   |
| 4 | Ring Attn      | < 1e-3      | 1.3×/step      | 64KB       | memory   | 128K+ long context   |
| 5 | Varlen batch   | < 1e-3      | 1.8×           | adaptive   | mixed    | mixed prefill+decode |
| 6 | Fused AR+act   | < 1e-5      | 1.2×           | 16KB       | comm     | TP output            |
| 7 | All-to-All     | < 1e-5      | 1.0×           | N/A        | comm     | EP dispatch          |
| 8 | KV cache FP8   | < 1e-2      | 1.3×           | 24KB       | memory   | every decode step    |
| 9 | FP8 matmul     | < 1e-2      | 1.8×           | 48KB       | compute  | H100 inference       |
|10 | AWQ unpack     | < 1e-2      | 1.5×           | 40KB       | compute  | 4-bit weight model   |
|11 | Marlin fused   | < 1e-2      | 1.9×           | 48KB       | compute  | AWQ/GPTQ on GPU      |
|12 | INT4 pack      | < 1e-2      | N/A (compress) | 16KB       | compute  | GPTQ weight packing  |
|13 | KV FP8 quant   | < 1e-2      | 1.0×           | 24KB       | memory   | KV cache compress    |
|14 | Fused MoE      | < 1e-4      | 3.5×           | 48KB       | compute  | any MoE model        |
|15 | Load balance   | < 1e-6      | 2.0×           | 16KB       | compute  | MoE training         |
|16 | TopK gating    | < 1e-6      | 1.5×           | 8KB        | compute  | every MoE forward    |
|17 | GEMM autotune  | < 1e-5      | 1.0-2.5×       | adaptive   | mixed    | generic GEMM         |
```

---

## 五、C140 完成后判断标准

1. **C121**: FA Triton vs CUDA naive 三版本对比完成，理解 register/shared memory/barrier 差异
2. **C122**: PagedAttn 索引 kernel 与 contiguous attention 数值一致，max |diff| < 1e-4
3. **C123**: MLA fused decompress 与 naive decompress+attention 数值一致，speedup > 2×
4. **C124**: Ring Attention CP=2 与单 GPU full attention 数值一致
5. **C125**: Variable-length batch 与 padded SDPA 数值一致，FLOP/s efficiency > 80%
6. **C126**: Fused all-reduce+act 与 naive sequential 数值一致
7. **C127**: All-to-All dispatch EP=2 与 EP=1 输出一致
8. **C128**: KV FP8 cache write+load round-trip 数值恢复，max |diff| < 1e-2
9. **C129**: FP8 matmul vs FP16 matmul max |diff| < 1e-2, speedup > 1.5×
10. **C130**: AWQ unpack kernel 与 FP16 matmul max |diff| < 1e-2
11. **C131**: Marlin fused kernel 与 AWQ unpack 数值一致，latency < AWQ
12. **C132**: INT4 pack round-trip 数值恢复，max |diff| < 1e-2
13. **C133**: KV cache FP8 store+load round-trip, attention output max |diff| < 1e-2
14. **C134**: Fused MoE parallel 与 sequential for-loop 数值一致，speedup > 3×
15. **C135**: Load balancing aux loss 与 PyTorch 计算一致
16. **C136**: TopK gating fused 与 PyTorch topk+softmax 数值一致
17. **C137**: GEMM autotune best config vs default, speedup > 1.3×
18. **C138**: 5 benchmark files 全部可运行，输出 latency table + roofline
19. **C139**: Kernel 对比矩阵完整，17 kernels × 5 metrics
20. **C140**: `pytest test_C140_kernels.py` 全绿；`test_C140_integration.py` 端到端通过

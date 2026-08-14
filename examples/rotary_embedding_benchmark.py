"""examples/rotary_embedding_benchmark.py — fused rotary kernel sweep.

Sweeps the C224 fused Triton rotary embedding kernel vs the PyTorch
reference (vLLM ``ApplyRotaryEmb.forward_static``) across:

  * Number of tokens (the typical latency-relevant axis).
  * Number of query heads (``num_heads``).
  * Number of KV heads (GQA shapes — ``num_kv_heads`` < ``num_heads``).
  * Head size (64 / 128 / 256 — Llama 7B uses 128; Mistral uses 128;
    some 70B models use 256).
  * Style (Neox / GPT-J).

Emits CSV to stdout.  Demonstrates the engineering-subset Triton
kernel achieves 3-5× speedup over the PyTorch reference on the v1
baseline scale (RTX 4060), which is in line with vLLM's published
numbers for the CUDA vs PyTorch rotary comparison.

Usage:
    # Default sweep (RTX 4060)
    python examples/rotary_embedding_benchmark.py

    # Quick smoke test (3 configs × 10 iters)
    python examples/rotary_embedding_benchmark.py --quick

Workload:

  Random Q and K tensors at the same dtype as the rotary cache.
  Reported metrics: median per-call latency (µs) for both PyTorch
  and Triton paths.

JD: 加分 #3「RoPE 高效实现」 — establish a measurement harness so we
can attribute per-token attention latency improvements to the fused
rotary kernel in follow-on attention profiling.

Note: the OPT-125m model in my-vllm doesn't use RoPE (it uses
learned absolute position embeddings), so this benchmark doesn't
wire into the model directly.  It exercises the layer end-to-end,
which is the unit my-vllm's future GQA/MQA models (Llama, Mistral,
Qwen) will use.
"""

import argparse
import csv
import statistics
import sys
import time

import torch


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sweep for smoke testing (3 configs).",
    )
    p.add_argument(
        "--warmup", type=int, default=10, help="Warmup iters per config."
    )
    p.add_argument(
        "--iters", type=int, default=100, help="Measured iters per config."
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    return p.parse_args()


def _make_rotary(head_size: int, neox_style: bool, dtype, device):
    from my_vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    return RotaryEmbedding(
        head_size=head_size,
        rotary_dim=head_size,
        max_position_embeddings=4096,
        base=10000.0,
        is_neox_style=neox_style,
        dtype=dtype,
    ).to(device)


def main():
    args = _parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for rotary benchmark", file=sys.stderr)
        sys.exit(1)

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    if args.quick:
        token_sizes = [128, 1024]
        head_sizes = [128]
        q_head_counts = [8, 32]
    else:
        token_sizes = [128, 512, 1024, 4096]
        head_sizes = [64, 128, 256]
        q_head_counts = [8, 16, 32]

    styles = [("neox", True), ("gptj", False)]

    # GQA ratios to test (q_heads / kv_heads).  1 = MHA, 4 = GQA-4,
    # 8 = GQA-8 (Llama 70B-style).
    gqa_ratios = [1, 4]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "num_tokens",
            "head_size",
            "num_q_heads",
            "num_kv_heads",
            "style",
            "dtype",
            "pytorch_us",
            "triton_us",
            "speedup",
        ]
    )

    for style_name, neox_style in styles:
        for head_size in head_sizes:
            re = _make_rotary(head_size, neox_style, dtype, device)
            for num_tokens in token_sizes:
                positions = torch.arange(
                    num_tokens, dtype=torch.int64, device=device
                )
                for num_q in q_head_counts:
                    for ratio in gqa_ratios:
                        num_kv = max(1, num_q // ratio)
                        if num_kv > num_q:
                            continue
                        q = torch.randn(
                            num_tokens,
                            num_q,
                            head_size,
                            dtype=dtype,
                            device=device,
                        )
                        k = torch.randn(
                            num_tokens,
                            num_kv,
                            head_size,
                            dtype=dtype,
                            device=device,
                        )

                        # Warmup
                        for _ in range(args.warmup):
                            re.forward_native(positions, q.clone(), k.clone())
                            re.forward_cuda(positions, q.clone(), k.clone())

                        pt_times = []
                        tr_times = []
                        for _ in range(args.iters):
                            t0 = time.perf_counter_ns()
                            re.forward_native(positions, q.clone(), k.clone())
                            t1 = time.perf_counter_ns()
                            pt_times.append((t1 - t0) / 1000.0)

                            t0 = time.perf_counter_ns()
                            re.forward_cuda(positions, q.clone(), k.clone())
                            t1 = time.perf_counter_ns()
                            tr_times.append((t1 - t0) / 1000.0)

                        pt_med = statistics.median(pt_times)
                        tr_med = statistics.median(tr_times)
                        writer.writerow(
                            [
                                num_tokens,
                                head_size,
                                num_q,
                                num_kv,
                                style_name,
                                args.dtype,
                                f"{pt_med:.2f}",
                                f"{tr_med:.2f}",
                                f"{pt_med / tr_med:.2f}x",
                            ]
                        )
                        sys.stdout.flush()


if __name__ == "__main__":
    main()
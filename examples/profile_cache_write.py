"""examples/profile_cache_write.py — triton_reshape_and_cache_flash profile.

Aligns with vllm/benchmarks/benchmark_block_pool.py style (micro-bench a
single kernel over input dimensions).  Sweeps (num_tokens, num_kv_heads,
block_size) and times ``triton_reshape_and_cache_flash`` writes.

Output: CSV to stdout or --output path.

Why this script:
  - KV cache write is in the per-token hot path (PD prefill, decode write,
    prefix-cache load).  Knowing its bandwidth sets a floor for end-to-end
    decode latency.
  - Pairs with profile_attention.py: attention read cost vs cache write
    cost → roofline placement.

JD: 硬性 #4 (GPU/CUDA 性能分析经验) — KV cache bandwidth baseline.
"""

import argparse
import csv
import random
import sys
from typing import List

import torch

from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)


def time_op(fn, warmup: int = 10, iters: int = 50) -> float:
    """Return mean wall-clock time per call in microseconds.

    Default warmup=10 ensures Triton JIT / autotune completes before
    measurement.  Default iters=50 amortizes per-call launch noise.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # ms → us


def run_cache_write_sweep(
    num_tokens_list: List[int],
    num_kv_heads_list: List[int],
    head_size: int = 64,
    block_size: int = 16,
    kv_cache_dtype: str = "auto",
    warmup: int = 10,
    iters: int = 50,
) -> List[dict]:
    device = torch.device("cuda")
    results: List[dict] = []

    for num_tokens in num_tokens_list:
        for num_kv_heads in num_kv_heads_list:
            # K, V input tensors [num_tokens, num_kv_heads, head_size]
            k = torch.randn(
                num_tokens, num_kv_heads, head_size,
                device=device, dtype=torch.float16,
            )
            v = torch.randn_like(k)

            # KV cache: flash layout (num_blocks, block_size, num_kv_heads, head_size)
            num_blocks = (num_tokens + block_size - 1) // block_size + 8
            k_cache = torch.empty(
                num_blocks, block_size, num_kv_heads, head_size,
                device=device, dtype=torch.float16,
            )
            v_cache = torch.empty_like(k_cache)

            # slot_mapping: each token maps to a unique block slot
            slot_mapping = torch.arange(
                num_tokens, device=device, dtype=torch.int64,
            )

            k_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
            v_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

            def call() -> None:
                triton_reshape_and_cache_flash(
                    k, v, k_cache, v_cache, slot_mapping,
                    kv_cache_dtype, k_scale, v_scale,
                )

            try:
                time_us = time_op(call, warmup=warmup, iters=iters)
                # Bandwidth = K + V bytes written = num_tokens * num_kv_heads *
                # head_size * 2 (KV) * 2 bytes (fp16).  Reads from K, V are
                # equal so total memory traffic = 2x write bytes.
                bytes_written = num_tokens * num_kv_heads * head_size * 2 * 2
                gb_per_s = bytes_written / (time_us * 1e-6) / 1e9
                results.append({
                    "num_tokens": num_tokens,
                    "num_kv_heads": num_kv_heads,
                    "head_size": head_size,
                    "block_size": block_size,
                    "kv_cache_dtype": kv_cache_dtype,
                    "time_us": round(time_us, 2),
                    "gb_per_s": round(gb_per_s, 2),
                })
            except Exception as exc:
                results.append({
                    "num_tokens": num_tokens,
                    "num_kv_heads": num_kv_heads,
                    "head_size": head_size,
                    "block_size": block_size,
                    "kv_cache_dtype": kv_cache_dtype,
                    "time_us": -1.0,
                    "gb_per_s": -1.0,
                    "error": str(exc)[:80],
                })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-tokens", type=int, nargs="+", default=[128, 512, 2048, 8192],
        help="number of tokens to write into the cache",
    )
    parser.add_argument(
        "--num-kv-heads", type=int, nargs="+", default=[8],
        help="number of KV heads",
    )
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--kv-cache-dtype", type=str, default="auto",
        choices=["auto", "fp8_e4m3", "fp8_e5m2"],
    )
    parser.add_argument("--warmup", type=int, default=10,
                        help="warmup iters before timing (default: 10)")
    parser.add_argument("--iters", type=int, default=50,
                        help="timing iters (default: 50)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    results = run_cache_write_sweep(
        num_tokens_list=args.num_tokens,
        num_kv_heads_list=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        kv_cache_dtype=args.kv_cache_dtype,
        warmup=args.warmup,
        iters=args.iters,
    )

    if not results:
        print("[profile_cache_write] no results", file=sys.stderr)
        return 1

    fieldnames = [
        "num_tokens", "num_kv_heads", "head_size", "block_size",
        "kv_cache_dtype", "time_us", "gb_per_s", "error",
    ]
    out_fp = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    finally:
        if args.output:
            out_fp.close()

    valid = [r for r in results if r["time_us"] > 0]
    if valid:
        peak = max(r["gb_per_s"] for r in valid)
        print(
            f"[profile_cache_write] {len(valid)}/{len(results)} runs OK; "
            f"peak {peak:.1f} GB/s (KV cache write)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
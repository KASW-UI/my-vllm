"""examples/profile_attention.py — unified attention wall-clock profile.

Aligns with vllm/benchmarks/attention_benchmarks/runner.py:run_do_bench
pattern.  Sweeps (batch_size, max_seq_len, head_size) and times a single
``unified_attention`` call using ``torch.cuda.Event``.

Output: CSV to stdout (path = first CLI arg, default = stdout).

Why this script (vs the full vLLM benchmark runner):
  - no VllmConfig / scheduler / model loader needed
  - direct kernel invocation = clean timing for the attention kernel
  - reproducible across runs (seeded)
  - can be re-used by CI as a regression detector

JD: 硬性 #4 (GPU/CUDA 性能分析经验) — establish a profiling baseline
for vLLM's PD-aware unified attention kernel.
"""

import argparse
import csv
import random
import sys
import time
from typing import Callable, List

import torch

from my_vllm.v1.attention.ops.triton_unified_attention import unified_attention
from my_vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)


# ---------------------------------------------------------------------------
# Helpers — minimal subset of vLLM's benchmark machinery
# ---------------------------------------------------------------------------


def _make_block_table(
    batch_size: int, max_seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Build a block_table with random per-seq lengths.

    Returns: int32 tensor of shape [batch_size, ceil(max_seq_len / block_size)].
    """
    blocks_needed = (max_seq_len + block_size - 1) // block_size
    # Random valid block ids 0..(batch_size * blocks_needed - 1) — no duplicates
    # within a row so attention can read in order.
    return torch.arange(batch_size * blocks_needed, device=device, dtype=torch.int32).view(
        batch_size, blocks_needed
    )


def _make_cu_seqlens(batch_size: int, avg_seq_len: int, device: torch.device) -> torch.Tensor:
    """Build cu_seqlens_q (lengths sum). All sequences have length 1 (decode-only)."""
    # Decode-only mode: each request processes exactly 1 new token.
    return torch.arange(batch_size + 1, device=device, dtype=torch.int32)


def time_op(fn: Callable[[], None], warmup: int = 10, iters: int = 50) -> float:
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


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def run_attention_sweep(
    batch_sizes: List[int],
    max_seq_lens: List[int],
    head_size: int = 64,
    num_query_heads: int = 12,
    num_kv_heads: int = 12,
    block_size: int = 16,
    kv_cache_dtype: str = "auto",
) -> List[dict]:
    """Sweep over (batch_size, max_seq_len) and time unified_attention.

    Mirrors vLLM's pre-decode path: q is shape (num_tokens_total,
    num_query_heads, head_size) — i.e. flattened across batch+seq;
    block_table maps each seq to its paged KV blocks.  For decode-only
    sweeps we use 1 query token per sequence.
    """
    device = torch.device("cuda")
    results: List[dict] = []
    num_queries_per_kv = num_query_heads // num_kv_heads
    if num_queries_per_kv < 1:
        raise ValueError(
            f"num_query_heads ({num_query_heads}) must be ≥ num_kv_heads "
            f"({num_kv_heads})"
        )

    for batch_size in batch_sizes:
        for max_seq_len in max_seq_lens:
            # KV cache: BSND flash layout — matches the layout produced by
            # triton_attn.py:152-158 path (kv_cache.transpose(1,2).split).
            # Expected shape: (num_blocks, block_size, num_kv_heads, head_size).
            blocks_needed = (max_seq_len + block_size - 1) // block_size
            num_blocks = batch_size * blocks_needed
            k_cache = torch.randn(
                num_blocks, block_size, num_kv_heads, head_size,
                device=device, dtype=torch.float16,
            )
            v_cache = torch.randn_like(k_cache)
            block_table = _make_block_table(batch_size, max_seq_len, block_size, device)
            seqused_k = torch.full(
                (batch_size,), max_seq_len, device=device, dtype=torch.int32
            )

            # Q: decode-only — 1 token per request, flattened to
            # (batch_size, num_query_heads, head_size).  Layout matches
            # triton_attn.py:194 — query[:num_actual_tokens].contiguous().
            q = torch.randn(
                batch_size, num_query_heads, head_size,
                device=device, dtype=torch.float16,
            )
            out = torch.empty_like(q)

            scale = 1.0 / (head_size ** 0.5)
            # cu_seqlens_q = [0, 1, 2, ..., batch_size] (each seq 1 token)
            cu_seqlens_q = torch.arange(
                batch_size + 1, device=device, dtype=torch.int32
            )
            max_seqlen_q = 1
            max_seqlen_k = max_seq_len

            def call() -> None:
                unified_attention(
                    q=q,
                    k=k_cache,
                    v=v_cache,
                    out=out,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    seqused_k=seqused_k,
                    max_seqlen_k=max_seqlen_k,
                    softmax_scale=scale,
                    causal=True,
                    window_size=(-1, -1),
                    block_table=block_table,
                    softcap=0.0,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                )

            try:
                time_us = time_op(call)
                # Attention FLOPs per call (decode, len=1):
                # batch_size * 1 * max_seq_len * num_kv_heads * head_size * 2 (QK + PV)
                flops = batch_size * max_seq_len * num_kv_heads * head_size * 2
                gflops = flops / (time_us * 1e-6) / 1e9
                results.append({
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "head_size": head_size,
                    "num_kv_heads": num_kv_heads,
                    "time_us": round(time_us, 2),
                    "gflops": round(gflops, 2),
                    "kv_cache_dtype": kv_cache_dtype,
                })
            except Exception as exc:
                results.append({
                    "batch_size": batch_size,
                    "max_seq_len": max_seq_len,
                    "head_size": head_size,
                    "num_kv_heads": num_kv_heads,
                    "time_us": -1.0,
                    "gflops": -1.0,
                    "kv_cache_dtype": kv_cache_dtype,
                    "error": str(exc)[:80],
                })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16],
        help="batch sizes to sweep",
    )
    parser.add_argument(
        "--max-seq-lens", type=int, nargs="+", default=[256, 1024, 4096],
        help="max sequence lengths to sweep",
    )
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--num-query-heads", type=int, default=12,
                        help="Q heads per layer (matches OPT-125m default)")
    parser.add_argument("--num-kv-heads", type=int, default=12,
                        help="KV heads per layer (MHA: = num_query_heads)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="optional path to write CSV (default: stdout)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    results = run_attention_sweep(
        batch_sizes=args.batch_sizes,
        max_seq_lens=args.max_seq_lens,
        head_size=args.head_size,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
    )

    if not results:
        print("[profile_attention] no results — kernel calls all failed", file=sys.stderr)
        return 1

    fieldnames = [
        "batch_size", "max_seq_len", "head_size", "num_kv_heads",
        "time_us", "gflops", "kv_cache_dtype", "error",
    ]
    out_fp = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    finally:
        if args.output:
            out_fp.close()

    # Summary line
    valid = [r for r in results if r["time_us"] > 0]
    if valid:
        max_gflops = max(r["gflops"] for r in valid)
        print(
            f"[profile_attention] {len(valid)}/{len(results)} runs OK; "
            f"peak {max_gflops:.1f} GFLOPS (decode, batch × seq sweep)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
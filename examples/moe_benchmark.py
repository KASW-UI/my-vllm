"""examples/moe_benchmark.py — Fused vs unfused MoE synthetic benchmark.

Compares the Triton fused MoE kernel against a naive per-expert
Python loop on synthetic input.  The benchmark measures wall-clock
for both paths and reports TFLOPS achieved.

Mixture-of-Experts is the canonical sparse activation pattern in
modern LLMs (Mixtral 8x7B, DeepSeek-V3).  The "fused" path sorts
tokens by expert assignment and runs a single grouped matmul
kernel; the "unfused" path iterates per-expert (one matmul per
expert).  The fused path avoids kernel-launch overhead and gets
better L2 cache locality because adjacent blocks share an expert.

Usage:

    # Default sweep
    python examples/moe_benchmark.py

    # Specific config (Mixtral-style)
    python examples/moe_benchmark.py --num-tokens 512 --num-experts 8 \
        --top-k 2 --hidden-dim 4096 --intermediate-dim 14336

Output:
    CSV (to stdout or path) with columns:
        provider, num_tokens, num_experts, top_k, hidden_dim,
        intermediate_dim, time_us, tflops

JD: 异构推理 — establishes a measurement harness for MoE kernel
selection (when to use fused_moe vs per-expert loop).

Notes:

  * The "unfused" path uses ``torch.bmm`` after constructing per-
    expert input tensors.  It does NOT model the kernel-launch
    overhead of `for expert in experts:` (which is 50-100x more
    overhead) — instead it represents an "idealized" per-expert
    implementation that already amortizes launch cost.

  * The "fused" path uses my-vllm's ``fused_moe_triton`` which is
    the FP16-only Triton kernel (C221).
"""

import argparse
import csv
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch


def build_fused_runner(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
    block_size_m: int = 64,
) -> Callable[[], torch.Tensor]:
    """Returns a closure that runs the fused Triton MoE kernel."""
    from my_vllm.model_executor.layers.fused_moe.fused_moe_triton import (
        fused_moe_triton,
    )

    def run() -> torch.Tensor:
        return fused_moe_triton(
            x, w1, w2, topk_weights, topk_ids, top_k, num_experts,
            block_size_m=block_size_m,
        )
    return run


def build_unfused_runner(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    num_experts: int,
) -> Callable[[], torch.Tensor]:
    """Returns a closure that runs the unfused (per-expert) MoE path.

    For each expert, we:
      1. Find which (token, k) pairs route to this expert.
      2. Gather their activations.
      3. Run MLP = silu(x @ w1_gate.T) @ w2.T.
      4. Scatter weighted outputs back.

    This is the natural "loop over experts" implementation that
    vLLM's fused kernel replaces.
    """
    # Pre-compute the routing index per expert to avoid the per-call
    # scan cost (we want to measure the matmul time, not the Python
    # loop overhead).
    expert_to_tokens: Dict[int, List[Tuple[int, int]]] = {}
    for t in range(x.size(0)):
        for k in range(top_k):
            e = int(topk_ids[t, k].item())
            expert_to_tokens.setdefault(e, []).append((t, k))

    inter_dim = w1.size(1) // 2

    def run() -> torch.Tensor:
        output = torch.zeros_like(x)
        for e, pairs in expert_to_tokens.items():
            if not pairs:
                continue
            idx = torch.tensor(pairs, device=x.device, dtype=torch.long)
            token_idx = idx[:, 0]
            k_idx = idx[:, 1]
            x_e = x[token_idx]
            # Gate (first half of w1) only — matches fused path.
            gate = x_e @ w1[e, :inter_dim, :].T
            gate = torch.nn.functional.silu(gate)
            expert_out = gate @ w2[e].T  # [num_assigned, hidden_dim]
            w = topk_weights[token_idx, k_idx].unsqueeze(-1).to(expert_out.dtype)
            output.index_add_(0, token_idx, w * expert_out)
        return output

    return run


PROVIDERS: Dict[str, Callable] = {
    "fused":   build_fused_runner,
    "unfused": build_unfused_runner,
}


def time_kernel(
    fn: Callable[[], torch.Tensor],
    warmup: int = 5,
    iters: int = 50,
) -> float:
    """Median wall-clock per call in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return (start.elapsed_time(end) / iters) * 1e3  # ms → us


def make_inputs(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hidden_dim: int,
    intermediate_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    random.seed(seed)

    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype) * 0.05
    # w1 has shape [E, 2*inter, hidden] — gate || up (Mixtral convention).
    w1 = torch.randn(
        num_experts, 2 * intermediate_dim, hidden_dim,
        device=device, dtype=dtype,
    ) * 0.05
    # w2 has shape [E, hidden, inter].
    w2 = torch.randn(
        num_experts, hidden_dim, intermediate_dim,
        device=device, dtype=dtype,
    ) * 0.05

    # Routing: softmax over num_experts, take top_k.
    router_logits = torch.randn(num_tokens, num_experts, device=device)
    topk_weights, topk_ids = torch.topk(
        torch.softmax(router_logits, dim=-1), top_k, dim=-1,
    )
    topk_weights = topk_weights.float()
    topk_ids = topk_ids.int()

    return x, w1, w2, topk_weights, topk_ids


def run_sweep(
    configs: List[dict],
    providers: List[str],
    dtype: torch.dtype,
    seed: int,
    warmup: int,
    iters: int,
) -> List[dict]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for moe_benchmark")

    device = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"# dtype: {dtype}", file=sys.stderr)
    print(f"# providers: {providers}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[dict] = []
    for cfg in configs:
        num_tokens   = cfg["num_tokens"]
        num_experts  = cfg["num_experts"]
        top_k        = cfg["top_k"]
        hidden_dim   = cfg["hidden_dim"]
        intermediate = cfg["intermediate_dim"]

        x, w1, w2, tw, ti = make_inputs(
            num_tokens, num_experts, top_k, hidden_dim, intermediate,
            device, dtype, seed,
        )

        # MoE FLOPs per call: 2 matmuls per (token, expert) pair.
        #   gate:  num_tokens * top_k * (2 * hidden_dim * intermediate_dim)
        #   down:  num_tokens * top_k * (2 * hidden_dim * intermediate_dim)
        flops = 2 * num_tokens * top_k * (2 * hidden_dim * intermediate)

        for prov_name in providers:
            try:
                runner = PROVIDERS[prov_name](
                    x, w1, w2, tw, ti, top_k, num_experts,
                )
                t_us = time_kernel(runner, warmup=warmup, iters=iters)
            except Exception as exc:
                print(
                    f"# skipped {prov_name} num_tokens={num_tokens} E={num_experts}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue

            tflops = flops * 1e-12 / (t_us * 1e-6)
            results.append({
                "provider":        prov_name,
                "num_tokens":      num_tokens,
                "num_experts":     num_experts,
                "top_k":           top_k,
                "hidden_dim":      hidden_dim,
                "intermediate_dim": intermediate,
                "time_us":         round(t_us, 3),
                "tflops":          round(tflops, 2),
            })
    return results


def print_table(results: List[dict]) -> None:
    print("# --- summary (TFLOP/s, larger is better) ---", file=sys.stderr)
    header = (
        f"{'tokens':>7} {'E':>3} {'K':>2} {'H':>5} {'I':>6}  "
        f"{'provider':<8} {'t_us':>10} {'TFLOPS':>8}"
    )
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for r in results:
        print(
            f"{r['num_tokens']:>7} {r['num_experts']:>3} {r['top_k']:>2} "
            f"{r['hidden_dim']:>5} {r['intermediate_dim']:>6}  "
            f"{r['provider']:<8} {r['time_us']:>10.2f} {r['tflops']:>8.2f}",
            file=sys.stderr,
        )


def write_csv(results: List[dict], path: str | None) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    out = open(path, "w", newline="") if path else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    finally:
        if path:
            out.close()


# Default Mixtral-8x7B-shaped configs
DEFAULT_CONFIGS = [
    # (num_tokens, num_experts, top_k, hidden_dim, intermediate_dim)
    {"num_tokens": 64,   "num_experts": 8,  "top_k": 2,
     "hidden_dim": 512,  "intermediate_dim": 2048},
    {"num_tokens": 256,  "num_experts": 8,  "top_k": 2,
     "hidden_dim": 1024, "intermediate_dim": 4096},
    {"num_tokens": 512,  "num_experts": 8,  "top_k": 2,
     "hidden_dim": 2048, "intermediate_dim": 8192},
    {"num_tokens": 1024, "num_experts": 8,  "top_k": 2,
     "hidden_dim": 4096, "intermediate_dim": 14336},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fused vs unfused MoE synthetic benchmark.",
    )
    p.add_argument(
        "--providers", nargs="+",
        default=["fused", "unfused"],
        choices=list(PROVIDERS.keys()),
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16"],
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--num-tokens", type=int, default=None,
        help="Override num_tokens for all configs.",
    )
    p.add_argument(
        "--num-experts", type=int, default=None,
        help="Override num_experts for all configs.",
    )
    p.add_argument(
        "--top-k", type=int, default=None,
        help="Override top_k for all configs.",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=None,
    )
    p.add_argument(
        "--intermediate-dim", type=int, default=None,
    )
    p.add_argument(
        "csv_out", nargs="?", default=None,
        help="CSV output path (default: stdout).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}

    configs = [dict(c) for c in DEFAULT_CONFIGS]
    if args.num_tokens is not None:
        for c in configs:
            c["num_tokens"] = args.num_tokens
    if args.num_experts is not None:
        for c in configs:
            c["num_experts"] = args.num_experts
    if args.top_k is not None:
        for c in configs:
            c["top_k"] = args.top_k
    if args.hidden_dim is not None:
        for c in configs:
            c["hidden_dim"] = args.hidden_dim
    if args.intermediate_dim is not None:
        for c in configs:
            c["intermediate_dim"] = args.intermediate_dim

    t0 = time.perf_counter()
    results = run_sweep(
        configs=configs,
        providers=args.providers,
        dtype=dtype_map[args.dtype],
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
    )
    elapsed = time.perf_counter() - t0

    print_table(results)
    write_csv(results, args.csv_out)
    print(f"# elapsed: {elapsed:.2f}s, "
          f"{len(results)} measurements", file=sys.stderr)


if __name__ == "__main__":
    main()

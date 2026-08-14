"""examples/quant_gemm_benchmark.py — FP16 vs FP8 vs INT8 GEMM comparison.

Sweeps (M, K, N) shapes and measures wall-clock + TFLOPS for three
quantized GEMM paths exposed by ``my_vllm._custom_ops``:

  1. **BF16 baseline** — ``torch.nn.functional.linear`` (no quant).
  2. **FP8** (``float8_e4m3fn``) — ``scaled_fp8_quant`` +
     ``cutlass_scaled_mm`` (dispatches to ``torch._scaled_mm``,
     cuBLAS / CUTLASS-backed on sm_89+).
  3. **INT8** — ``scaled_int8_quant`` + ``cutlass_scaled_mm``
     (dispatches to ``torch._int_mm``).

INT4 (Marlin) is listed as a reference comparison but **not** measured
in my-vllm — the CUTLASS Marlin kernel is 1000+ lines of templates and
out of scope for this engineering-subset project (see
``docs/notes/cutlass-scaled-mm.md`` §4).

Usage:

    # Default sweep (small / medium / large shapes)
    python examples/quant_gemm_benchmark.py

    # Custom M sweep
    python examples/quant_gemm_benchmark.py --ms 1 16 128 1024 4096

    # Specific K×N (e.g. an OPT-125M attention projection)
    python examples/quant_gemm_benchmark.py --k 768 --n 768

Output: CSV to stdout (path = first CLI arg, default = stdout).  Columns:

    provider, m, n, k, dtype_a, dtype_b, time_us, tflops, numel_in, numel_w

The TFLOPS convention follows vLLM's ``benchmark_fp8_gemm.py``:

    tflops = 2 * M * N * K * 1e-12 / (time_s)

JD: 异构推理 + 加分 #1 — establish a measurement harness so we can
attribute GEMM time to FP8 vs INT8 vs BF16 in quantized serving
configurations.  This is the first step toward deciding which quant
format to use for OPT / Llama-class models at batch sizes typical of
LLM serving (M in [1, 16K]).

Why this script (vs the full vLLM benchmark runner):

  * No VllmConfig / scheduler / model loader needed.
  * Direct kernel invocation = clean timing of just the matmul.
  * Reproducible across runs (seeded).
  * CSV output → grep / awk / pandas downstream.
"""

import argparse
import csv
import random
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch

from my_vllm._custom_ops import (
    cutlass_scaled_mm,
    cutlass_scaled_mm_fn,
    cutlass_scaled_mm_supports_fp8,
    scaled_fp8_quant,
    scaled_int8_quant,
)


# ---------------------------------------------------------------------------
# Default shape sweep — covers latency-bound (M=1) → BW-bound (M=4096+)
# ---------------------------------------------------------------------------

# Default M sweep mirrors vLLM's ``benchmark_fp8_gemm.py``:
#   [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
# We trim to a representative subset to keep the script fast.

DEFAULT_MS = [1, 16, 128, 1024, 4096]


# (K, N) shapes — typical LLM linear layers
DEFAULT_SHAPES: List[Tuple[int, int]] = [
    (768, 768),       # OPT-125M q_proj / k_proj / v_proj
    (768, 3072),      # OPT-125M fc1 (MLP up)
    (3072, 768),      # OPT-125M fc2 (MLP down)
    (1024, 1024),     # square
    (4096, 4096),     # larger square
]


# ---------------------------------------------------------------------------
# Provider builders — each returns a closure that runs ONE matmul
# ---------------------------------------------------------------------------


def build_bf16_runner(a: torch.Tensor, b: torch.Tensor,
                      out_dtype: torch.dtype
                      ) -> Callable[[], torch.Tensor]:
    """BF16 baseline: torch.nn.functional.linear.

    ``out_dtype`` is accepted but ignored — the baseline just runs the
    matmul in the input dtype.
    """
    def run() -> torch.Tensor:
        return torch.nn.functional.linear(a, b)
    return run


def build_fp8_runner(a: torch.Tensor, b: torch.Tensor,
                     out_dtype: torch.dtype
                     ) -> Callable[[], torch.Tensor]:
    """FP8 path: dynamic per-tensor quant + scaled_mm.

    Quantizes both A and B once (the benchmark loop measures only the
    matmul, not the quant — the quant is a fixed pre-pass).  In
    production this would be either static (calibrated) or per-token
    dynamic; per-tensor dynamic is the simplest measurement case.
    """
    # Quantize outside the timed loop.
    a_fp8, scale_a = scaled_fp8_quant(a)
    b_fp8, scale_b = scaled_fp8_quant(b)  # b is [N, K] so quant per-row

    def run() -> torch.Tensor:
        return cutlass_scaled_mm_fn(
            a_fp8, b_fp8, scale_a, scale_b, out_dtype,
        )
    return run


def build_int8_runner(a: torch.Tensor, b: torch.Tensor,
                      out_dtype: torch.dtype
                      ) -> Callable[[], torch.Tensor]:
    """INT8 path: symmetric per-tensor quant + int_mm → cast to out_dtype."""
    a_i8, scale_a, _ = scaled_int8_quant(a, symmetric=True)
    b_i8, scale_b, _ = scaled_int8_quant(b, symmetric=True)

    def run() -> torch.Tensor:
        return cutlass_scaled_mm_fn(
            a_i8, b_i8, scale_a, scale_b, out_dtype,
        )
    return run


PROVIDERS: Dict[str, Callable[[torch.Tensor, torch.Tensor, torch.dtype],
                              Callable[[], torch.Tensor]]] = {
    "bf16": build_bf16_runner,
    "fp8":  build_fp8_runner,
    "int8": build_int8_runner,
}


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def time_kernel(fn: Callable[[], torch.Tensor],
                warmup: int = 10,
                iters: int = 50) -> float:
    """Median wall-clock per call in microseconds.

    Uses ``torch.cuda.Event`` (GPU-side timing) bracketing the loop,
    following the pattern in vLLM's ``benchmark_fp8_gemm.py``.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    return (total_ms / iters) * 1e3  # ms → us


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def run_sweep(
    ms: List[int],
    shapes: List[Tuple[int, int]],
    providers: List[str],
    out_dtype: torch.dtype,
    seed: int,
    warmup: int,
    iters: int,
) -> List[dict]:
    """Run the full sweep and return a list of result dicts."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for quant_gemm_benchmark")

    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda")
    fp8_supported = cutlass_scaled_mm_supports_fp8()
    print(f"# device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
    print(f"# fp8_supported: {fp8_supported}", file=sys.stderr)
    print(f"# dtype: {out_dtype}", file=sys.stderr)
    print(f"# providers: {providers}", file=sys.stderr)
    print(file=sys.stderr)

    results: List[dict] = []

    for M in ms:
        for K, N in shapes:
            a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
            b = torch.randn((N, K), device=device, dtype=torch.bfloat16)

            for prov_name in providers:
                # Skip FP8 if hardware doesn't support it.
                if prov_name == "fp8" and not fp8_supported:
                    continue
                try:
                    runner = PROVIDERS[prov_name](a, b, out_dtype)
                    t_us = time_kernel(runner, warmup=warmup, iters=iters)
                except Exception as exc:
                    print(
                        f"# skipped {prov_name} M={M} N={N} K={K}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                tflops = 2.0 * M * N * K * 1e-12 / (t_us * 1e-6)
                results.append({
                    "provider":   prov_name,
                    "m":          M,
                    "n":          N,
                    "k":          K,
                    "dtype_a":    str(a.dtype).replace("torch.", ""),
                    "dtype_b":    str(b.dtype).replace("torch.", ""),
                    "time_us":    round(t_us, 3),
                    "tflops":     round(tflops, 2),
                    "numel_in":   M * K,
                    "numel_w":    N * K,
                })
    return results


def print_table(results: List[dict]) -> None:
    """Human-readable summary to stderr."""
    print("# --- summary (TFLOP/s, larger is better) ---", file=sys.stderr)
    header = f"{'M':>6} {'N':>6} {'K':>6}  {'provider':<6} {'t_us':>10} {'TFLOPS':>8}"
    print(header, file=sys.stderr)
    print("-" * len(header), file=sys.stderr)
    for r in results:
        print(
            f"{r['m']:>6} {r['n']:>6} {r['k']:>6}  "
            f"{r['provider']:<6} {r['time_us']:>10.2f} "
            f"{r['tflops']:>8.2f}",
            file=sys.stderr,
        )


def write_csv(results: List[dict], path: str | None) -> None:
    """Write CSV to ``path`` (or stdout if path is None)."""
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FP16 / FP8 / INT8 GEMM benchmark (my-vllm).",
    )
    p.add_argument(
        "--ms", nargs="+", type=int, default=DEFAULT_MS,
        help=f"M values to sweep (default: {DEFAULT_MS})",
    )
    p.add_argument(
        "--shape", nargs=2, type=int, action="append",
        metavar=("K", "N"),
        help="(K, N) shape pair; can be repeated.  Default: see DEFAULT_SHAPES.",
    )
    p.add_argument(
        "--providers", nargs="+",
        default=["bf16", "fp8", "int8"],
        choices=list(PROVIDERS.keys()),
        help="Providers to compare.",
    )
    p.add_argument(
        "--dtype", default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Output / baseline dtype.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for reproducible timing.",
    )
    p.add_argument(
        "--warmup", type=int, default=10,
        help="Warmup iterations per measurement.",
    )
    p.add_argument(
        "--iters", type=int, default=50,
        help="Timing iterations per measurement.",
    )
    p.add_argument(
        "csv_out", nargs="?", default=None,
        help="CSV output path (default: stdout).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    out_dtype = dtype_map[args.dtype]

    if args.shape:
        shapes = [(k, n) for k, n in args.shape]
    else:
        shapes = DEFAULT_SHAPES

    t0 = time.perf_counter()
    results = run_sweep(
        ms=args.ms,
        shapes=shapes,
        providers=args.providers,
        out_dtype=out_dtype,
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

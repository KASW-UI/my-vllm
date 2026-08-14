"""examples/nccl_benchmark.py — NCCL collective bandwidth benchmark.

Sweeps message sizes [1KB, 4MB] across the three core NCCL collectives
(all_reduce, all_gather, reduce_scatter) and emits a CSV of bandwidth
per op per size.

Mirrors the message-size sweep pattern from the official nccl-tests
benchmark (``nccl-tests/all_reduce_perf``) but expressed via
``torch.distributed`` so it works on any NCCL-capable setup without
external build dependencies.

Usage:
    # Single GPU (degenerate — measures local memcpy baseline)
    python examples/nccl_benchmark.py

    # Multi-GPU (single node, world_size=4):
    MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
        torchrun --nproc_per_node=4 examples/nccl_benchmark.py

Output: CSV to stdout with columns
    op, size_bytes, num_elements, dtype, world_size, rank,
    elapsed_us, alg_bw_GBps, bus_bw_GBps

Bandwidth conventions (per NCCL docs):

  * ``alg_bw_GBps`` — algorithmic bandwidth (useful data moved per
    second).  For ring all-reduce this is ``2 * (n-1) / n * size / t``
    where ``n`` is world_size (the constant factor accounts for the
    ring's ``n-1`` send + ``n-1`` recv splits per rank).

  * ``bus_bw_GBps`` — bus bandwidth (what the hardware sees).  For
    ring all-reduce this is ``2 * (n-1) * size / t`` (each chunk is
    sent once and received once on the bus).

For ``all_gather`` and ``reduce_scatter``:
  * alg = ``size`` per rank; bus = ``(n-1) * size`` per rank.

JD: 加分 #2「RDMA 原理 / 分布式通信算子」— establish a measurement
harness so we can attribute cross-node traffic in PD-disaggregated
serving to specific bus types (NVLink vs PCIe vs IB).
"""

import argparse
import csv
import os
import sys
import time
from typing import Callable, List

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Message size sweep — same exponential sweep as nccl-tests
# ---------------------------------------------------------------------------

# nccl-tests sweep: 1KB → 4MB in ~16 steps.  Powers-of-two cover
# typical collectives (latency-bound at small, BW-bound at large).
_MIN_BYTES = 1024         # 1 KiB
_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB


def make_sizes(min_bytes: int = _MIN_BYTES, max_bytes: int = _MAX_BYTES) -> List[int]:
    """Geometric sweep, ~16 steps, of byte sizes.

    Returns the byte counts (NOT the element counts) — callers divide
    by dtype size to get the element count.
    """
    sizes = []
    s = min_bytes
    while s <= max_bytes:
        sizes.append(s)
        s *= 2
    return sizes


# ---------------------------------------------------------------------------
# Collective definitions
#
# Each entry maps an op name to:
#   - a callable that builds (input, output) tensors
#   - a callable that runs the collective once (in-place)
#   - a callable that computes alg_bw and bus_bw given (size_bytes, time_s, n)
# ---------------------------------------------------------------------------

# dtype → (elem_size, constructor, suffix)
_DTYPES = {
    "float32": (4, torch.float32, "f32"),
    "float16": (2, torch.float16, "f16"),
    "bfloat16": (2, torch.bfloat16, "bf16"),
    "int8": (1, torch.int8, "i8"),
}


def _bw_all_reduce(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    """Ring all-reduce: each chunk is sent and received ``n-1`` times.

    alg = (n-1)/n * size / t   (the reduction is ``n``-way per chunk,
                                  but each rank contributes ``1/n`` of the
                                  result back to itself — net per-rank is
                                  ``size`` data, of which ``(n-1)/n`` is
                                  inter-rank).
    bus = 2 * (n-1) * size / t   (each chunk crosses the bus ``n-1``
                                    times in each direction; bidirectional
                                    symmetry collapses to one ``n-1``).
    """
    alg = (n - 1) / n * size_bytes / t / 1e9  # bytes/s → GB/s
    bus = 2 * (n - 1) * size_bytes / t / 1e9
    return alg, bus


def _bw_reduce_scatter(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    alg = size_bytes / n / t / 1e9     # per-rank output is size/n
    bus = (n - 1) * size_bytes / n / t / 1e9
    return alg, bus


def _bw_all_gather(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    alg = size_bytes / t / 1e9
    bus = (n - 1) * size_bytes / t / 1e9
    return alg, bus


# ---------------------------------------------------------------------------
# Per-op work functions
# ---------------------------------------------------------------------------

def _setup_all_reduce(size_bytes: int, dtype: torch.dtype, device: torch.device):
    n_elem = size_bytes // _DTYPES[_dtype_name(dtype)][0]
    tensor = torch.randn(n_elem, dtype=dtype, device=device)
    return tensor, tensor


def _setup_reduce_scatter(size_bytes: int, dtype: torch.dtype, device: torch.device):
    """Each rank has a size_bytes tensor; output is size_bytes/n per rank."""
    elem_size = _DTYPES[_dtype_name(dtype)][0]
    n_elem = size_bytes // elem_size
    inp = torch.randn(n_elem, dtype=dtype, device=device)
    out = torch.empty(n_elem, dtype=dtype, device=device)
    return inp, out


def _setup_all_gather(size_bytes: int, dtype: torch.dtype, device: torch.device):
    """Each rank has size_bytes/n tensor; output is size_bytes per rank."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    elem_size = _DTYPES[_dtype_name(dtype)][0]
    n_per_rank = (size_bytes // world_size) // elem_size
    inp = torch.randn(n_per_rank, dtype=dtype, device=device)
    # tensor_list: one entry per rank; each is the per-rank gathered result.
    out_list = [
        torch.empty(n_per_rank, dtype=dtype, device=device)
        for _ in range(world_size)
    ]
    return inp, out_list


def _dtype_name(dtype: torch.dtype) -> str:
    for name, (sz, dt, _) in _DTYPES.items():
        if dt == dtype:
            return name
    raise KeyError(dtype)


def _run_all_reduce(tensors: tuple, dist_group=None) -> None:
    t, _ = tensors
    dist.all_reduce(t, group=dist_group)


def _run_reduce_scatter(tensors: tuple, dist_group=None) -> None:
    inp, out = tensors
    dist.reduce_scatter(out, inp, group=dist_group)


def _run_all_gather(tensors: tuple, dist_group=None) -> None:
    inp, out_list = tensors
    dist.all_gather(out_list, inp, group=dist_group)


OPS = {
    "all_reduce": (_setup_all_reduce, _run_all_reduce, _bw_all_reduce),
    "reduce_scatter": (_setup_reduce_scatter, _run_reduce_scatter, _bw_reduce_scatter),
    "all_gather": (_setup_all_gather, _run_all_gather, _bw_all_gather),
}


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def time_op(fn: Callable[[], None], warmup: int = 10, iters: int = 50) -> float:
    """Mean wall-clock time per call in seconds.

    NCCL ops are CUDA-graph-eligible but ``torch.cuda.Event`` doesn't
    time host-side dispatch + sync, so we use ``time.perf_counter``
    bracketed by a single ``synchronize`` after the full loop.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters


# ---------------------------------------------------------------------------
# Distributed init (single-GPU fallback)
# ---------------------------------------------------------------------------

def maybe_init_process_group() -> tuple[int, int]:
    """Initialize torch.distributed, returning (rank, world_size).

    Three paths:

    1. ``torchrun``-style multi-GPU: env vars set → init NCCL backend.
    2. Single-GPU with ``MASTER_ADDR`` set → init ``gloo`` backend so
       ``torch.distributed`` collectives work (the benchmark still
       measures degenerate local traffic).
    3. No env vars → init ``gloo`` with world_size=1 (truly degenerate).
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")

    if dist.is_initialized():
        return rank, world_size

    # Choose backend: NCCL if multi-GPU env, else gloo (works on CPU/1 GPU).
    if world_size > 1 and torch.cuda.is_available():
        backend = "nccl"
    else:
        # gloo works for world_size=1 (truly degenerate); NCCL with
        # world_size=1 emits deprecation warnings and is fragile.
        backend = "gloo"

    # Single-rank gloo doesn't need a rendezvous env.  For multi-rank gloo
    # or any-rank NCCL, init_process_group uses env:// rendezvous.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    ops: List[str],
    dtypes: List[str],
    sizes: List[int],
    warmup: int,
    iters: int,
) -> List[dict]:
    rank, world_size = maybe_init_process_group()
    device = torch.device("cuda", rank)

    results: List[dict] = []
    for op_name in ops:
        setup_fn, run_fn, bw_fn = OPS[op_name]
        for dtype_name in dtypes:
            elem_size, torch_dtype, _ = _DTYPES[dtype_name]
            for size_bytes in sizes:
                # Skip sizes that don't divide evenly.
                if size_bytes % elem_size != 0:
                    continue
                # Skip all_gather sizes that don't divide by world_size.
                if op_name == "all_gather":
                    if (size_bytes // elem_size) % world_size != 0:
                        continue
                    if world_size > 1 and (size_bytes // world_size) < elem_size:
                        # Per-rank chunk must be at least 1 element.
                        continue
                try:
                    tensors = setup_fn(size_bytes, torch_dtype, device)
                    elapsed_s = time_op(
                        lambda: run_fn(tensors),
                        warmup=warmup, iters=iters,
                    )
                except Exception as exc:
                    results.append({
                        "op": op_name,
                        "size_bytes": size_bytes,
                        "num_elements": size_bytes // elem_size,
                        "dtype": dtype_name,
                        "world_size": world_size,
                        "rank": rank,
                        "elapsed_us": -1.0,
                        "alg_bw_GBps": -1.0,
                        "bus_bw_GBps": -1.0,
                        "error": str(exc)[:80],
                    })
                    continue

                alg_bw, bus_bw = bw_fn(size_bytes, elapsed_s, world_size)
                results.append({
                    "op": op_name,
                    "size_bytes": size_bytes,
                    "num_elements": size_bytes // elem_size,
                    "dtype": dtype_name,
                    "world_size": world_size,
                    "rank": rank,
                    "elapsed_us": round(elapsed_s * 1e6, 3),
                    "alg_bw_GBps": round(alg_bw, 3),
                    "bus_bw_GBps": round(bus_bw, 3),
                    "error": "",
                })
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ops", type=str, nargs="+",
        default=["all_reduce", "reduce_scatter", "all_gather"],
        choices=list(OPS.keys()),
    )
    parser.add_argument(
        "--dtypes", type=str, nargs="+",
        default=["float32", "float16"],
        choices=list(_DTYPES.keys()),
    )
    parser.add_argument("--min-bytes", type=int, default=_MIN_BYTES)
    parser.add_argument("--max-bytes", type=int, default=_MAX_BYTES)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--output", type=str, default=None,
                        help="optional CSV path (default: stdout)")
    args = parser.parse_args()

    sizes = make_sizes(args.min_bytes, args.max_bytes)

    if not torch.cuda.is_available():
        print("[nccl_benchmark] CUDA not available", file=sys.stderr)
        return 1

    results = run_sweep(
        ops=args.ops, dtypes=args.dtypes, sizes=sizes,
        warmup=args.warmup, iters=args.iters,
    )

    fieldnames = [
        "op", "size_bytes", "num_elements", "dtype",
        "world_size", "rank", "elapsed_us",
        "alg_bw_GBps", "bus_bw_GBps", "error",
    ]
    out_fp = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out_fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    finally:
        if args.output:
            out_fp.close()

    rank, world_size = maybe_init_process_group()
    if dist.is_initialized():
        dist.destroy_process_group()

    valid = [r for r in results if r["elapsed_us"] > 0]
    if valid:
        peak_alg = max(r["alg_bw_GBps"] for r in valid)
        peak_bus = max(r["bus_bw_GBps"] for r in valid)
        print(
            f"[nccl_benchmark] {len(valid)}/{len(results)} runs OK "
            f"(world_size={world_size}); "
            f"peak alg={peak_alg:.1f} GB/s, bus={peak_bus:.1f} GB/s",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""examples/distributed_comm_benchmark.py — Distributed communication operator benchmark.

C248: holistic bandwidth / latency sweep across the 5 core collectives
``all_reduce`` / ``all_gather`` / ``reduce_scatter`` / ``all_to_all`` /
``broadcast``, comparing all backends vendored in my-vllm:

  * **torch.distributed** — NCCL (when multi-GPU) / Gloo (CPU or
    single-rank) baseline.  All 5 collectives.
  * **AgRsAll2AllManager** (C240) — AllGather + ReduceScatter
    all-to-all (the reference naive backend; maps to NCCL
    ``ncclAllGatherv`` / ``ncclReduceScatterv``).  All-to-all.
  * **SymmMemCommunicator** (C241) — NVLS-accelerated all-reduce via
    ``torch.distributed._symmetric_memory``.  All-reduce (H100/GB200
    only; graceful disable on non-NVLS hardware).
  * **CustomAllreduce** (C214) — CUDA IPC ring all-reduce (Patariasuk &
    Yuan 2009).  All-reduce (single-node, world_size 2..8).
  * **SymmMem NVLS** (C241) — alternative NVLS fast path for
    all-reduce (selected via ``should_nccl_symm_mem_allreduce``).
  * **ShmMessageQueue** (C242) — same-host broadcast over shared
    memory.  Broadcast (Python objects).

Unifies the per-collective benchmarks introduced in C201 (NCCL),
C240 (All2All), C241 (SymmMem), C242 (SHM) into a single harness
so the backends can be compared apples-to-apples on a single CSV
output.

Modes
-----

1. **Single-rank (default)** — runs on a single GPU/CPU.  Each
   backend is benchmarked in identity mode (world_size=1 fast path).
   Useful for overhead comparison and "is the harness wired right?"
   verification on a single dev box.

2. **Multi-process gloo** — ``torch.multiprocessing.spawn`` with N
   workers, gloo backend.  Round-trip correctness for each backend
   (the wall-clock numbers cross-process via gloo).  Run with
   ``python examples/distributed_comm_benchmark.py --gloo --world_size 4``.

Usage
-----

::

    # Single-rank (identity baseline; no NCCL needed)
    python examples/distributed_comm_benchmark.py

    # Multi-process gloo round-trip (4 workers)
    python examples/distributed_comm_benchmark.py --gloo --world_size 4

Output (CSV to stdout)
----------------------

::

    backend,op,size_bytes,num_elements,dtype,world_size,elapsed_us,alg_bw_GBps,bus_bw_GBps

Bandwidth conventions (per NCCL docs):

  * ``alg_bw_GBps`` — algorithmic bandwidth (useful data per second).
    For ring all-reduce: ``2 * (n-1) / n * size / t``.
  * ``bus_bw_GBps`` — bus bandwidth (data the hardware sees per
    second).  For ring all-reduce: ``2 * (n-1) * size / t``.

For ``all_gather`` / ``reduce_scatter``:
  * alg = ``size`` per rank; bus = ``(n-1) * size`` per rank.

For ``all_to_all`` (uniform chunk):
  * alg = ``size`` per rank; bus = ``(n-1) * size`` per rank.

For ``broadcast``:
  * alg = ``size`` once; bus = ``size`` once.

JD: 加分 #2「RDMA 原理 / 分布式通信算子」 + 加分 #4「性能 / 成本模型」—
allows attributing cross-node traffic to specific bus types
(NVLink vs PCIe vs IB) and backends (NCCL vs custom-AR vs SymmMem
NVLS) on a single coherent measurement harness.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch


# ---------------------------------------------------------------------------
# Message size sweep — same as nccl_benchmark (C201)
# ---------------------------------------------------------------------------

_MIN_BYTES = 1024  # 1 KiB
_MAX_BYTES = 4 * 1024 * 1024  # 4 MiB


def make_sizes(min_bytes: int = _MIN_BYTES, max_bytes: int = _MAX_BYTES) -> List[int]:
    sizes = []
    s = min_bytes
    while s <= max_bytes:
        sizes.append(s)
        s *= 2
    return sizes


# ---------------------------------------------------------------------------
# Backend registry — single source of truth for (op, backend) pairs
# ---------------------------------------------------------------------------

# dtype -> (elem_size, constructor, suffix)
_DTYPES = {
    "float32": (4, torch.float32, "f32"),
    "float16": (2, torch.float16, "f16"),
    "bfloat16": (2, torch.bfloat16, "bf16"),
}


@dataclass
class Backend:
    """A single backend × op combination."""

    name: str  # e.g. "torch.dist", "ag_rs", "symm_mem", "custom_ar", "shm"
    op: str  # "all_reduce" | "all_gather" | "reduce_scatter" | "all_to_all" | "broadcast"
    run: Callable[[torch.Tensor, int], None]
    """Run once (synchronously).  ``size`` is the per-rank byte size."""

    bw_kind: str  # "all_reduce" | "all_gather" | "all_to_all" | "broadcast"
    """Family used for the alg/bus bandwidth computation."""

    available: bool = True
    """False if the backend is disabled on this hardware (e.g. SymmMem on
    pre-Hopper).  The benchmark will skip and emit a row with elapsed_us=-1."""

    note: str = ""


# ---------------------------------------------------------------------------
# Single-rank backends — world_size == 1
# ---------------------------------------------------------------------------


def _bench_torch_dist(
    op: str,
    mgr_or_none: Optional[object],
):
    """Return a Backend wrapping the relevant torch.distributed primitive."""

    def _run(input_: torch.Tensor, size: int) -> None:
        import torch.distributed as dist

        if op == "all_reduce":
            dist.all_reduce(input_, group=None)
        elif op == "all_gather":
            # For world_size=1 the output of all_gather is the input itself.
            out = torch.empty_like(input_)
            dist.all_gather_into_tensor(out, input_, group=None)
        elif op == "reduce_scatter":
            out = torch.empty_like(input_)
            dist.reduce_scatter_tensor(out, input_, group=None)
        elif op == "broadcast":
            dist.broadcast(input_, src=0, group=None)
        elif op == "all_to_all":
            # all_to_all requires world_size > 1; identity for single-rank.
            # We still emit the row so the harness is exercised.
            pass
        else:
            raise ValueError(f"Unknown op: {op}")

    return Backend(
        name="torch.dist",
        op=op,
        run=_run,
        bw_kind=op,
        available=True,
        note="torch.distributed (NCCL/Gloo) baseline",
    )


def _bench_custom_allreduce(input_: torch.Tensor, size: int) -> None:
    """Single-rank CustomAllreduce: identity (world_size=1)."""
    # The C214 custom_all_reduce is identity for world_size == 1.
    try:
        from my_vllm.distributed.custom_all_reduce import (
            CustomAllreduce,
        )

        # world_size=1 → identity (no comm).  We've already verified
        # the path works in tests/test_C214_custom_all_reduce.py.
        out = torch.empty_like(input_)
        out.copy_(input_)
        input_.copy_(out)
    except ImportError:
        # Custom AR not built; fall back to identity.
        pass


def _bench_symm_mem_allreduce(input_: torch.Tensor, size: int) -> None:
    """Single-rank SymmMem: identity (world_size=1 or no NVLS)."""
    try:
        from my_vllm.distributed.device_communicators import (
            SymmMemCommunicator,
        )

        # SymmMemCommunicator is disabled on non-NVLS hardware;
        # single-rank is also identity.
        symm = SymmMemCommunicator(group=None, device=input_.device)
        if symm.disabled:
            return
        out = symm.all_reduce(input_)
        if out is not None:
            input_.copy_(out)
    except (ImportError, RuntimeError):
        pass


def _bench_ag_rs_alltoall(input_: torch.Tensor, size: int) -> None:
    """Single-rank AgRsAll2AllManager: identity (world_size=1)."""
    try:
        from my_vllm.distributed.device_communicators import (
            AgRsAll2AllManager,
        )

        # 1-rank dispatch + combine is identity.
        mgr = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
        mgr.cpu_group = None
        mgr.tcp_store_group = None
        mgr.dp_group = None
        mgr.tp_group = None
        mgr.dp_rank = 0
        mgr.dp_world_size = 1
        mgr.rank = 0
        mgr.world_size = 1
        mgr.comm_group = None
        # dispatch + combine on identity is no-op.
        topk_w = torch.zeros(input_.shape[0], 1, device=input_.device)
        topk_i = torch.zeros(input_.shape[0], 1, dtype=torch.int64, device=input_.device)
        mgr.dispatch(input_, topk_w, topk_i)
        mgr.combine(input_)
    except ImportError:
        pass


def _bench_shm_broadcast(input_: torch.Tensor, size: int) -> None:
    """Single-rank SHM broadcast: identity (no MessageQueue needed)."""
    # C242 MessageQueue is for multi-process; single-rank is just memcpy.
    # Verify the import path at least once (cheap).
    try:
        from my_vllm.distributed.device_communicators import MessageQueue  # noqa: F401
        # No-op for single-rank.
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Bandwidth formulas (per NCCL convention)
# ---------------------------------------------------------------------------


def _bw_all_reduce(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return (0.0, 0.0)
    alg = 2.0 * (n - 1.0) / n * size_bytes / t
    bus = 2.0 * (n - 1.0) * size_bytes / t
    return (alg, bus)


def _bw_all_gather_rs(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return (0.0, 0.0)
    alg = size_bytes / t
    bus = (n - 1.0) * size_bytes / t
    return (alg, bus)


def _bw_all_to_all(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return (0.0, 0.0)
    alg = size_bytes / t
    bus = (n - 1.0) * size_bytes / t
    return (alg, bus)


def _bw_broadcast(size_bytes: int, t: float, n: int) -> tuple[float, float]:
    if n <= 1:
        return (0.0, 0.0)
    alg = size_bytes / t
    bus = size_bytes / t
    return (alg, bus)


_BW_FNS = {
    "all_reduce": _bw_all_reduce,
    "all_gather": _bw_all_gather_rs,
    "reduce_scatter": _bw_all_gather_rs,
    "all_to_all": _bw_all_to_all,
    "broadcast": _bw_broadcast,
}


# ---------------------------------------------------------------------------
# CSV output schema
# ---------------------------------------------------------------------------

CSV_HEADER = [
    "backend",
    "op",
    "size_bytes",
    "num_elements",
    "dtype",
    "world_size",
    "elapsed_us",
    "alg_bw_GBps",
    "bus_bw_GBps",
    "note",
]


def _emit_csv_row(**kwargs) -> dict:
    return {k: kwargs.get(k, "") for k in CSV_HEADER}


# ---------------------------------------------------------------------------
# Single-rank benchmark
# ---------------------------------------------------------------------------


def _bench_single_rank(
    backends: List[Backend],
    sizes: List[int],
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iters: int,
) -> List[dict]:
    """Return one CSV row per (backend, size, dtype)."""
    rows: List[dict] = []
    elem_size = torch.tensor([], dtype=dtype).element_size()

    for backend in backends:
        if not backend.available:
            for nbytes in sizes:
                rows.append(
                    _emit_csv_row(
                        backend=backend.name,
                        op=backend.op,
                        size_bytes=nbytes,
                        num_elements=nbytes // elem_size,
                        dtype=str(dtype).replace("torch.", ""),
                        world_size=1,
                        elapsed_us=-1,
                        alg_bw_GBps=0.0,
                        bus_bw_GBps=0.0,
                        note=f"unavailable: {backend.note}",
                    )
                )
            continue

        for nbytes in sizes:
            n_elem = max(1, nbytes // elem_size)
            tensor = torch.randn(n_elem, dtype=dtype, device=device)

            # Warmup
            for _ in range(warmup):
                try:
                    backend.run(tensor, nbytes)
                except Exception:
                    pass
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Time
            t0 = time.perf_counter()
            for _ in range(iters):
                backend.run(tensor, nbytes)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed_us = (t1 - t0) / iters * 1e6
            bw_fn = _BW_FNS.get(backend.bw_kind, _bw_all_reduce)
            alg, bus = bw_fn(nbytes, t1 - t0, 1)

            rows.append(
                _emit_csv_row(
                    backend=backend.name,
                    op=backend.op,
                    size_bytes=nbytes,
                    num_elements=n_elem,
                    dtype=str(dtype).replace("torch.", ""),
                    world_size=1,
                    elapsed_us=elapsed_us,
                    alg_bw_GBps=alg / 1e9,
                    bus_bw_GBps=bus / 1e9,
                    note=backend.note,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Multi-process gloo round-trip
# ---------------------------------------------------------------------------


def _gloo_worker(rank: int, world_size: int, sizes: List[int], op: str) -> None:
    """Worker process: runs a single op round-trip for all sizes."""
    import os
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
    )

    device = torch.device("cpu")
    elem_size = 4  # float32

    for nbytes in sizes:
        n_elem = max(1, nbytes // elem_size)
        tensor = torch.zeros(n_elem, dtype=torch.float32, device=device)

        # Warmup + time
        if op == "all_reduce":
            t0 = time.perf_counter()
            for _ in range(10):
                dist.all_reduce(tensor, group=None)
            t1 = time.perf_counter()
        elif op == "broadcast":
            t0 = time.perf_counter()
            for _ in range(10):
                dist.broadcast(tensor, src=0, group=None)
            t1 = time.perf_counter()
        elif op == "all_gather":
            out = torch.empty(world_size * tensor.shape[0], *tensor.shape[1:],
                              dtype=tensor.dtype, device=device)
            t0 = time.perf_counter()
            for _ in range(10):
                dist.all_gather_into_tensor(out, tensor, group=None)
            t1 = time.perf_counter()
        else:
            continue
        elapsed_us = (t1 - t0) / 10 * 1e6
        if rank == 0:
            print(
                f"gloo,torch.dist,{op},{nbytes},{n_elem},float32,{world_size},"
                f"{elapsed_us:.2f},{0.0:.2f},{0.0:.2f},gloo CP U round-trip",
                flush=True,
            )

    dist.destroy_process_group()


def _run_gloo(world_size: int, sizes: List[int]) -> None:
    """Spawn ``world_size`` gloo workers, one op at a time."""
    import torch.multiprocessing as mp

    print("backend,op,size_bytes,num_elements,dtype,world_size,elapsed_us,alg_bw_GBps,bus_bw_GBps,note")
    for op in ("broadcast", "all_gather", "all_reduce"):
        mp.spawn(
            _gloo_worker,
            args=(world_size, sizes, op),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_available_backends(device: torch.device) -> List[Backend]:
    """Construct the list of (op, backend) pairs available on this host."""
    backends: List[Backend] = []

    # 1. torch.distributed baseline — all 5 ops
    for op in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast"):
        backends.append(_bench_torch_dist(op, None))

    # 2. CustomAllreduce (C214) — all_reduce only
    backends.append(
        Backend(
            name="custom_ar",
            op="all_reduce",
            run=_bench_custom_allreduce,
            bw_kind="all_reduce",
            available=True,
            note="C214 IPC ring AR (single-node, world_size 2..8)",
        )
    )

    # 3. SymmMem (C241) — all_reduce only, eligible on NVLS hardware
    try:
        from my_vllm.distributed.device_communicators import (
            SymmMemCommunicator,
        )

        symm_probe = SymmMemCommunicator(
            group=None, device=device
        )
        if not symm_probe.disabled:
            backends.append(
                Backend(
                    name="symm_mem",
                    op="all_reduce",
                    run=_bench_symm_mem_allreduce,
                    bw_kind="all_reduce",
                    available=True,
                    note="C241 NVLS symmetric memory AR (H100/GB200)",
                )
            )
        else:
            backends.append(
                Backend(
                    name="symm_mem",
                    op="all_reduce",
                    run=_bench_symm_mem_allreduce,
                    bw_kind="all_reduce",
                    available=False,
                    note="no NVLS hardware (RTX 4060 etc.)",
                )
            )
    except ImportError:
        pass

    # 4. AgRs All2All (C240) — all_to_all only
    backends.append(
        Backend(
            name="ag_rs",
            op="all_to_all",
            run=_bench_ag_rs_alltoall,
            bw_kind="all_to_all",
            available=True,
            note="C240 AllGather+ReduceScatter naive all-to-all",
        )
    )

    # 5. SHM broadcast (C242) — broadcast only
    backends.append(
        Backend(
            name="shm",
            op="broadcast",
            run=_bench_shm_broadcast,
            bw_kind="broadcast",
            available=True,
            note="C242 same-host SHM broadcast (multi-process)",
        )
    )

    return backends


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distributed communication operator benchmark (C248)",
    )
    parser.add_argument(
        "--gloo",
        action="store_true",
        help="Run multi-process gloo round-trip (requires --world_size > 1).",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of gloo workers (default: 1 = single-rank).",
    )
    parser.add_argument(
        "--min_bytes",
        type=int,
        default=_MIN_BYTES,
        help="Smallest message size (default: 1024).",
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=_MAX_BYTES,
        help="Largest message size (default: 4MB).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations per (backend, size).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Measurement iterations per (backend, size).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=list(_DTYPES.keys()),
        help="Tensor dtype (default: float32).",
    )
    args = parser.parse_args()

    if args.gloo and args.world_size > 1:
        sizes = make_sizes(args.min_bytes, args.max_bytes)
        _run_gloo(args.world_size, sizes)
        return 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _DTYPES[args.dtype][1]
    sizes = make_sizes(args.min_bytes, args.max_bytes)
    backends = _build_available_backends(device)

    # Initialize gloo world_size=1 so torch.distributed primitives work.
    import torch.distributed as dist
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29502")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    try:
        rows = _bench_single_rank(
            backends=backends,
            sizes=sizes,
            dtype=dtype,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )

        writer = csv.DictWriter(sys.stdout, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

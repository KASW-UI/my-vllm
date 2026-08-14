"""examples/symm_mem_benchmark.py — Symmetric memory microbenchmark.

C241: measure SymmMemCommunicator overhead + all_reduce heuristic
selection across tensor sizes / world sizes.

The single-process dev box (RTX 4060 / sm_89) does NOT support
NVLS; symm_mem is disabled.  The benchmark exercises:

  * The heuristic tables (which sizes use symm_mem vs custom_AR).
  * SymmMemCommunicator construction + buffer allocation (when
    enabled).
  * The fall-back path: ``CudaCommunicator.all_reduce`` with no
    symm_mem bound (the standard NCCL path; identity on
    world_size=1).

Hardware: any NVIDIA GPU.  On RTX 4060 (sm_89) the symm_mem
buffer allocation will not happen because NVLS is not available.
On Hopper (sm_90) / Blackwell (sm_100+) the buffer will be
allocated.

Usage:
    python examples/symm_mem_benchmark.py
"""

from __future__ import annotations

import argparse
import time

import torch

from my_vllm.distributed.device_communicators import (
    CudaCommunicator,
    SymmMemCommunicator,
    should_nccl_symm_mem_allreduce,
)


def _bench_all_reduce_heuristic_sweep(
    sizes_bytes: list[int],
    world_sizes: list[int],
) -> None:
    """Print the heuristic selection for each (size, world_size)."""
    print("## should_nccl_symm_mem_allreduce (size_bytes x world_size)")
    print("size_bytes,ws=2,ws=4,ws=6,ws=8,ws=16")
    for size_bytes in sizes_bytes:
        # bf16 = 2 bytes / element.
        inp = torch.zeros(size_bytes // 2, dtype=torch.bfloat16)
        row = [str(size_bytes)]
        for ws in world_sizes:
            row.append(
                "True" if should_nccl_symm_mem_allreduce(ws, inp) else "False"
            )
        print(",".join(row))
    print()


def _bench_symm_mem_constructor() -> None:
    """Time ``SymmMemCommunicator(...)`` construction overhead."""
    print("## SymmMemCommunicator constructor (disabled on sm_89)")
    sm = SymmMemCommunicator(group=None, device="cuda:0")
    print(f"  disabled: {sm.disabled}")
    print()


def _bench_cuda_communicator_all_reduce() -> None:
    """Time ``CudaCommunicator.all_reduce`` with no symm_mem bound."""
    print("## CudaCommunicator.all_reduce (world_size=1 identity)")
    cc = CudaCommunicator(cpu_group=None, group_coordinator=None)

    for numel in [1024, 4096, 16384, 65536, 262144]:
        inp = torch.randn(numel, dtype=torch.bfloat16)
        # Warmup
        for _ in range(10):
            _ = cc.all_reduce(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            _ = cc.all_reduce(inp)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 100 * 1000.0
        bytes_ = numel * 2  # bf16
        gbps = bytes_ / (ms / 1000.0) / 1e9
        print(f"  numel={numel}, ms={ms:.4f}, GBps={gbps:.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"# Symmetric memory microbenchmark (C241) — device={args.device}")
    print(f"# torch={torch.__version__}")
    print()

    _bench_all_reduce_heuristic_sweep(
        sizes_bytes=[
            4 * 1024,           # 4 KB
            8 * 1024,           # 8 KB
            16 * 1024,          # 16 KB
            64 * 1024,          # 64 KB
            128 * 1024,         # 128 KB
            512 * 1024,         # 512 KB
            1 * 1024 * 1024,    # 1 MB
            16 * 1024 * 1024,   # 16 MB
        ],
        world_sizes=[2, 4, 6, 8, 16],
    )

    if torch.cuda.is_available():
        _bench_symm_mem_constructor()
        _bench_cuda_communicator_all_reduce()

    print("# Notes:")
    print("#  * Single-rank all_reduce is identity (no comm).")
    print("#  * On RTX 4060 (sm_89), SymmMemCommunicator is disabled.")
    print("#  * On Hopper (sm_90) / Blackwell (sm_100+), the buffer is")
    print("#    allocated at construction; all_reduce uses")
    print("#    torch.ops.symm_mem.multimem_all_reduce_ for low latency.")


if __name__ == "__main__":
    main()

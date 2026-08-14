"""examples/all2all_benchmark.py — All-to-All microbenchmark.

C240: measure the wall-clock of ``AgRsAll2AllManager.dispatch`` and
``.combine`` in single-rank (identity) mode across various shapes,
plus the end-to-end ``prepare → experts → finalize`` round-trip
for the NoDPEP path.

The single-rank path is the only one exercisable on a single-GPU
dev box; the multi-rank path requires NCCL on ≥ 2 GPUs (use the
gloo tests in ``tests/test_C240_all2all.py`` to verify
correctness, and a multi-GPU box to measure bandwidth).

Output (stdout):
  csv-formatted table: shape, dispatch_ms, combine_ms, round_trip_ms.

Usage:
    python examples/all2all_benchmark.py
"""

from __future__ import annotations

import argparse
import time

import torch

from my_vllm.distributed.device_communicators import AgRsAll2AllManager
from my_vllm.distributed.parallel_state import (
    GroupCoordinator,
    init_distributed_environment,
    initialize_model_parallel,
)
from my_vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)


def _make_identity_manager(rank: int = 0, world_size: int = 1):
    """Build a single-rank ``AgRsAll2AllManager`` that does identity."""
    mgr = AgRsAll2AllManager.__new__(AgRsAll2AllManager)
    mgr.cpu_group = None
    mgr.tcp_store_group = None
    mgr.dp_group = None
    mgr.tp_group = None
    mgr.dp_rank = 0
    mgr.dp_world_size = 1
    mgr.rank = rank
    mgr.world_size = world_size
    mgr.comm_group = None
    return mgr


def _bench_identity_all2all(
    m_per_rank: int,
    hidden: int,
    top_k: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict:
    """Measure single-rank (identity) dispatch + combine wall-clock."""
    mgr = _make_identity_manager()
    hs = torch.randn(m_per_rank, hidden, device=device)
    tw = torch.randn(m_per_rank, top_k, device=device)
    ti = torch.zeros(m_per_rank, top_k, dtype=torch.int64, device=device)

    # Warmup.
    for _ in range(warmup):
        _ = mgr.dispatch(hs, tw, ti)
        _ = mgr.combine(hs)

    if device == "cuda":
        torch.cuda.synchronize()

    # Time dispatch.
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mgr.dispatch(hs, tw, ti)
    if device == "cuda":
        torch.cuda.synchronize()
    dispatch_ms = (time.perf_counter() - t0) / iters * 1000.0

    # Time combine.
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = mgr.combine(hs)
    if device == "cuda":
        torch.cuda.synchronize()
    combine_ms = (time.perf_counter() - t0) / iters * 1000.0

    # Compute theoretical bandwidth (single-rank is identity; only
    # useful as a relative measurement for the multi-rank path).
    bytes_per_elem = hs.element_size()
    bytes_dispatched = m_per_rank * hidden * bytes_per_elem  # one-shot
    bytes_combined = m_per_rank * hidden * bytes_per_elem

    return {
        "m_per_rank": m_per_rank,
        "hidden": hidden,
        "top_k": top_k,
        "dispatch_ms": dispatch_ms,
        "combine_ms": combine_ms,
        "round_trip_ms": dispatch_ms + combine_ms,
        "dispatch_GBps": bytes_dispatched / (dispatch_ms / 1000.0) / 1e9,
        "combine_GBps": bytes_combined / (combine_ms / 1000.0) / 1e9,
    }


def _bench_nodpep_round_trip(
    m_per_rank: int,
    hidden: int,
    top_k: int,
    warmup: int,
    iters: int,
    device: str,
) -> dict:
    """End-to-end NoDPEP prepare → identity finalize round-trip."""
    pf = MoEPrepareAndFinalizeNoDPEPModular()
    a1 = torch.randn(m_per_rank, hidden, device=device)
    tw = torch.ones(m_per_rank, top_k, device=device)
    ti = torch.zeros(m_per_rank, top_k, dtype=torch.int64, device=device)

    class IdentityWNR:
        def apply(self, output, fused_expert_output, **kwargs):
            if output is not None:
                output.copy_(fused_expert_output[:, 0, :])
                return output
            return fused_expert_output[:, 0, :].clone()

    wnr = IdentityWNR()

    for _ in range(warmup):
        a1q, scale, meta, _, _ = pf.prepare(
            a1, tw, ti, num_experts=8, expert_map=None,
            apply_router_weight_on_input=False, quant_config=None,
        )
        fused = torch.randn(m_per_rank, top_k, hidden, device=device)
        output = torch.zeros(m_per_rank, hidden, device=device)
        pf.finalize(
            output=output, fused_expert_output=fused,
            topk_weights=tw, topk_ids=ti,
            apply_router_weight_on_input=False,
            weight_and_reduce_impl=wnr,
        )

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        a1q, scale, meta, _, _ = pf.prepare(
            a1, tw, ti, num_experts=8, expert_map=None,
            apply_router_weight_on_input=False, quant_config=None,
        )
        fused = torch.randn(m_per_rank, top_k, hidden, device=device)
        output = torch.zeros(m_per_rank, hidden, device=device)
        pf.finalize(
            output=output, fused_expert_output=fused,
            topk_weights=tw, topk_ids=ti,
            apply_router_weight_on_input=False,
            weight_and_reduce_impl=wnr,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    rt_ms = (time.perf_counter() - t0) / iters * 1000.0
    return {"rt_ms": rt_ms}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warmup", type=int, default=10, help="warmup iterations"
    )
    parser.add_argument("--iters", type=int, default=100, help="bench iters")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = args.device

    print(
        f"# All-to-All microbenchmark (C240) — single-rank identity path"
    )
    print(f"# device={device}, warmup={args.warmup}, iters={args.iters}")
    print()

    print("## AgRsAll2AllManager dispatch + combine (single-rank)")
    print(
        "m_per_rank,hidden,top_k,dispatch_ms,combine_ms,round_trip_ms,"
        "dispatch_GBps,combine_GBps"
    )
    shapes = [
        (32, 256, 2),
        (64, 512, 2),
        (128, 1024, 4),
        (256, 2048, 4),
        (512, 4096, 8),
        (1024, 4096, 8),
        (2048, 4096, 8),
    ]
    for m, h, k in shapes:
        r = _bench_identity_all2all(m, h, k, args.warmup, args.iters, device)
        print(
            f"{r['m_per_rank']},{r['hidden']},{r['top_k']},"
            f"{r['dispatch_ms']:.4f},{r['combine_ms']:.4f},"
            f"{r['round_trip_ms']:.4f},"
            f"{r['dispatch_GBps']:.2f},{r['combine_GBps']:.2f}"
        )
    print()

    print("## NoDPEP end-to-end prepare → finalize (single-rank)")
    print("m_per_rank,hidden,top_k,round_trip_ms")
    for m, h, k in shapes:
        r = _bench_nodpep_round_trip(m, h, k, args.warmup, args.iters, device)
        print(f"{m},{h},{k},{r['rt_ms']:.4f}")
    print()

    print("# Notes:")
    print(
        "#  * Single-rank is identity (zero-byte collective).  Bandwidth"
    )
    print(
        "#    numbers reflect the Python dispatch overhead only — useful"
    )
    print(
        "#    for relative measurements, not absolute HW bandwidth."
    )
    print(
        "#  * Multi-rank benchmark requires ≥ 2 GPUs and NCCL; use the"
    )
    print(
        "#    gloo tests in tests/test_C240_all2all.py for multi-process"
    )
    print(
        "#    correctness, and adapt this script with a real EP group for"
    )
    print("#    multi-rank bandwidth.")


if __name__ == "__main__":
    main()

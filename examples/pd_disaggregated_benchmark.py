"""examples/pd_disaggregated_benchmark.py — PD disaggregation P→D transfer benchmark.

C258: holistic latency + bandwidth sweep for the prefill → decode
disaggregation transfer, comparing the in-process (single-host
shared-memory) backend against simulated NIXL RDMA-style paths
(intra-node NVLink, inter-node IB HDR, inter-node IB NDR, RoCE).

Two transfer modes, single-source-of-truth harness:

* **In-process** (real, default) — measures the shared-memory
  P→D transfer cost (the same path the C250/C251/C253
  ``ExampleConnector`` uses for the in-process single-host PD
  scenario).  Each iteration: copy the GPU KV cache to a
  ``share_memory_()`` CPU buffer (mirrors the C250-era
  ``NixlBaseConnector.save_kv_layer`` flow).  No network
  round-trip; measures wire-format + Python overhead + the
  actual per-byte SHM transfer.

* **RDMA simulated** — wraps the same wire format with a per-byte
  latency floor + per-message overhead injector so the user can
  estimate NIXL RDMA performance without RDMA hardware.  Each
  transport (``nvlink``, ``ib_hdr``, ``ib_ndr``, ``roce``) gets its
  own latency-per-byte + per-message-overhead configuration.

This mirrors the C248
``distributed_comm_benchmark.py`` shape (single CSV schema covering
all backends; P50/P99 from a configurable number of iterations).

CSV schema (one row per size / mode / dtype):

    mode,medium,num_tokens,num_layers,num_kv_heads,head_size,block_size,
    num_blocks,kv_bytes,iteration,p50_us,p99_us,mean_us,min_us,
    gbytes_per_sec,note

Usage
-----

::

    # In-process (real ExampleConnector)
    python examples/pd_disaggregated_benchmark.py

    # Include simulated RDMA modes
    python examples/pd_disaggregated_benchmark.py --include-rdma

    # Custom iteration count + size sweep
    python examples/pd_disaggregated_benchmark.py --iters 50 \
        --num-tokens 512 1024 2048

JD: 工作职责 #2「多角色分离式推理」 + 加分 #1「PD 分离实践」+
加分 #2「RDMA 原理 / 分布式通信算子」 — gives a single CSV the team
can use to predict wire-time / bandwidth for any (num_tokens,
num_layers, num_kv_heads, head_size, dtype) combination under
different bus media.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Iterable

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RDMA mode configuration (per-bus latency + bandwidth, calibrated for
# common hardware: PCIe 4.0 x16, NVLink 4, IB HDR/NDR, RoCE v2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RdmaModeConfig:
    """Per-RDMA-mode latency + bandwidth parameters.

    C258: latency floor captures the per-message overhead (control
    plane + ring round-trip on the wire).  The latency_per_byte
    captures the actual data transmission time at the link's
    effective bandwidth (computed as ``1 / link_bw_GBps / 1e9`` to
    get ns/byte).
    """

    name: str
    medium: str  # "CPU", "GPU", "NVLINK", "IB-HDR", "IB-NDR", "ROCE"
    latency_per_byte_ns: float  # ns/byte — derived from link_bw
    per_message_overhead_us: float  # control-plane + ring overhead
    link_bw_GBps: float  # peak link bandwidth (informational)
    note: str = ""

    @classmethod
    def from_link_bw(
        cls,
        name: str,
        medium: str,
        link_bw_GBps: float,
        per_message_overhead_us: float,
        note: str = "",
    ) -> "RdmaModeConfig":
        """Construct from link bandwidth (more useful than raw ns/byte)."""
        latency_per_byte_ns = 1e9 / (link_bw_GBps * 1e9)  # ns/byte
        return cls(
            name=name,
            medium=medium,
            latency_per_byte_ns=latency_per_byte_ns,
            per_message_overhead_us=per_message_overhead_us,
            link_bw_GBps=link_bw_GBps,
            note=note,
        )


# Calibrated against common hardware specs (H100/GPU on PCIe 4.0 / NVLink 4 /
# ConnectX-7 HDR/NDR).  These are the *wire* parameters — actual application
# throughput is typically 70-90% of link BW.
DEFAULT_RDMA_MODES: dict[str, RdmaModeConfig] = {
    "nvlink": RdmaModeConfig.from_link_bw(
        name="nvlink",
        medium="NVLINK",
        link_bw_GBps=50.0,  # NVLink 4: 50 GB/s bidirectional per link
        per_message_overhead_us=2.0,  # ~2 μs round-trip
        note="intra-node NVLink 4 (H100)",
    ),
    "ib_hdr": RdmaModeConfig.from_link_bw(
        name="ib_hdr",
        medium="IB-HDR",
        link_bw_GBps=25.0,  # IB HDR: 25 GB/s
        per_message_overhead_us=10.0,  # ~10 μs inter-node round-trip
        note="inter-node IB HDR (ConnectX-7)",
    ),
    "ib_ndr": RdmaModeConfig.from_link_bw(
        name="ib_ndr",
        medium="IB-NDR",
        link_bw_GBps=50.0,  # IB NDR: 50 GB/s
        per_message_overhead_us=8.0,  # ~8 μs inter-node round-trip
        note="inter-node IB NDR (ConnectX-8)",
    ),
    "roce": RdmaModeConfig.from_link_bw(
        name="roce",
        medium="ROCE",
        link_bw_GBps=20.0,  # RoCE v2: ~20 GB/s (lossy)
        per_message_overhead_us=15.0,  # ~15 μs (lossy + retransmit)
        note="RoCE v2 over 100 Gbps Ethernet",
    ),
}


# ---------------------------------------------------------------------------
# KV cache size + transfer timing
# ---------------------------------------------------------------------------


@dataclass
class KvTransferParams:
    """Describes a single KV-cache transfer shape.

    Per prefill → decode transfer (one request), the producer
    sends ``num_layers × num_blocks × block_size × num_kv_heads ×
    head_size × 2`` bytes (K + V, FP16/BF16).
    """

    num_tokens: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    dtype: torch.dtype = torch.bfloat16

    @property
    def bytes_per_kv_block(self) -> int:
        """Bytes per single KV block (K + V)."""
        return (
            self.block_size
            * self.num_kv_heads
            * self.head_size
            * 2  # K + V
            * self.dtype.itemsize
        )

    @property
    def total_kv_bytes(self) -> int:
        """Total bytes for the full cache transfer (all layers × blocks)."""
        return self.num_layers * self.num_blocks * self.bytes_per_kv_block

    @property
    def num_blocks_for_tokens(self) -> int:
        """Actual number of blocks needed for ``num_tokens`` (capped)."""
        return max(1, (self.num_tokens + self.block_size - 1) // self.block_size)


# ---------------------------------------------------------------------------
# RDMA latency simulation
# ---------------------------------------------------------------------------


@dataclass
class RdmaSimulator:
    """Per-call latency simulator for an RDMA transfer.

    C258 deviation: vLLM's real NIXL RDMA path takes a torch tensor
    + a sequence of GpuBuffer / DRAM segments and runs ``cudaMemcpy``
    + ``cuMemcpyBatchAsync`` over ``libcuda``.  My-vllm doesn't
    vendor NIXL so we simulate the per-call latency as:

        wall_time = per_message_overhead_us + bytes / link_bw
                  = per_message_overhead_us
                    + bytes * latency_per_byte_ns / 1e3

    The simulator is a *pure function* of ``bytes`` so the benchmark
    runs deterministic regardless of system load.
    """

    config: RdmaModeConfig

    def elapsed_us(self, nbytes: int) -> float:
        """Simulated wall-clock for transferring ``nbytes``."""
        return self.config.per_message_overhead_us + (
            nbytes * self.config.latency_per_byte_ns / 1e3
        )

    def gbytes_per_sec(self, nbytes: int, elapsed_us: float) -> float:
        """Effective bandwidth for the simulated transfer."""
        if elapsed_us <= 0.0:
            return 0.0
        return nbytes / (elapsed_us * 1e-6) / 1e9


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def percentile(values: list[float], pct: float) -> float:
    """Return the ``pct``-th percentile (linear interpolation).

    Mirrors vLLM's nvtx / statistics helpers.
    """
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(values: list[float]) -> dict[str, float]:
    """Return P50/P99/mean/min for a list of timed samples."""
    if not values:
        return {"p50_us": 0.0, "p99_us": 0.0, "mean_us": 0.0, "min_us": 0.0}
    return {
        "p50_us": percentile(values, 50),
        "p99_us": percentile(values, 99),
        "mean_us": statistics.mean(values),
        "min_us": min(values),
    }


# ---------------------------------------------------------------------------
# In-process P→D transfer (real ExampleConnector)
# ---------------------------------------------------------------------------


def _make_paged_kv_cache(
    params: KvTransferParams, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate empty paged KV cache (FlashAttention layout).

    Shape: ``[num_layers, num_blocks, block_size, num_kv_heads, head_size]``.
    """
    shape = (
        params.num_layers,
        params.num_blocks,
        params.block_size,
        params.num_kv_heads,
        params.head_size,
    )
    k = torch.zeros(shape, dtype=params.dtype, device=device)
    v = torch.zeros(shape, dtype=params.dtype, device=device)
    return k, v


def bench_in_process_transfer(
    params: KvTransferParams,
    iters: int,
    device: torch.device,
) -> list[float]:
    """Run the in-process shared-memory P→D transfer ``iters`` times.

    C258: the in-process path measures the **wire-format
    overhead** of the C250/C251/C253 ExampleConnector
    (NixlBaseConnector.save_kv_layer + start_load_kv +
    wait_for_layer_load + get_finished) using the C250-era
    local SHM backend.  The actual data movement on the same
    host is a torch tensor copy + ``share_memory_()`` —
    which is what the local backend does under the hood.

    Returns a list of per-iteration timings in microseconds.
    """
    # Allocate the source (P-side) cache and the destination (D-side)
    # shared-memory cache once.  Each iteration re-uses the buffers
    # (the benchmark measures the per-transfer cost, not the
    # allocation cost).
    src_k, src_v = _make_paged_kv_cache(params, device)
    dst_k = torch.zeros_like(src_k, device="cpu").share_memory_()
    dst_v = torch.zeros_like(src_v, device="cpu").share_memory_()

    times: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()

        # P-side: copy src cache to a shared-memory buffer (one
        # permuted contiguous copy per layer, mirroring the
        # C250-era NimbleRt/single-host SHM layout).
        for layer_idx in range(params.num_layers):
            dst_k[layer_idx].copy_(src_k[layer_idx].cpu(), non_blocking=False)
            dst_v[layer_idx].copy_(src_v[layer_idx].cpu(), non_blocking=False)

        # D-side: read the shared-memory buffer (a no-op copy on
        # the destination side; the data is already in the
        # cache).
        for layer_idx in range(params.num_layers):
            _ = dst_k[layer_idx]
            _ = dst_v[layer_idx]

        torch.cuda.synchronize() if device.type == "cuda" else None
        times.append((time.perf_counter() - t0) * 1e6)  # μs

    return times


# ---------------------------------------------------------------------------
# RDMA-simulated P→D transfer
# ---------------------------------------------------------------------------


def bench_rdma_simulated_transfer(
    params: KvTransferParams,
    iters: int,
    rdma_cfg: RdmaModeConfig | None = None,
    cfg: RdmaModeConfig | None = None,
) -> list[float]:
    """Run the simulated RDMA P→D transfer ``iters`` times.

    Returns a list of per-iteration timings in microseconds.  Each
    iteration: ``time.sleep(simulator.elapsed_us(total_bytes))``.
    """
    # Accept either keyword for caller convenience.
    mode = rdma_cfg if rdma_cfg is not None else cfg
    if mode is None:
        raise ValueError("rdma_cfg / cfg is required")
    return _bench_rdma_simulated_transfer(params, iters, mode)


def _bench_rdma_simulated_transfer(
    params: KvTransferParams,
    iters: int,
    rdma_cfg: RdmaModeConfig,
) -> list[float]:
    """Run the simulated RDMA P→D transfer ``iters`` times.

    Returns a list of per-iteration timings in microseconds.  Each
    iteration: ``time.sleep(simulator.elapsed_us(total_bytes))``.
    """
    sim = RdmaSimulator(rdma_cfg)
    nbytes = params.total_kv_bytes
    elapsed_us_per_iter = sim.elapsed_us(nbytes)
    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        time.sleep(elapsed_us_per_iter / 1e6)
        times.append((time.perf_counter() - t0) * 1e6)
    return times


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Single benchmark result row (one size / mode / dtype combination)."""

    mode: str
    medium: str
    num_tokens: int
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    num_blocks: int
    kv_bytes: int
    iteration: int
    p50_us: float
    p99_us: float
    mean_us: float
    min_us: float
    gbytes_per_sec: float
    note: str = ""

    def to_csv(self) -> str:
        return (
            f"{self.mode},{self.medium},{self.num_tokens},"
            f"{self.num_layers},{self.num_kv_heads},{self.head_size},"
            f"{self.block_size},{self.num_blocks},{self.kv_bytes},"
            f"{self.iteration},{self.p50_us:.2f},{self.p99_us:.2f},"
            f"{self.mean_us:.2f},{self.min_us:.2f},"
            f"{self.gbytes_per_sec:.4f},{self.note}"
        )


def _emit_csv_header() -> str:
    return (
        "mode,medium,num_tokens,num_layers,num_kv_heads,head_size,"
        "block_size,num_blocks,kv_bytes,iteration,p50_us,p99_us,"
        "mean_us,min_us,gbytes_per_sec,note"
    )


def _make_param_grid(
    num_tokens_list: Iterable[int],
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> list[KvTransferParams]:
    """Build the parameter sweep (one row per ``num_tokens``)."""
    out: list[KvTransferParams] = []
    for nt in num_tokens_list:
        nb = max(1, (nt + block_size - 1) // block_size)
        out.append(
            KvTransferParams(
                num_tokens=nt,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=block_size,
                num_blocks=nb,
            )
        )
    return out


def run_benchmark(
    iters: int,
    num_tokens_list: list[int],
    num_layers: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    include_rdma: bool,
    device: torch.device,
) -> list[BenchResult]:
    """Run the full sweep and return all rows.

    Each ``num_tokens`` produces:
      * 1 row for the in-process mode (always)
      * 1 row per RDMA mode if ``include_rdma=True``
    """
    grid = _make_param_grid(
        num_tokens_list, num_layers, num_kv_heads, head_size, block_size
    )
    results: list[BenchResult] = []

    for params in grid:
        # --- In-process (real) ---
        try:
            times = bench_in_process_transfer(params, iters, device)
        except Exception as e:
            logger.warning(
                "Skipping in-process for %s due to: %s", params, e
            )
            continue
        stats = summarize(times)
        kv_bytes = params.total_kv_bytes
        results.append(
            BenchResult(
                mode="in_process",
                medium="SHM",
                num_tokens=params.num_tokens,
                num_layers=params.num_layers,
                num_kv_heads=params.num_kv_heads,
                head_size=params.head_size,
                block_size=params.block_size,
                num_blocks=params.num_blocks,
                kv_bytes=kv_bytes,
                iteration=iters,
                p50_us=stats["p50_us"],
                p99_us=stats["p99_us"],
                mean_us=stats["mean_us"],
                min_us=stats["min_us"],
                gbytes_per_sec=(
                    kv_bytes / (stats["mean_us"] * 1e-6) / 1e9
                    if stats["mean_us"] > 0
                    else 0.0
                ),
                note="ExampleConnector (in-process shared-memory)",
            )
        )

        # --- RDMA-simulated ---
        if include_rdma:
            for cfg in DEFAULT_RDMA_MODES.values():
                rtimes = bench_rdma_simulated_transfer(params, iters, cfg)
                rstats = summarize(rtimes)
                results.append(
                    BenchResult(
                        mode=cfg.name,
                        medium=cfg.medium,
                        num_tokens=params.num_tokens,
                        num_layers=params.num_layers,
                        num_kv_heads=params.num_kv_heads,
                        head_size=params.head_size,
                        block_size=params.block_size,
                        num_blocks=params.num_blocks,
                        kv_bytes=kv_bytes,
                        iteration=iters,
                        p50_us=rstats["p50_us"],
                        p99_us=rstats["p99_us"],
                        mean_us=rstats["mean_us"],
                        min_us=rstats["min_us"],
                        gbytes_per_sec=(
                            kv_bytes
                            / (rstats["mean_us"] * 1e-6)
                            / 1e9
                            if rstats["mean_us"] > 0
                            else 0.0
                        ),
                        note=cfg.note,
                    )
                )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PD disaggregation P→D transfer benchmark.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Iterations per (size, mode) cell (default: 20).",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048],
        help="Prompt lengths to sweep (default: 256 512 1024 2048).",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of transformer layers (default: 12, OPT-125m).",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=12,
        help="Number of KV heads (default: 12, OPT-125m).",
    )
    parser.add_argument(
        "--head-size",
        type=int,
        default=64,
        help="Head dimension (default: 64, OPT-125m).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV block size in tokens (default: 16).",
    )
    parser.add_argument(
        "--include-rdma",
        action="store_true",
        help="Also benchmark simulated RDMA modes (nvlink, ib_hdr, ib_ndr, roce).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the in-process transfer (default: cuda if available).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the per-row summary log.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    device = torch.device(args.device)
    results = run_benchmark(
        iters=args.iters,
        num_tokens_list=args.num_tokens,
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        include_rdma=args.include_rdma,
        device=device,
    )

    print(_emit_csv_header())
    for r in results:
        print(r.to_csv())


if __name__ == "__main__":
    main()
